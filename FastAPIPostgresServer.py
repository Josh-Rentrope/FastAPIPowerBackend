import os
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, ProgrammingError

# --- Clerk and Webhook Imports ---
from svix import Webhook, WebhookVerificationError

from clerk_backend_api import Clerk
from clerk_backend_api.models.user import User
#from clerk_backend_api.jwks_helpers import AuthenticateRequestOptions


# --- Database Configuration ---
# Load credentials from environment variables
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "clinical_trials")

# --- NEW: Clerk Configuration ---
# It's crucial to set these in your environment for security
CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
CLERK_WEBHOOK_SECRET = os.environ.get("CLERK_WEBHOOK_SECRET")

if CLERK_SECRET_KEY:
    #clerk_client = Clerk(secret_key=CLERK_SECRET_KEY)
    clerk_client = Clerk()
else:
    clerk_client = None



DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- FastAPI App Initialization & Lifespan Events ---
app = FastAPI(
    title="Augmented Clinical Trials API",
    description="API for querying clinical trials and supporting real-time chat.",
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://127.0.0.1:3000", "http://localhost:3000", "https://yourfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_db_client():
    """Create the database connection pool on application startup."""
    print(f"Connecting to database at: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    try:
        app.state.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        app.state.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=app.state.engine)
        with app.state.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database connection successful.")
    except OperationalError as e:
        print(f"FATAL: Could not connect to the database. Error: {e}")
        app.state.engine = None
        app.state.SessionLocal = None

@app.on_event("shutdown")
async def shutdown_db_client():
    """Dispose of the database connection pool on application shutdown."""
    if hasattr(app.state, 'engine') and app.state.engine:
        app.state.engine.dispose()
        print("Database connection pool closed.")

# --- Pydantic Models ---

class StudyLocation(BaseModel):
    location_text: str
    latitude: float
    longitude: float

class Study(BaseModel):
    nct_number: str
    study_title: str
    study_url: str
    status: Optional[str] = None
    conditions: Optional[str] = None
    phase: Optional[str] = None
    enrollment_count: Optional[int] = None
    locations: List[StudyLocation] = []
    
    class Config:
        from_attributes = True

class Cluster(BaseModel):
    is_cluster: bool = True
    point_count: int
    latitude: float
    longitude: float

MapResponse = List[Union[Cluster, Study]]

class PaginatedStudies(BaseModel):
    total: int
    page: int
    limit: int
    data: List[Study]

# --- NEW: Chat Pydantic Models ---
class Message(BaseModel):
    id: str
    conversation_id: str
    sender_id: str
    content: str
    sent_at: str

class Conversation(BaseModel):
    id: str
    created_at: str
    updated_at: str

class CreateConversationRequest(BaseModel):
    participant_ids: List[str]

# --- Dependency for Database Session ---
def get_db(request: Request):
    if not request.app.state.SessionLocal:
        raise HTTPException(status_code=503, detail="Database connection is not configured or has failed.")
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Dependency using clerk-backend-api ---
async def get_current_user(request: Request) -> User:
    """
    Dependency to authenticate a request with Clerk and return the user object.
    Raises an HTTPException if the user is not authenticated.
    """
    if not clerk_client:
        raise HTTPException(status_code=500, detail="Clerk client not configured.")
    try:
        # The 'authorization' header (e.g., 'Bearer <token>') is automatically extracted by the SDK
        request_state = clerk_client.authenticate_request(request)
        if not request_state.is_signed_in or not request_state.user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return request_state.user
    except Exception as e:
        # Catches potential issues with token verification (e.g., expired, invalid)
        raise HTTPException(status_code=401, detail=f"Authentication failed: {e}")




# --- Main API Endpoints ---
main_router = APIRouter()

@main_router.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Augmented Clinical Trials API. Visit /docs for documentation."}

@main_router.get("/studies", response_model=PaginatedStudies)
def get_studies(db=Depends(get_db), q: Optional[str] = None, status: Optional[str] = None, phase: Optional[str] = None, bbox: Optional[str] = None, page: int = 1, limit: int = 10):
    # This function's implementation remains the same
    offset = (page - 1) * limit
    where_clauses, params = [], {}
    if q:
        where_clauses.append("(pd.study_title ILIKE :q OR pd.conditions ILIKE :q)")
        params['q'] = f"%{q}%"
    if status:
        where_clauses.append("pd.status ILIKE :status")
        params['status'] = f"%{status}%"
    if phase:
        where_clauses.append("pd.phase ILIKE :phase")
        params['phase'] = f"%{phase}%"
    if bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
            # This clause ensures that we only consider studies that have at least one location within the bbox
            where_clauses.append("EXISTS (SELECT 1 FROM study_locations sl_sub JOIN locations l_sub ON sl_sub.location_id = l_sub.id WHERE sl_sub.study_nct_number = pd.nct_number AND ST_Intersects(l_sub.geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)))")
            params.update({'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat})
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid bbox format. Use 'min_lon,min_lat,max_lon,max_lat'.")

    # --- Construct the query parts ---
    base_query = "FROM processed_data pd"
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    try:
        # --- 1. Get the total count of matching studies ---
        count_query = text(f"SELECT COUNT(pd.nct_number) {base_query} {where_sql}")
        total_count = db.execute(count_query, params).scalar_one()

        if total_count == 0:
            return {"total": 0, "page": page, "limit": limit, "data": []}

        # --- 2. Get the paginated list of study NCT numbers ---
        ids_query = text(f"""
            SELECT pd.nct_number
            {base_query}
            {where_sql}
            ORDER BY pd.last_updated_date DESC NULLS LAST
            LIMIT :limit OFFSET :offset
        """)
        params.update({'limit': limit, 'offset': offset})
        paginated_ids_result = db.execute(ids_query, params).fetchall()
        paginated_nct_ids = [row[0] for row in paginated_ids_result]

        if not paginated_nct_ids:
            return {"total": total_count, "page": page, "limit": limit, "data": []}

        # --- 3. Fetch the full details for the paginated studies and their locations ---
        details_query = text("""
            SELECT
                pd.nct_number, pd.study_title, pd.study_url, pd.status, pd.conditions, pd.phase, pd.enrollment_count,
                l.location_text,
                ST_Y(l.geom) as latitude,
                ST_X(l.geom) as longitude
            FROM processed_data pd
            LEFT JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
            LEFT JOIN locations l ON sl.location_id = l.id
            WHERE pd.nct_number IN :nct_ids
        """)
        details_result = db.execute(details_query, {'nct_ids': tuple(paginated_nct_ids)}).mappings().all()

        # --- 4. Structure the data for the response model ---
        studies_dict = {}
        for row in details_result:
            nct_id = row['nct_number']
            if nct_id not in studies_dict:
                studies_dict[nct_id] = {**row, "locations": []}
            if row['location_text']:
                studies_dict[nct_id]['locations'].append({
                    "location_text": row['location_text'],
                    "latitude": row['latitude'],
                    "longitude": row['longitude']
                })
        
        final_studies = [Study.model_validate(study_data) for study_data in studies_dict.values()]
        final_studies.sort(key=lambda s: paginated_nct_ids.index(s.nct_number))

        return {
            "total": total_count,
            "page": page,
            "limit": limit,
            "data": final_studies
        }
    except ProgrammingError as e:
        print(f"An SQL error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while querying the database.")



@main_router.get("/studies/map-data", response_model=MapResponse)
def get_map_data(
    db=Depends(get_db),
    bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    zoom: int = Query(..., description="Current map zoom level."),
    q: Optional[str] = None,
    status: Optional[str] = None,
    phase: Optional[str] = None
):
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid bbox format.")

    # --- 1. Build a dynamic WHERE clause and parameters for filtering ---
    where_clauses, params = [], {}
    # Bbox is always required for map data, applied to the locations table (aliased as 'l')
    where_clauses.append("l.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)")
    params.update({'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat})

    # Add optional filters, applied to the processed_data table (aliased as 'pd')
    if q:
        where_clauses.append("(pd.study_title ILIKE :q OR pd.conditions ILIKE :q)")
        params['q'] = f"%{q}%"
    if status:
        where_clauses.append("pd.status ILIKE :status")
        params['status'] = f"%{status}%"
    if phase:
        where_clauses.append("pd.phase ILIKE :phase")
        params['phase'] = f"%{phase}%"
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    # --- 2. Define a zoom threshold to switch between clusters and points ---
    CLUSTER_THRESHOLD = 12

    try:
        if zoom < CLUSTER_THRESHOLD:
            # --- STRATEGY FOR CLUSTERING (ZOOMED OUT) ---
            # This query first filters the studies, then groups their locations into a grid for clustering.
            cluster_query = text(f"""
                WITH filtered_locations AS (
                    SELECT l.geom
                    FROM locations l
                    JOIN study_locations sl ON l.id = sl.location_id
                    JOIN processed_data pd ON sl.study_nct_number = pd.nct_number
                    {where_sql}
                ),
                grid AS (
                    SELECT
                        ST_SnapToGrid(geom, :min_lon, :min_lat, (:max_lon - :min_lon) / 10, (:max_lat - :min_lat) / 10) as grid_point,
                        COUNT(*) as point_count
                    FROM filtered_locations
                    GROUP BY grid_point
                )
                SELECT
                    point_count,
                    ST_Y(grid_point) as latitude,
                    ST_X(grid_point) as longitude
                FROM grid WHERE grid_point IS NOT NULL;
            """)
            results = db.execute(cluster_query, params).mappings().all()
            return [Cluster(**row) for row in results]

        else:
            # --- STRATEGY FOR INDIVIDUAL POINTS (ZOOMED IN) ---
            # This query fetches individual studies that match the filters and are within the bbox.
            point_query = text(f"""
                SELECT
                    pd.nct_number, pd.study_title, pd.study_url,
                    l.location_text,
                    ST_Y(l.geom) as latitude,
                    ST_X(l.geom) as longitude
                FROM processed_data pd
                JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
                JOIN locations l ON sl.location_id = l.id
                {where_sql};
            """)
            results = db.execute(point_query, params).mappings().all()

            # Structure the data for the response model, grouping locations by study
            studies_dict = {}
            for row in results:
                nct_id = row['nct_number']
                if nct_id not in studies_dict:
                    studies_dict[nct_id] = {
                        "nct_number": nct_id,
                        "study_title": row['study_title'],
                        "study_url": row.get('study_url', ''),
                        "locations": []
                    }
                studies_dict[nct_id]['locations'].append({
                    "location_text": row['location_text'],
                    "latitude": row['latitude'],
                    "longitude": row['longitude']
                })
            
            # Use model_validate (Pydantic v2) or parse_obj (v1)
            return [Study.model_validate(study_data) for study_data in studies_dict.values()]
    except ProgrammingError as e:
        print(f"An SQL error occurred in get_map_data: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching map data.")




# --- Webhook Endpoint for Clerk User Sync ---
webhook_router = APIRouter()

@webhook_router.post("/api/webhooks/clerk-events")
async def clerk_webhook(request: Request, db=Depends(get_db)):
    if not CLERK_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret is not configured.")
    
    headers = request.headers
    try:
        payload_bytes = await request.body()
        payload_str = payload_bytes.decode("utf-8")
        wh = Webhook(CLERK_WEBHOOK_SECRET)
        event = wh.verify(payload_str, headers)
    except WebhookVerificationError as e:
        print(f"Webhook verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    event_type = event['type']
    data = event['data']

    if event_type in ['user.created', 'user.updated']:
        clerk_user_id = data['id']
        # Clerk stores role in public_metadata. Adapt if you use a different field.
        role = data.get('public_metadata', {}).get('role', None)

        # Use an "upsert" operation to handle both create and update
        upsert_query = text("""
            INSERT INTO users (clerk_user_id, role)
            VALUES (:clerk_user_id, :role)
            ON CONFLICT (clerk_user_id) DO UPDATE SET
                role = EXCLUDED.role;
        """)
        db.execute(upsert_query, {'clerk_user_id': clerk_user_id, 'role': role})
        db.commit()
        print(f"Processed '{event_type}' for user {clerk_user_id}")
        
    return {"status": "success"}

# --- Chat API Endpoints ---
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])

@chat_router.get("/conversations", response_model=List[Conversation])
def get_user_conversations(db=Depends(get_db), user: User = Depends(get_current_user)): # <--- MODIFICATION
    user_id = user.id
    query = text("""
        SELECT c.id, c.created_at, c.updated_at
        FROM conversations c
        JOIN participants p ON c.id = p.conversation_id
        WHERE p.user_id = :user_id
        ORDER BY c.updated_at DESC;
    """)
    results = db.execute(query, {'user_id': user_id}).mappings().all()
    return results

@chat_router.post("/conversations", response_model=Conversation, status_code=201)
def create_conversation(req_body: CreateConversationRequest, db=Depends(get_db), user: User = Depends(get_current_user)): # <--- MODIFICATION
    creator_id = user.id
    # Ensure creator is always in the conversation
    participant_ids = list(set(req_body.participant_ids + [creator_id]))

    with db.begin():
        conv_query = text("INSERT INTO conversations DEFAULT VALUES RETURNING id, created_at, updated_at;")
        new_conv = db.execute(conv_query).mappings().one()
        
        part_data = [{"conv_id": new_conv['id'], "user_id": pid} for pid in participant_ids]
        part_query = text("INSERT INTO participants (conversation_id, user_id) VALUES (:conv_id, :user_id);")
        db.execute(part_query, part_data)

    return new_conv


@chat_router.get("/conversations/{conv_id}/messages", response_model=List[Message])
def get_conversation_messages(conv_id: str, db=Depends(get_db), user: User = Depends(get_current_user)): # <--- MODIFICATION
    user_id = user.id
    
    check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
    is_participant = db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none()
    if not is_participant:
        raise HTTPException(status_code=403, detail="Not a participant of this conversation")

    msg_query = text("""
        SELECT id, conversation_id, sender_id, content, sent_at
        FROM messages WHERE conversation_id = :conv_id ORDER BY sent_at ASC;
    """)
    results = db.execute(msg_query, {'conv_id': conv_id}).mappings().all()
    return results

@chat_router.post("/conversations/{conv_id}/messages", response_model=Message, status_code=201)
def post_message(conv_id: str, request_body: dict, db=Depends(get_db), user: User = Depends(get_current_user)): # <--- MODIFICATION
    user_id = user.id
    content = request_body.get('content')
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    with db.begin():
        check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
        is_participant = db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none()
        if not is_participant:
            raise HTTPException(status_code=403, detail="Not a participant of this conversation")
        
        msg_query = text("""
            INSERT INTO messages (conversation_id, sender_id, content)
            VALUES (:conv_id, :sender_id, :content)
            RETURNING id, conversation_id, sender_id, content, sent_at;
        """)
        new_msg = db.execute(msg_query, {'conv_id': conv_id, 'sender_id': user_id, 'content': content}).mappings().one()

        update_conv_query = text("UPDATE conversations SET updated_at = now() WHERE id = :conv_id;")
        db.execute(update_conv_query, {'conv_id': conv_id})

    return new_msg

@chat_router.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, db=Depends(get_db), user: User = Depends(get_current_user)):
    user_id = user.id
    
    with db.begin():
        # Step 1: Verify the user is a participant
        check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
        is_participant = db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none()
        
        if not is_participant:
            raise HTTPException(status_code=403, detail="You do not have permission to delete this conversation.")
        
        # Step 2: Delete associated messages (assuming ON DELETE CASCADE is not set)
        delete_messages_query = text("DELETE FROM messages WHERE conversation_id = :conv_id")
        db.execute(delete_messages_query, {'conv_id': conv_id})

        # Step 3: Delete participants
        delete_participants_query = text("DELETE FROM participants WHERE conversation_id = :conv_id")
        db.execute(delete_participants_query, {'conv_id': conv_id})

        # Step 4: Delete the conversation itself
        delete_conv_query = text("DELETE FROM conversations WHERE id = :conv_id")
        result = db.execute(delete_conv_query, {'conv_id': conv_id})

        if result.rowcount == 0:
            # This case might occur in a race condition, but it's good practice to handle.
            raise HTTPException(status_code=404, detail="Conversation not found.")

    return Response(status_code=status.HTTP_204_NO_CONTENT)



# --- Register all the routers ---
app.include_router(main_router)
app.include_router(webhook_router)
app.include_router(chat_router)


if __name__ == "__main__":
    if not CLERK_SECRET_KEY or not CLERK_WEBHOOK_SECRET:
        print("--- WARNING: Clerk environment variables are not set. Auth and webhooks will fail. ---")
    print("--- Starting Augmented Clinical Trials API Server ---")
    print("API documentation will be available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)