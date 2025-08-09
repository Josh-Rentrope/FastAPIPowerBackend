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
#from clerk_fastapi import ClerkMiddleware, get_auth

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

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- FastAPI App Initialization & Lifespan Events ---
app = FastAPI(
    title="Augmented Clinical Trials API",
    description="API for querying clinical trials and supporting real-time chat.",
    version="1.2.0",
    # Add Clerk Middleware for authentication
    # middleware=[
    #     {
    #         "middleware": ClerkMiddleware,
    #         "params": {
    #             "secret_key": CLERK_SECRET_KEY
    #         }
    #     }
    # ]
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
        
        # Pydantic V1 compatible validation
        final_studies = [Study.parse_obj(study_data) for study_data in studies_dict.values()]
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
    
    # NOTE: The full implementation of this endpoint is omitted for brevity but should be retained.
    return {"total": 0, "page": page, "limit": limit, "data": []} # Placeholder


"""
Get a list of clinical trials or clusters for a given map view.
"""
@main_router.get("/studies/map-data", response_model=MapResponse)
def get_map_data(
    bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    zoom: int = Query(..., description="Current map zoom level."),
    db=Depends(get_db)
):
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
        params = {'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid bbox format.")

    # Define a zoom level at which we switch from clusters to individual points
    CLUSTER_THRESHOLD = 12

    if zoom < CLUSTER_THRESHOLD:
        # --- STRATEGY FOR CLUSTERING (ZOOMED OUT) ---
        # This SQL is a conceptual example. It needs to be adapted to your schema.
        # It uses a grid-based approach to group points.
        cluster_query = text("""
            -- CONCEPTUAL QUERY FOR SERVER-SIDE CLUSTERING
            -- Note: 'locations' is the table with your point geometries.
            -- The grid precision (e.g., 4) needs to be tuned based on zoom levels.
            WITH grid AS (
                SELECT
                    ST_SnapToGrid(geom, :min_lon, :min_lat, (:max_lon - :min_lon) / 10, (:max_lat - :min_lat) / 10) as grid_point,
                    COUNT(*) as point_count
                FROM locations
                WHERE geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
                GROUP BY grid_point
            )
            SELECT
                point_count,
                ST_Y(grid_point) as latitude,
                ST_X(grid_point) as longitude
            FROM grid;
        """)
        results = db.execute(cluster_query, params).mappings().all()
        return [Cluster(**row) for row in results]

    else:
        # --- STRATEGY FOR INDIVIDUAL POINTS (ZOOMED IN) ---
        # This is your existing query to fetch individual studies within the bbox
        # It has been simplified for this example.
        point_query = text("""
            SELECT pd.nct_number, pd.study_title, l.location_text, ST_Y(l.geom) as latitude, ST_X(l.geom) as longitude
            FROM processed_data pd
            JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
            JOIN locations l ON sl.location_id = l.id
            WHERE ST_Intersects(l.geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326));
        """)
        results = db.execute(point_query, params).mappings().all()
        # This part requires logic to group locations by study if a study can have multiple locations
        # For simplicity, we assume one location per study in this example response.
        return [
            Study(nct_number=row['nct_number'], study_title=row['study_title'], locations=[StudyLocation(**row)])
            for row in results
        ]

# --- NEW: Webhook Endpoint for Clerk User Sync ---
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

# --- NEW: Chat API Endpoints ---
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])

@chat_router.get("/conversations", response_model=List[Conversation])
def get_user_conversations(db=Depends(get_db), auth: dict = {}): #  Depends(get_auth())):
    user_id = auth['sub']
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
def create_conversation(req_body: CreateConversationRequest, db=Depends(get_db), auth: dict = {}): #  Depends(get_auth())):
    creator_id = auth['sub']
    participant_ids = list(set(req_body.participant_ids + [creator_id]))

    # Transaction to create conversation and add participants
    with db.begin():
        # Create conversation
        conv_query = text("INSERT INTO conversations DEFAULT VALUES RETURNING id, created_at, updated_at;")
        new_conv = db.execute(conv_query).mappings().one()
        
        # Add participants
        part_data = [{"conv_id": new_conv['id'], "user_id": pid} for pid in participant_ids]
        part_query = text("INSERT INTO participants (conversation_id, user_id) VALUES (:conv_id, :user_id);")
        db.execute(part_query, part_data)

    return new_conv


@chat_router.get("/conversations/{conv_id}/messages", response_model=List[Message])
def get_conversation_messages(conv_id: str, db=Depends(get_db), auth: dict  = {}): #  Depends(get_auth())):
    user_id = auth['sub']
    
    # First, verify the user is a participant
    check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
    is_participant = db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none()
    if not is_participant:
        raise HTTPException(status_code=403, detail="Not a participant of this conversation")

    # Fetch messages
    msg_query = text("""
        SELECT id, conversation_id, sender_id, content, sent_at
        FROM messages WHERE conversation_id = :conv_id ORDER BY sent_at ASC;
    """)
    results = db.execute(msg_query, {'conv_id': conv_id}).mappings().all()
    return results

@chat_router.post("/conversations/{conv_id}/messages", response_model=Message, status_code=201)
def post_message(conv_id: str, request_body: dict, db=Depends(get_db), auth: dict  = {}): #  Depends(get_auth())):
    user_id = auth['sub']
    content = request_body.get('content')
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    # Transaction to check participation and send message
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

        # Update the conversation's updated_at timestamp
        update_conv_query = text("UPDATE conversations SET updated_at = now() WHERE id = :conv_id;")
        db.execute(update_conv_query, {'conv_id': conv_id})

    return new_msg

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