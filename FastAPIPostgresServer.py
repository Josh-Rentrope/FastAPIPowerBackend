import os
import json
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query, Request, APIRouter, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, ProgrammingError

# --- Clerk and Webhook Imports ---
from svix import Webhook, WebhookVerificationError

# --- Clerk auth libraries commented out as requested ---
# from clerk_backend_api import Clerk
# from clerk_backend_api.models.user import User
# from clerk_backend_api.jwks_helpers import AuthenticateRequestOptions

import openai # Using OpenAI as an example provider

from api.countydata import router as county_data_router

# --- Database Configuration ---
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "clinical_trials")

# --- Clerk Configuration commented out ---
# CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY")
CLERK_WEBHOOK_SECRET = os.environ.get("CLERK_WEBHOOK_SECRET")

# if CLERK_SECRET_KEY:
#     clerk_client = Clerk(bearer_auth=CLERK_SECRET_KEY)
# else:
#     clerk_client = None


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

# --- Chat Pydantic Models ---
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

# --- NEW: Insecure dependency for demo purposes ---
async def get_user_email_from_header(x_user_email: str = Header(...)) -> str:
    """
    INSECURE: This dependency gets the user's identity from a custom request header.
    This is for demo purposes only and should be replaced with proper authentication.
    """
    if not x_user_email:
        raise HTTPException(status_code=400, detail="X-User-Email header is missing.")
    return x_user_email


# --- Main API Endpoints ---
main_router = APIRouter()

# ... (get_studies and get_map_data endpoints remain the same) ...
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
# This can remain as is, since it's for syncing users to your DB from Clerk events.
webhook_router = APIRouter()
@webhook_router.post("/api/webhooks/clerk-events")
async def clerk_webhook(request: Request, db=Depends(get_db)):
    if not CLERK_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret is not configured.")
    # ... (webhook logic) ...
    
# --- Chat API Endpoints (Modified for Demo Auth) ---
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])

@chat_router.get("/conversations", response_model=List[Conversation])
def get_user_conversations(db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email # Use email as the user identifier
    query = text("""
        SELECT c.id, c.created_at, c.updated_at
        FROM conversations c
        JOIN participants p ON c.id = p.conversation_id
        WHERE p.user_id = :user_id
        ORDER BY c.updated_at DESC;
    """)
    results = db.execute(query, {'user_id': user_id}).mappings().all()
    returnResults = []
    for result in results:
        returnResult = {}
        for key in result:
            returnResult[key] = str(result[key])
        returnResults.append(returnResult)

    return returnResults

@chat_router.post("/conversations/create", response_model=Conversation, status_code=201)
def create_conversation(req_body: CreateConversationRequest, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):

    creator_id = email # Use email as the user identifier
    participant_ids = list(set(req_body.participant_ids + [creator_id]))
  

    with db.begin():

    
        upsert_user_query = text("""
            INSERT INTO users (clerk_user_id) VALUES (:user_id)
            ON CONFLICT (clerk_user_id) DO NOTHING;
        """)
        for user_id in participant_ids:
            db.execute(upsert_user_query, {"user_id": user_id})

        conv_query = text("INSERT INTO conversations DEFAULT VALUES RETURNING id, created_at, updated_at;")
        new_conv = db.execute(conv_query).mappings().one()
        part_data = [{"conv_id": new_conv['id'], "user_id": pid} for pid in participant_ids]
        part_query = text("INSERT INTO participants (conversation_id, user_id) VALUES (:conv_id, :user_id);")
        db.execute(part_query, part_data)

        
    return_new_conv = {}

    for key in new_conv:
        return_new_conv[key] = str(new_conv[key])
    return return_new_conv

@chat_router.get("/conversations/{conv_id}/messages", response_model=List[Message])
def get_conversation_messages(conv_id: str, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email # Use email as the user identifier
    check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
    is_participant = db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none()
    if not is_participant:
        raise HTTPException(status_code=403, detail="Not a participant of this conversation")
    msg_query = text("SELECT id, conversation_id, sender_id, content, sent_at FROM messages WHERE conversation_id = :conv_id ORDER BY sent_at ASC;")
    results = db.execute(msg_query, {'conv_id': conv_id}).mappings().all()
    
    returnResults = []
    for result in results:
        returnResult = {}
        for key in result:
            returnResult[key] = str(result[key])
        returnResults.append(returnResult)

    return returnResults

@chat_router.post("/conversations/{conv_id}/messages", response_model=Message, status_code=201)
@chat_router.post("/conversations/{conv_id}/messages", response_model=Message, status_code=201)
def post_message(conv_id: str, request_body: dict, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email
    content = request_body.get('content')
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    with db.begin():
        # --- Authentication and User Message Saving (Same as before) ---
        check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
        # ... (participant check logic) ...
        
        # 1. Save the user's message
        user_msg_query = text("INSERT INTO messages (conversation_id, sender_id, content) VALUES (:conv_id, :sender_id, :content) RETURNING id, conversation_id, sender_id, content, sent_at;")
        db.execute(user_msg_query, {'conv_id': conv_id, 'sender_id': user_id, 'content': content})

        # 2. Fetch recent conversation history for LLM context
        history_query = text("SELECT sender_id, content FROM messages WHERE conversation_id = :conv_id ORDER BY sent_at ASC LIMIT 10;")
        history_results = db.execute(history_query, {'conv_id': conv_id}).mappings().all()
        
        messages_for_llm = [{"role": "system", "content": "You are a clinical trial discovery expert. When a user asks you to find trials, use the search_clinical_trials tool. Summarize the results for the user and include the NCT numbers."}]
        for msg in history_results:
            messages_for_llm.append({"role": "user" if msg['sender_id'] == user_id else "assistant", "content": msg['content']})

        # 3. Send the query to the LLM


        llm_response = llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            tools=available_tools
        )   #.chat(model="gpt-4", messages=messages_for_llm, ) # Simplified for brevity

        # 4 & 5. Check if the LLM wants to use a tool
        if llm_response.choices[0].message.tool_calls:
            tool_call = llm_response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tool_output = available_tools[function_name](db=db, **function_args)
            
            # Instead of a hardcoded string, dynamically generate the response from the tool's output.
            studies = tool_output.get("studies", [])
            if studies:
                response_parts = ["I found the following trials for you:"]
                for study in studies:
                    response_parts.append(f"- {study['study_title']} ({study['nct_number']})")
                final_content = "\n".join(response_parts)
            else:
                final_content = "I couldn't find any clinical trials matching your request. Please try a different search."

        else:
            final_content = llm_response.get("choices")[0].get("message").get("content")

        upsert_ai_query = text("""
            INSERT INTO users (clerk_user_id) VALUES (:user_id)
            ON CONFLICT (clerk_user_id) DO NOTHING;
        """)
        db.execute(upsert_ai_query, {"user_id": "00000000-0000-0000-0000-000000000000"})


        # 6. Save the AI's final response to the database
        ai_msg_query = text("INSERT INTO messages (conversation_id, sender_id, content) VALUES (:conv_id, '00000000-0000-0000-0000-000000000000', :content) RETURNING id, conversation_id, sender_id, content, sent_at;")
        new_ai_msg = db.execute(ai_msg_query, {'conv_id': conv_id, 'content': final_content}).mappings().one()

        # Update conversation timestamp
        db.execute(text("UPDATE conversations SET updated_at = now() WHERE id = :conv_id;"), {'conv_id': conv_id})

        # ... (return the new AI message) ...
        return_new_msg = {}

        for key in new_ai_msg:
            return_new_msg[key] = str(new_ai_msg[key])

    return return_new_msg
    return new_ms

@chat_router.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email # Use email as the user identifier
    with db.begin():
        check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
        is_participant = db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none()
        if not is_participant:
            raise HTTPException(status_code=403, detail="You do not have permission to delete this conversation.")
        db.execute(text("DELETE FROM messages WHERE conversation_id = :conv_id"), {'conv_id': conv_id})
        db.execute(text("DELETE FROM participants WHERE conversation_id = :conv_id"), {'conv_id': conv_id})
        result = db.execute(text("DELETE FROM conversations WHERE id = :conv_id"), {'conv_id': conv_id})
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)



# In a real scenario, this would be loaded securely from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# For this example, we'll mock the LLM client
class MockOpenAI:
    def chat(self, *args, **kwargs):
        user_message = kwargs["messages"][-1]["content"].lower()
        # MODIFIED: Made the trigger condition smarter for better testing
        if any(keyword in user_message for keyword in ["find", "trial", "study"]):
            # Simulate the LLM deciding to call our tool
            # Extracts a query by removing the keywords
            query_text = user_message.replace("find", "").replace("trial", "").replace("study", "").strip()
            tool_call_args = json.dumps({"q": query_text, "limit": 3})
            return {"choices": [{"message": {"tool_calls": [{"function": {"name": "search_clinical_trials", "arguments": tool_call_args}}]}}]}
        else:
            # Simulate a standard conversational response
            return {"choices": [{"message": {"content": "I can help you find information about clinical trials. What are you looking for?"}}]}

llm_client = MockOpenAI() # Replace with openai.OpenAI() in production
llm_client = openai.OpenAI() 


# --- LLM Tool Definition ---
def search_clinical_trials(db, q: Optional[str] = None, status: Optional[str] = None, phase: Optional[str] = None, limit: int = 3):
    """
    This function is a "tool" for the LLM. It searches the database for clinical trials.
    It reuses the logic from the existing get_studies endpoint.
    """
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
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    
    query = text(f"""
        SELECT pd.nct_number, pd.study_title, pd.conditions
        FROM processed_data pd {where_sql}
        ORDER BY pd.last_updated_date DESC NULLS LAST
        LIMIT :limit
    """)
    params['limit'] = limit
    
    results = db.execute(query, params).mappings().all()
    return {"studies": [dict(row) for row in results]}

available_tools = {"search_clinical_trials": search_clinical_trials}




# --- Register all the routers ---
app.include_router(main_router)
app.include_router(webhook_router)
app.include_router(chat_router)

if __name__ == "__main__":
    # if not CLERK_SECRET_KEY:
    #     print("--- WARNING: Clerk environment variables are not set. Auth and webhooks will fail. ---")
    print("--- Starting Augmented Clinical Trials API Server ---")
    print("API documentation will be available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)