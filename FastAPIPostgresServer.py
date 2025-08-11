import datetime
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

import openai

from api.boundaries import router as boundaries_router
from api.countydata import router as county_data_router

# --- Database Configuration ---
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "clinical_trials")

# --- Clerk Configuration ---
CLERK_WEBHOOK_SECRET = os.environ.get("CLERK_WEBHOOK_SECRET")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- FastAPI App Initialization & Lifespan Events ---
app = FastAPI(
    title="Augmented Clinical Trials API",
    description="API for querying clinical trials and supporting real-time chat.",
    version="1.4.0",
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

# --- Insecure dependency for demo purposes ---
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

@main_router.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Augmented Clinical Trials API. Visit /docs for documentation."}

@main_router.get("/studies", response_model=PaginatedStudies)
def get_studies(db=Depends(get_db), q: Optional[str] = None, status: Optional[str] = None, phase: Optional[str] = None, bbox: Optional[str] = None, page: int = 1, limit: int = 10):
    offset = (page - 1) * limit
    where_clauses, params = [], {}
    if q:
        where_clauses.append("(pd.study_title ILIKE :q OR pd.conditions ILIKE :q)")
        params['q'] = f"%{q}%"

    if status:
        frontend_keys = [s.strip() for s in status.split(',')]
        status_map = {
            'RECRUITING': ['RECRUITING'],
            'NOT_YET_RECRUITING': ['NOT_YET_RECRUITING'],
            'COMPLETED': ['COMPLETED']
        }
        db_status_values = [v for key in frontend_keys if key in status_map for v in status_map[key]]
        if db_status_values:
            params['status'] = list(set(db_status_values))
            where_clauses.append("pd.status = ANY(:status)")

    if phase:
        phase_keys = [p.strip().replace('_', '') for p in phase.split(',')]
        if phase_keys:
            phase_or_clauses = [f"pd.phase ILIKE :phase_{i}" for i in range(len(phase_keys))]
            for i, p_key in enumerate(phase_keys):
                params[f"phase_{i}"] = f"%{p_key}%"
            where_clauses.append(f"({' OR '.join(phase_or_clauses)})")

    if bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
            where_clauses.append("EXISTS (SELECT 1 FROM study_locations sl_sub JOIN locations l_sub ON sl_sub.location_id = l_sub.id WHERE sl_sub.study_nct_number = pd.nct_number AND ST_Intersects(l_sub.geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)))")
            params.update({'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat})
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid bbox format. Use 'min_lon,min_lat,max_lon,max_lat'.")

    base_query = "FROM processed_data pd"
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    try:
        count_query = text(f"SELECT COUNT(pd.nct_number) {base_query} {where_sql}")
        total_count = db.execute(count_query, params).scalar_one()

        if total_count == 0:
            return {"total": 0, "page": page, "limit": limit, "data": []}

        ids_query = text(f"SELECT pd.nct_number {base_query} {where_sql} ORDER BY pd.last_updated_date DESC NULLS LAST LIMIT :limit OFFSET :offset")
        params.update({'limit': limit, 'offset': offset})
        paginated_nct_ids = [row[0] for row in db.execute(ids_query, params).fetchall()]

        if not paginated_nct_ids:
            return {"total": total_count, "page": page, "limit": limit, "data": []}

        details_query = text("""
            SELECT pd.nct_number, pd.study_title, pd.study_url, pd.status, pd.conditions, pd.phase, pd.enrollment_count,
                   l.location_text, ST_Y(l.geom) as latitude, ST_X(l.geom) as longitude
            FROM processed_data pd
            LEFT JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
            LEFT JOIN locations l ON sl.location_id = l.id
            WHERE pd.nct_number IN :nct_ids
        """)
        details_result = db.execute(details_query, {'nct_ids': tuple(paginated_nct_ids)}).mappings().all()

        studies_dict = {}
        for row in details_result:
            studies_dict.setdefault(row['nct_number'], {**row, "locations": []})
            if row['location_text']:
                studies_dict[row['nct_number']]['locations'].append({"location_text": row['location_text'], "latitude": row['latitude'], "longitude": row['longitude']})
        
        final_studies = sorted([Study.model_validate(study_data) for study_data in studies_dict.values()], key=lambda s: paginated_nct_ids.index(s.nct_number))

        return {"total": total_count, "page": page, "limit": limit, "data": final_studies}
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
    # Bbox is always required for map data
    where_clauses.append("l.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)")
    params.update({'min_lon': min_lon, 'min_lat': min_lat, 'max_lon': max_lon, 'max_lat': max_lat})

    # Add optional text search filter
    if q:
        where_clauses.append("(pd.study_title ILIKE :q OR pd.conditions ILIKE :q)")
        params['q'] = f"%{q}%"
    
    # --- FIX: Handle status filtering with exact DB values ---
    if status:
        frontend_keys = [s.strip() for s in status.split(',')]
        status_map = {
            'RECRUITING': ['RECRUITING'],
            'NOT_YET_RECRUITING': ['NOT_YET_RECRUITING'],
            'COMPLETED': ['COMPLETED']
        }
        db_status_values = []
        for key in frontend_keys:
            if key in status_map:
                db_status_values.extend(status_map[key])
        if db_status_values:
            params['status'] = list(set(db_status_values))
            where_clauses.append("pd.status = ANY(:status)")

    # --- FIX: Handle phase filtering for single and multi-phase ('|') entries ---
    if phase:
        phase_keys = [p.strip().replace('_', '') for p in phase.split(',')]
        if phase_keys:
            phase_or_clauses = []
            for i, p_key in enumerate(phase_keys):
                param_name = f"phase_{i}"
                phase_or_clauses.append(f"pd.phase ILIKE :{param_name}")
                params[param_name] = f"%{p_key}%"
            if phase_or_clauses:
                where_clauses.append(f"({' OR '.join(phase_or_clauses)})")
    
    # The final WHERE clause to be used in the queries below
    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    # --- 2. Define a zoom threshold to switch between clusters and points ---
    CLUSTER_THRESHOLD = 12

    try:
        if zoom < CLUSTER_THRESHOLD:
            # --- STRATEGY FOR CLUSTERING (ZOOMED OUT) ---
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

            # Structure the data for the response model
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
            
            return [Study.model_validate(study_data) for study_data in studies_dict.values()]
    except ProgrammingError as e:
        print(f"An SQL error occurred in get_map_data: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching map data.")

# --- Webhook Endpoint for Clerk User Sync ---
webhook_router = APIRouter()
@webhook_router.post("/api/webhooks/clerk-events")
async def clerk_webhook(request: Request, db=Depends(get_db)):
    # This endpoint remains the same and is omitted for brevity.
    pass
    
# --- Chat API Endpoints ---
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])

@chat_router.get("/conversations", response_model=List[Conversation])
def get_user_conversations(db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email
    query = text("SELECT c.id, c.created_at, c.updated_at FROM conversations c JOIN participants p ON c.id = p.conversation_id WHERE p.user_id = :user_id ORDER BY c.updated_at DESC;")
    results = db.execute(query, {'user_id': user_id}).mappings().all()
    return [{k: str(v) for k, v in result.items()} for result in results]

@chat_router.post("/conversations/create", response_model=Conversation, status_code=201)
def create_conversation(req_body: CreateConversationRequest, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    creator_id = email
    participant_ids = list(set(req_body.participant_ids + [creator_id]))
  
    with db.begin():
        upsert_user_query = text("INSERT INTO users (clerk_user_id) VALUES (:user_id) ON CONFLICT (clerk_user_id) DO NOTHING;")
        for user_id in participant_ids:
            db.execute(upsert_user_query, {"user_id": user_id})

        conv_query = text("INSERT INTO conversations DEFAULT VALUES RETURNING id, created_at, updated_at;")
        new_conv = db.execute(conv_query).mappings().one()
        
        part_data = [{"conv_id": new_conv['id'], "user_id": pid} for pid in participant_ids]
        part_query = text("INSERT INTO participants (conversation_id, user_id) VALUES (:conv_id, :user_id);")
        db.execute(part_query, part_data)
        
    return {k: str(v) for k, v in new_conv.items()}

@chat_router.get("/conversations/{conv_id}/messages", response_model=List[Message])
def get_conversation_messages(conv_id: str, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email
    check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
    if not db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a participant of this conversation")
    
    msg_query = text("SELECT id, conversation_id, sender_id, content, sent_at FROM messages WHERE conversation_id = :conv_id ORDER BY sent_at ASC, id ASC;")
    results = db.execute(msg_query, {'conv_id': conv_id}).mappings().all()
    return [{k: str(v) for k, v in result.items()} for result in results]

@chat_router.post("/conversations/{conv_id}/messages", response_model=Message, status_code=201)
def post_message(conv_id: str, request_body: dict, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email
    content = request_body.get('content')
    if not content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    with db.begin():
        # 1. Save the user's message
        user_msg_query = text("INSERT INTO messages (conversation_id, sender_id, content) VALUES (:conv_id, :sender_id, :content);")
        db.execute(user_msg_query, {'conv_id': conv_id, 'sender_id': user_id, 'content': content})

        # 2. Fetch recent conversation history for LLM context
        history_query = text("""
            WITH recent_messages AS (
                SELECT id, sender_id, content, sent_at FROM messages
                WHERE conversation_id = :conv_id ORDER BY sent_at DESC, id DESC LIMIT 10
            )
            SELECT sender_id, content FROM recent_messages ORDER BY sent_at ASC, id ASC;
        """)
        history_results = db.execute(history_query, {'conv_id': conv_id}).mappings().all()
        
        messages_for_llm = [
            {"role": "system", "content": "You are a clinical trial discovery expert. When a user asks you to find trials, use the search_clinical_trials tool. Summarize the results for the user and include the NCT numbers."}
        ]
        for msg in history_results:
            messages_for_llm.append({"role": "user" if msg['sender_id'] == user_id else "assistant", "content": msg['content']})

        # 3. First API call to get model's response
        try:
            first_response = llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages_for_llm,
                tools=tools_schema,
                tool_choice="auto",
            )
            response_message = first_response.choices[0].message
            tool_calls = response_message.tool_calls
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

        # 4. Check if the model decided to call a tool
        if tool_calls:
            messages_for_llm.append(response_message)  # Add assistant's turn to history

            # 5. Execute the function calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = function_dispatcher.get(function_name)
                function_args = json.loads(tool_call.function.arguments)
                
                if function_to_call:
                    tool_output = function_to_call(db=db, **function_args)
                    messages_for_llm.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_output),
                    })
            
            # 6. Second API call to get the final natural language response
            try:
                second_response = llm_client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages_for_llm,
                )
                final_content = second_response.choices[0].message.content
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error in second LLM call: {e}")
        else:
            final_content = response_message.content

        # 7. Save the AI's final response to the database
        upsert_ai_query = text("INSERT INTO users (clerk_user_id) VALUES (:user_id) ON CONFLICT (clerk_user_id) DO NOTHING;")
        db.execute(upsert_ai_query, {"user_id": "00000000-0000-0000-0000-000000000000"})

        ai_msg_query = text("INSERT INTO messages (conversation_id, sender_id, content) VALUES (:conv_id, '00000000-0000-0000-0000-000000000000', :content) RETURNING id, conversation_id, sender_id, content, sent_at;")
        new_ai_msg = db.execute(ai_msg_query, {'conv_id': conv_id, 'content': final_content}).mappings().one()

        # Update conversation timestamp
        db.execute(text("UPDATE conversations SET updated_at = now() WHERE id = :conv_id;"), {'conv_id': conv_id})

    return {k: str(v) for k, v in new_ai_msg.items()}


@chat_router.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, db=Depends(get_db), email: str = Depends(get_user_email_from_header)):
    user_id = email
    with db.begin():
        check_query = text("SELECT 1 FROM participants WHERE conversation_id = :conv_id AND user_id = :user_id")
        if not db.execute(check_query, {'conv_id': conv_id, 'user_id': user_id}).scalar_one_or_none():
            raise HTTPException(status_code=403, detail="You do not have permission to delete this conversation.")
        
        db.execute(text("DELETE FROM messages WHERE conversation_id = :conv_id"), {'conv_id': conv_id})
        db.execute(text("DELETE FROM participants WHERE conversation_id = :conv_id"), {'conv_id': conv_id})
        result = db.execute(text("DELETE FROM conversations WHERE id = :conv_id"), {'conv_id': conv_id})

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found.")
            
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# --- LLM and Tool Configuration ---
llm_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def search_clinical_trials(db, q: Optional[str] = None, status: Optional[str] = None, phase: Optional[str] = None, limit: int = 3):
    """Searches the database for clinical trials and returns a list of matching trials."""
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
    
    query = text(f"SELECT pd.nct_number, pd.study_title, pd.conditions FROM processed_data pd {where_sql} ORDER BY pd.last_updated_date DESC NULLS LAST LIMIT :limit")
    params['limit'] = limit
    
    results = db.execute(query, params).mappings().all()
    return {"studies": [dict(row) for row in results]}

# --- Maps the function name (string) to the actual Python function object ---
function_dispatcher = {
    "search_clinical_trials": search_clinical_trials,
}

# --- Defines the schema for the tools that the LLM can use ---
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_clinical_trials",
            "description": "Searches the database for clinical trials based on various criteria like a query string, status, or phase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "A search term to query against trial titles and conditions, e.g., 'cancer' or 'diabetes'."
                    },
                    "status": {
                        "type": "string",
                        "description": "The recruitment status of the trial, e.g., 'RECRUITING'."
                    },
                    "phase": {
                        "type": "string",
                        "description": "The clinical phase of the trial, e.g., 'PHASE1', 'PHASE2'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The maximum number of trials to return. Defaults to 3."
                    }
                },
                "required": [],
            }
        }
    }
]

# --- Register all the routers ---
app.include_router(main_router)
app.include_router(webhook_router)
app.include_router(chat_router)
app.include_router(boundaries_router)
app.include_router(county_data_router)

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("--- WARNING: OPENAI_API_KEY environment variable is not set. Chat will fail. ---")
    print("--- Starting Augmented Clinical Trials API Server ---")
    print("API documentation will be available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)