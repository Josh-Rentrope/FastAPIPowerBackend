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
    # ... (rest of the filtering logic is unchanged)
    # NOTE: The full implementation of this endpoint is omitted for brevity but should be retained.
    return {"total": 0, "page": page, "limit": limit, "data": []} # Placeholder

@main_router.get("/studies/map-data", response_model=MapResponse)
def get_map_data(db=Depends(get_db), bbox: str = Query(...), zoom: int = Query(...)):
    # This function's implementation remains the same
    # NOTE: The full implementation of this endpoint is omitted for brevity but should be retained.
    return [] # Placeholder

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