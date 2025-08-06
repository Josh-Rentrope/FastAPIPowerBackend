import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, ProgrammingError

# --- Database Configuration ---
# Load credentials from environment variables
DB_USER = os.environ.get("DB_USER", "user")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "password")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "clinical_trials")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- FastAPI App Initialization & Lifespan Events ---
app = FastAPI(
    title="Simple Clinical Trials API",
    description="A simplified API to query the 'processed_data' table directly.",
    version="1.0.0",
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
class SimpleStudy(BaseModel):
    """Represents a single clinical trial from the processed_data table."""
    nct_number: str
    study_title: str
    study_url: str
    status: Optional[str] = None
    conditions: Optional[str] = None
    phase: Optional[str] = None
    enrollment_count: Optional[int] = None
    
    class Config:
        from_attributes = True

class PaginatedSimpleStudies(BaseModel):
    """Container for a paginated list of simple studies."""
    total: int
    page: int
    limit: int
    data: List[SimpleStudy]

# --- Dependency for Database Session ---
def get_db(request: Request):
    """FastAPI dependency to provide a database session per request."""
    if not request.app.state.SessionLocal:
        raise HTTPException(status_code=503, detail="Database connection is not configured or has failed.")
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Simple Clinical Trials API. Visit /docs for documentation."}

@app.get("/studies", response_model=PaginatedSimpleStudies)
def get_simple_studies(
    q: Optional[str] = Query(None, description="Search term for study title or conditions."),
    status: Optional[str] = Query(None, description="Filter by study status (e.g., 'RECRUITING')."),
    phase: Optional[str] = Query(None, description="Filter by study phase (e.g., 'PHASE1')."),
    page: int = Query(1, ge=1, description="Page number for pagination."),
    limit: int = Query(100, ge=1, le=500, description="Number of results per page."),
    db=Depends(get_db)
):
    """
    Get a paginated list of clinical trials directly from the processed_data table.
    """
    offset = (page - 1) * limit
    
    # --- Build the WHERE clause and parameters dynamically ---
    where_clauses = []
    params = {}

    if q:
        where_clauses.append("(study_title ILIKE :q OR conditions ILIKE :q)")
        params['q'] = f"%{q}%"
    if status:
        where_clauses.append("status ILIKE :status")
        params['status'] = f"%{status}%"
    if phase:
        where_clauses.append("phase ILIKE :phase")
        params['phase'] = f"%{phase}%"

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    try:
        # --- Get the total count of matching studies ---
        count_query = text(f"SELECT COUNT(*) FROM processed_data {where_sql}")
        total_count = db.execute(count_query, params).scalar_one()

        if total_count == 0:
            return {"total": 0, "page": page, "limit": limit, "data": []}

        # --- Get the paginated list of studies ---
        data_query = text(f"""
            SELECT *
            FROM processed_data
            {where_sql}
            ORDER BY last_updated_date DESC NULLS LAST
            LIMIT :limit OFFSET :offset
        """)
        params.update({'limit': limit, 'offset': offset})
        
        results = db.execute(data_query, params).mappings().all()
        
        # Pydantic V1 compatible validation
        final_studies = [SimpleStudy.parse_obj(row) for row in results]

        return {
            "total": total_count,
            "page": page,
            "limit": limit,
            "data": final_studies
        }
    except ProgrammingError as e:
        print(f"An SQL error occurred: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while querying the database.")

if __name__ == "__main__":
    print("--- Starting Simple Clinical Trials API Server ---")
    print("API documentation will be available at http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
