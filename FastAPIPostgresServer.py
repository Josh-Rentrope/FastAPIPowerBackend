import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

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
    title="Augmented Clinical Trials API",
    description="API for querying clinical trials with text, status, and geospatial filters.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1","http://127.0.0.1:3000", "http://localhost:3000", "https://yourfrontend.com"],  # Specify allowed origins
    allow_credentials=True,  # Allow cookies and authorization headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
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
    """Represents a single geographic location for a study."""
    location_text: str
    latitude: float
    longitude: float

class Study(BaseModel):
    """Represents a single clinical trial with its locations."""
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

class PaginatedStudies(BaseModel):
    """Container for a paginated list of studies."""
    total: int
    page: int
    limit: int
    data: List[Study]

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
    return {"message": "Welcome to the Augmented Clinical Trials API. Visit /docs for documentation."}

@app.get("/studies", response_model=PaginatedStudies)
def get_studies(
    q: Optional[str] = Query(None, description="Search term for study title or conditions."),
    status: Optional[str] = Query(None, description="Filter by study status (e.g., 'RECRUITING')."),
    phase: Optional[str] = Query(None, description="Filter by study phase (e.g., 'PHASE1')."),
    bbox: Optional[str] = Query(None, description="Bounding box filter: min_lon,min_lat,max_lon,max_lat"),
    page: int = Query(1, ge=1, description="Page number for pagination."),
    limit: int = Query(10, ge=1, le=500, description="Number of results per page."),
    db=Depends(get_db)
):
    """
    Get a paginated list of clinical trials with text, status, and geospatial filters.
    """
    offset = (page - 1) * limit
    
    # --- Build the WHERE clause and parameters dynamically ---
    where_clauses = []
    params = {}

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

if __name__ == "__main__":
    print("--- Starting Augmented Clinical Trials API Server ---")
    print("API documentation will be available at http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
