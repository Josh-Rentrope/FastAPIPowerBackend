# api/boundaries.py
from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi.responses import JSONResponse

# --- API Router ---
router = APIRouter(
    prefix="/api/boundaries",
    tags=["Boundaries"],
)

def get_db(request: Request):
    """Dependency to get a DB session."""
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/counties", response_class=JSONResponse)
def get_county_boundaries(db: Session = Depends(get_db)):
    """
    This endpoint queries the database for county boundaries and returns them
    as a GeoJSON FeatureCollection, which is ideal for map rendering.
    """
    try:
        # This SQL query uses PostGIS and JSON functions to build a valid
        # GeoJSON FeatureCollection directly in the database.
        query = text("""
            SELECT json_build_object(
                'type', 'FeatureCollection',
                'features', json_agg(
                    json_build_object(
                        'type', 'Feature',
                        'geometry', ST_AsGeoJSON(geom)::json,
                        'properties', json_build_object(
                            'GEOID', fipscode,
                            'NAME', county_name
                        )
                    )
                )
            ) AS geojson
            FROM county_boundaries;
        """)
        
        result = db.execute(query).scalar_one_or_none()

        if result is None:
            # Return an empty FeatureCollection if no data is found
            return {"type": "FeatureCollection", "features": []}
            
        return result

    except Exception as e:
        print(f"An error occurred in get_county_boundaries: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

