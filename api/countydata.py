# api/county_data.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional

from pydantic import BaseModel, Field

# --- Pydantic Schemas (Updated) ---
class CompletedTrial(BaseModel):
    id: str = Field(..., alias='nct_number')
    title: str = Field(..., alias='study_title')
    status: str
    sponsor: Optional[str] = None

    class Config:
        from_attributes = True
        populate_by_name = True


class CountyCorrelation(BaseModel):
    fips_code: str
    county_name: str
    demographic_value: Optional[float] = None
    completed_trials: List[CompletedTrial] = []

    class Config:
        from_attributes = True

# --- API Router ---
router = APIRouter(prefix="/api/correlation", tags=["Correlation"])

def get_db(request: Request):
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/", response_model=List[CountyCorrelation])
def get_county_correlation(
    condition: str,
    demographic_variable: str,
    db: Session = Depends(get_db)
):
    """
    Provides data for the correlation map by fetching county-level demographic
    data and spatially joining it with completed clinical trials.
    """
    if not condition or not demographic_variable:
        return []
        
    try:
        if not demographic_variable.isalnum() and '_' not in demographic_variable:
             raise HTTPException(status_code=400, detail="Invalid demographic variable.")

        demographics_query = text(f"""
            SELECT fipscode, county_name, {demographic_variable} as demographic_value
            FROM county_health_rankings;
        """)
        demographics_result = db.execute(demographics_query).mappings().all()

        # --- FIX: Explicitly construct the dictionary to ensure all keys are present ---
        county_data_map = {}
        for row in demographics_result:
            county_data_map[row['fipscode']] = {
                'fips_code': row['fipscode'],
                'county_name': row['county_name'],
                'demographic_value': row['demographic_value'],
                'completed_trials': []  # Initialize with an empty list
            }

        trials_query = text("""
            SELECT 
                pd.nct_number, 
                pd.study_title, 
                pd.status, 
                pd.sponsor, 
                cb.fipscode
            FROM processed_data pd
            JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
            JOIN locations l ON sl.location_id = l.id
            JOIN county_boundaries cb ON ST_Contains(cb.geom, l.geom)
            WHERE pd.status = 'COMPLETED'
            AND pd.conditions ILIKE :condition;
        """)
        trials_result = db.execute(trials_query, {'condition': f"%{condition}%"}).mappings().all()

        for trial in trials_result:
            fips = trial['fipscode']
            if fips in county_data_map:
                # The 'completed_trials' key is guaranteed to exist now
                county_data_map[fips]['completed_trials'].append(CompletedTrial.model_validate(trial))

        response_data = [CountyCorrelation.model_validate(data) for data in county_data_map.values()]
        return response_data

    except Exception as e:
        print(f"An error occurred in get_county_correlation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

