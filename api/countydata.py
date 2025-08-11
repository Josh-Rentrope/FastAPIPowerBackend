# api/county_data.py
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional

from pydantic import BaseModel

# --- Pydantic Schemas ---

class CompletedTrial(BaseModel):
    """A simplified model representing a completed clinical trial."""
    id: str
    title: str
    status: str
    sponsor: Optional[str] = None

    class Config:
        from_attributes = True


class CountyCorrelation(BaseModel):
    """Represents a single county's data for the correlation map."""
    fips_code: str
    county_name: str
    demographic_value: Optional[float] = None
    completed_trials: List[CompletedTrial] = []

    class Config:
        from_attributes = True

# --- API Router ---

router = APIRouter(
    prefix="/api/correlation",
    tags=["Correlation"],
)

def get_db(request: Request):
    return request.app.state.SessionLocal()


@router.get("/", response_model=List[CountyCorrelation])
def get_county_correlation(
    condition: str,
    demographic_variable: str, # e.g., v021_raw_value
    db: Session = Depends(get_db)
):
    """
    This endpoint provides data for the correlation map.
    It fetches county-level demographic data and correlates it with
    completed clinical trials for a specific condition.
    """
    try:
        # Validate that the demographic_variable is a safe column name
        # In a production app, you'd have a strict allow-list of columns.
        if not demographic_variable.isalnum() and '_' not in demographic_variable:
             raise HTTPException(status_code=400, detail="Invalid demographic variable.")

        # 1. Fetch demographic data for all counties
        # The 'text()' construct helps prevent SQL injection with dynamic column names.
        demographics_query = text(f"""
            SELECT fipscode, county_name, {demographic_variable} as demographic_value
            FROM county_health_rankings
            WHERE state_name = 'North Carolina';
        """)
        demographics_result = db.execute(demographics_query).mappings().all()

        county_data_map = {row['fipscode']: dict(row) for row in demographics_result}

        # 2. Fetch completed trials for the given condition
        trials_query = text("""
            SELECT pd.nct_number, pd.study_title, pd.status, pd.sponsor, l.fipscode
            FROM processed_data pd
            JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
            JOIN locations l ON sl.location_id = l.id
            WHERE pd.status = 'COMPLETED'
            AND pd.conditions ILIKE :condition
            AND l.fipscode IS NOT NULL;
        """)
        trials_result = db.execute(trials_query, {'condition': f"%{condition}%"}).mappings().all()

        # 3. Aggregate trials by county
        for trial in trials_result:
            fips = trial['fipscode']
            if fips in county_data_map:
                if 'completed_trials' not in county_data_map[fips]:
                    county_data_map[fips]['completed_trials'] = []
                county_data_map[fips]['completed_trials'].append(CompletedTrial(
                    id=trial['nct_number'],
                    title=trial['study_title'],
                    status=trial['status'],
                    sponsor=trial['sponsor']
                ))

        # 4. Format the final response
        response_data = [CountyCorrelation(**data) for data in county_data_map.values()]
        
        return response_data

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

