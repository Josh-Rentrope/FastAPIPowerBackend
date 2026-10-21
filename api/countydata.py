"""County-level correlation between trial density and community health measures.

Rewritten from the original api/countydata.py. Three things changed:

* **The demographic variable is now an allowlist, not a character test.** The
  original interpolated the caller's `demographic_variable` straight into SQL
  behind this guard:

      if not demographic_variable.isalnum() and '_' not in demographic_variable:
          raise HTTPException(400, ...)

  which is inverted in effect -- the intended shape `v001_raw_value` contains an
  underscore, so the first clause is False and the whole condition is False, so
  nothing is ever rejected. A value like `x_1) OR 1=1--` passed cleanly. The
  values are a fixed, known set, so they are checked against that set.

* **A trailing-slash redirect.** The router was declared with
  `prefix="/api/correlation"` and the route as `"/"`, making the real path
  `/api/correlation/`. A client asking for `/api/correlation` got a 307, which a
  browser reports as a CORS failure rather than as a redirect. The route is now
  registered at both spellings by the caller.

* **Coarse points are excluded from the spatial join.** A location geocoded only
  to a country centroid lands in whichever county happens to contain the middle
  of the country -- usually none, sometimes the wrong one. Anything below `city`
  precision is dropped, which is what `geocode_precision` exists for.

The join itself is `ST_Contains` against the GiST index on
`county_boundaries.geom`, driven from the (much smaller) set of well-placed
points rather than from the counties.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

# The measures the schema actually carries, and the only values accepted. These
# names come from the front end's dropdown, which asks for the older CHR codes;
# etl/05_load_counties.py maps them to the current release's columns. Keeping the
# allowlist here rather than deriving it from information_schema means a new
# column is an explicit decision rather than something a caller can discover by
# probing.
ALLOWED_VARIABLES = {
    "v001_raw_value",  # Premature Death Rate
    "v009_raw_value",  # Adult Smoking Rate
    "v011_raw_value",  # Adult Obesity Rate
    "v021_raw_value",  # % 65 and over
    "v043_raw_value",  # % Uninsured
    "v057_raw_value",  # Children in Poverty Rate
}


class CompletedTrial(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: str = Field(..., alias="nct_number")
    title: str = Field(..., alias="study_title")
    status: str
    sponsor: Optional[str] = None


class CountyCorrelation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fips_code: str
    county_name: str
    demographic_value: Optional[float] = None
    completed_trials: List[CompletedTrial] = []


router = APIRouter(prefix="/api/correlation", tags=["Correlation"])


def get_db(request: Request):
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Registered at both "" and "/" so neither spelling redirects.
@router.get("", response_model=List[CountyCorrelation])
@router.get("/", response_model=List[CountyCorrelation], include_in_schema=False)
def get_county_correlation(
    condition: str,
    demographic_variable: str,
    db: Session = Depends(get_db),
):
    """County health measures alongside the completed trials in each county.

    Note the shape of the query: it fetches every county and then attaches trials
    that fall inside it, so the response is a full choropleth by design. A caller
    wanting a narrow answer should filter on `condition`, which is applied in SQL.
    """
    if not condition or not condition.strip():
        return []
    if demographic_variable not in ALLOWED_VARIABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown demographic variable. Allowed: {sorted(ALLOWED_VARIABLES)}",
        )

    # The column name is interpolated, but only after the allowlist check above,
    # which is the entire reason the check exists.
    demographics = db.execute(text(f"""
        SELECT fipscode, county_name, {demographic_variable} AS demographic_value
        FROM county_health_rankings
    """)).mappings().all()

    counties: dict[str, dict] = {}
    for row in demographics:
        counties[row["fipscode"]] = {
            "fips_code": row["fipscode"],
            "county_name": row["county_name"],
            "demographic_value": row["demographic_value"],
            "completed_trials": [],
        }
    if not counties:
        return []

    # `IN ('postal','city')` is the guard described in the module docstring: a
    # country centroid is not a location and must not be counted as one.
    trials = db.execute(text("""
        SELECT DISTINCT
               pd.nct_number, pd.study_title, pd.status, pd.sponsor, cb.fipscode
        FROM processed_data pd
        JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
        JOIN locations l        ON sl.location_id = l.id
        JOIN county_boundaries cb ON ST_Contains(cb.geom, l.geom)
        WHERE pd.status = 'COMPLETED'
          AND pd.conditions ILIKE :condition
          AND l.geom IS NOT NULL
          AND l.geocode_precision IN ('postal', 'city')
    """), {"condition": f"%{condition}%"}).mappings().all()

    for t in trials:
        bucket = counties.get(t["fipscode"])
        if bucket is not None:
            bucket["completed_trials"].append(CompletedTrial.model_validate(t))

    return [CountyCorrelation.model_validate(c) for c in counties.values()]
