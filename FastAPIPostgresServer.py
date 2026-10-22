#!/usr/bin/env python3
"""Augmented Clinical Trials API.

Rewritten from the original FastAPIPostgresServer.py. The endpoint surface the
front end already depends on is preserved exactly; what changed is listed below,
and every item on it was a live defect rather than a preference.

## Bugs fixed

* **The map clustering was silently wrong.** The cluster query called
  `ST_SnapToGrid(geom, :min_lon, :min_lat, dx, dy)` -- five arguments. In PostGIS
  the five-argument form is `(geom, size_x, size_y, size_z, size_m)`, *not* an
  origin form, so `min_lon`/`min_lat` became the x/y **cell sizes** (hundreds of
  degrees) and the degree deltas landed in z/m. Every point in the viewport
  collapsed into a single cluster. It looked plausible only because one large
  cluster is not obviously wrong on a map.

* **The bbox filter was a correlated subquery per row.** `EXISTS (...)` ran the
  join for all 607k studies. It is now a semi-join driven by the GiST index on
  `locations.geom`, and uses the `&&` operator (index-only) rather than
  `ST_Intersects`.

* **A location with no geometry crashed the response.** `latitude` and
  `longitude` are non-nullable in the model, but `geom` is NULL for ungeocoded
  rows, so a Pydantic 500 was one ungeocoded location away. Now filtered.

* **`phase ILIKE '%PHASE1%'` also matched `EARLY_PHASE1`** -- a wrong-answer bug,
  not a performance one. Phases are now compared as token membership.

* **The status filter only knew 3 of the 13 real statuses**, so filtering by
  TERMINATED or SUSPENDED silently returned nothing. It now passes through any
  status the export contains.

* **`/api/correlation` redirected.** The router prefix plus a "/" route made the
  real path `/api/correlation/`, and a request without the trailing slash got a
  307 -- which a browser reports as a CORS failure rather than as a redirect.

* **`sponsor` and `acronym` were never returned**, which is why the front end
  hardcodes the string "Sponsor from DB". They are columns in `processed_data`.

* CORS origins were hardcoded to localhost plus a `https://yourfrontend.com`
  placeholder. They come from the environment now.

## Security

Chat identity is a server-issued session cookie, not the `X-User-Email` header
the original trusted. See sessions.py. Conversation ownership is enforced on every
read and delete, which is what makes chat history a real boundary rather than a
comment.

Configuration (all environment variables):

    DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD DB_SSLMODE
    CORS_ORIGINS              comma-separated, exact origins, no wildcards
    CT_SESSION_COOKIE_SECURE  "0" only for plain-http local development
    OPENAI_API_KEY OPENAI_BASE_URL CHAT_MODEL   (any OpenAI-compatible endpoint)
    ENABLE_CHAT               "1" to mount the chat router
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Union

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

import qualification
import sessions as sessions_mod
from api.boundaries import router as boundaries_router
from api.qualification import router as qualification_router

# The correlation router declares prefix "/api/correlation" with a "/" route, so
# only "/api/correlation/" is a real path. It is mounted at both spellings below.
from api.countydata import router as county_data_router

# ── configuration ───────────────────────────────────────────────────────────

DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "clinical_trials")
DB_SSLMODE = os.environ.get("DB_SSLMODE", "require")

CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",") if o.strip()]

# Any OpenAI-compatible endpoint, so this can point at OpenAI, DeepSeek, or a
# local Ollama rather than requiring an OpenAI key.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
ENABLE_CHAT = os.environ.get("ENABLE_CHAT", "1") == "1"

# ── map clustering ──────────────────────────────────────────────────────────
#
# Cells across the viewport at zoom 3, doubling per level above that, capped so a
# dense viewport cannot return tens of thousands of markers. Above
# CLUSTER_ZOOM_THRESHOLD the map switches to individual points and clusters on
# the client instead.
#
# These are environment variables rather than constants because "how much detail"
# is a taste question that should be tunable without a code change.
# Cells across the viewport AT ZOOM 3, doubling per level above that. Measured
# against the whole-US viewport (49,285 locations):
#
#     base   clusters at zoom 3   largest cluster
#        4                20           11,865
#        8                62            4,734     <- default
#       16               206            3,367
#       32               627            1,814
#       64             1,687            1,506
#
# Lower base = fewer, larger clusters when zoomed out. At 64 the national view
# was returning ~1,700 markers, which reads as noise rather than density.
CLUSTER_GRID_BASE = int(os.environ.get("CLUSTER_GRID_BASE", "8"))
CLUSTER_GRID_MAX = int(os.environ.get("CLUSTER_GRID_MAX", "1024"))
CLUSTER_THRESHOLD = int(os.environ.get("CLUSTER_ZOOM_THRESHOLD", "12"))

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

app = FastAPI(
    title="Augmented Clinical Trials API",
    description="Query clinical trials and supporting county demographics.",
    version="2.0.0",
)

# `allow_credentials=True` and a wildcard origin are contradictory and browsers
# reject the combination outright, so a named list is required, not preferred.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class EnsureCorsOnError(BaseHTTPMiddleware):
    """Put CORS headers on unhandled 500s.

    Starlette installs its `ServerErrorMiddleware` outside every user middleware,
    so an exception that escapes a route produces a response with no
    `Access-Control-Allow-Origin` at all -- and a browser can only report that as
    "blocked by CORS policy". The real fault is a server error, and this makes the
    response say so, with the headers the browser needs in order to show it.

    Added AFTER CORSMiddleware, because `add_middleware` puts each new layer on
    the outside; this one has to be outside CORS to catch what CORS cannot.
    """

    def __init__(self, app, origins: list[str]) -> None:
        super().__init__(app)
        self.origins = set(origins)

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:  # noqa: BLE001
            origin = request.headers.get("origin")
            headers = {}
            if origin and origin in self.origins:
                headers["Access-Control-Allow-Origin"] = origin
                headers["Access-Control-Allow-Credentials"] = "true"
                headers["Vary"] = "Origin"
            # Logged, because the point of this middleware is that the failure is
            # otherwise invisible from the client side.
            print(f"UNHANDLED {request.method} {request.url.path}: "
                  f"{type(exc).__name__}: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error",
                         "error": type(exc).__name__},
                headers=headers,
            )


app.add_middleware(EnsureCorsOnError, origins=CORS_ORIGINS)


@app.on_event("startup")
async def startup_db_client():
    print(f"connecting to {DB_HOST}:{DB_PORT}/{DB_NAME} (sslmode={DB_SSLMODE})")
    try:
        app.state.engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,       # survives the pool holding a dead socket
            pool_recycle=1800,        # ...and managed failovers, which drop
            pool_size=5,              #    connections well inside an hour
            max_overflow=5,
            poolclass=QueuePool,
            connect_args={"sslmode": DB_SSLMODE, "connect_timeout": 10},
        )
        app.state.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=app.state.engine)
        with app.state.engine.connect() as c:
            c.execute(text("SELECT 1"))
        print("database connection ok")
        print(f"cors origins: {CORS_ORIGINS}")
        print(f"chat: {'enabled' if ENABLE_CHAT and LLM_API_KEY else 'disabled'} "
              f"(model={CHAT_MODEL}, base_url={LLM_BASE_URL or 'default'})")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: could not connect to the database: {exc}")
        app.state.engine = None
        app.state.SessionLocal = None


@app.on_event("shutdown")
async def shutdown_db_client():
    if getattr(app.state, "engine", None):
        app.state.engine.dispose()


def get_db(request: Request):
    if not request.app.state.SessionLocal:
        raise HTTPException(503, "database connection is not configured or failed")
    db = request.app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_conn(request: Request):
    """Raw DBAPI connection, for the session lookup that needs one."""
    if not request.app.state.engine:
        raise HTTPException(503, "database connection is not configured or failed")
    return request.app.state.engine.raw_connection()


# ── models ──────────────────────────────────────────────────────────────────


class StudyLocation(BaseModel):
    location_text: str
    city: Optional[str] = None
    country_code: Optional[str] = None
    latitude: float
    longitude: float
    # Which rung of the geocode ladder produced this point. The UI can dim or
    # exclude low-confidence pins; "country" means the point is the middle of a
    # country, not a location.
    precision: Optional[str] = None


class Study(BaseModel):
    nct_number: str
    study_title: str
    study_url: Optional[str] = None
    acronym: Optional[str] = None
    status: Optional[str] = None
    conditions: Optional[str] = None
    phase: Optional[str] = None
    sponsor: Optional[str] = None
    enrollment_count: Optional[int] = None
    # Structured eligibility, straight from the registry's own module. These are
    # the highest-signal fields a patient scans ("18+ · All sexes") and they cost
    # one LEFT JOIN. Absent for studies with no eligibility record.
    min_age_years: Optional[int] = None
    max_age_years: Optional[int] = None
    sex: Optional[str] = None
    healthy_volunteers: Optional[bool] = None
    locations: List[StudyLocation] = []

    class Config:
        from_attributes = True


class Cluster(BaseModel):
    is_cluster: bool = True
    point_count: int
    latitude: float
    longitude: float


class Site(BaseModel):
    """One geocoded location hosting several studies.

    Locations dedupe at city level, so a single point can represent a large
    research hub -- Houston alone carries over 15,000 studies. Returning it as one
    arbitrary Study told the user a city held one trial. This carries the count
    and a short, most-recent-first list, which is what a popup can actually show.
    """
    is_site: bool = True
    point_count: int
    latitude: float
    longitude: float
    trials: List[Study] = []


MapResponse = List[Union[Cluster, Site, Study]]


class PaginatedStudies(BaseModel):
    total: int
    page: int
    limit: int
    data: List[Study]


class SessionInfo(BaseModel):
    display_name: Optional[str] = None


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


# ── shared query fragments ──────────────────────────────────────────────────

# Every status the export actually contains. The original mapped only three,
# which made the other ten silently unqueryable.
KNOWN_STATUSES = {
    "COMPLETED", "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION", "TERMINATED", "WITHDRAWN", "SUSPENDED",
    "UNKNOWN", "WITHHELD", "NO_LONGER_AVAILABLE", "AVAILABLE",
    "APPROVED_FOR_MARKETING",
}


def build_filters(q, status, phase):
    """Shared WHERE fragments for /studies and /studies/map-data."""
    clauses: list[str] = []
    params: dict = {}

    if q:
        # Trigram-indexed on both columns; see db/00_setup_database.py.
        clauses.append("(pd.study_title ILIKE :q OR pd.conditions ILIKE :q)")
        params["q"] = f"%{q}%"

    if status:
        wanted = [s.strip().upper().replace(" ", "_") for s in status.split(",") if s.strip()]
        wanted = [s for s in wanted if s in KNOWN_STATUSES]
        if wanted:
            clauses.append("pd.status = ANY(:status)")
            params["status"] = wanted

    if phase:
        # Token membership, not a LIKE. `phase ILIKE '%PHASE1%'` also matches
        # EARLY_PHASE1, which is a different phase and a wrong answer.
        wanted = [p.strip().upper().replace(" ", "").replace("_", "")
                  for p in phase.split(",") if p.strip()]
        if wanted:
            clauses.append("""
                EXISTS (
                    SELECT 1 FROM unnest(string_to_array(pd.phase, '|')) AS ph
                    WHERE upper(replace(ph, ' ', '')) = ANY(:phases)
                )
            """)
            params["phases"] = wanted

    return clauses, params


def parse_bbox(bbox: str | None) -> Optional[tuple[float, float, float, float]]:
    if not bbox:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    except ValueError:
        raise HTTPException(400, "Invalid bbox. Use 'min_lon,min_lat,max_lon,max_lat'.")
    if min_lon > max_lon or min_lat > max_lat:
        raise HTTPException(400, "bbox min values must be less than max values.")
    return min_lon, min_lat, max_lon, max_lat


# ── core routes ─────────────────────────────────────────────────────────────

main_router = APIRouter()


@main_router.get("/", include_in_schema=False)
def root():
    return {"message": "Augmented Clinical Trials API. See /docs."}


@main_router.get("/health")
def health(request: Request):
    """App Platform polls this.

    Deliberately a bare connectivity check and nothing more: the platform calls
    it every 30s with a 10s timeout, so putting a real query here would turn a
    slow database into a container the platform replaces -- a performance
    problem escalated into an outage.
    """
    try:
        if request.app.state.engine:
            with request.app.state.engine.connect() as c:
                c.execute(text("SELECT 1"))
        else:
            raise HTTPException(503, "database not configured")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@main_router.get("/stats")
def stats(db=Depends(get_db)):
    """Cheap counts, so a load can be confirmed without guessing."""
    row = db.execute(text("""
        SELECT (SELECT count(*) FROM processed_data)                  AS studies,
               (SELECT count(*) FROM locations)                       AS locations,
               (SELECT count(*) FROM locations WHERE geom IS NOT NULL) AS placed,
               (SELECT count(*) FROM study_locations)                 AS study_locations,
               (SELECT count(*) FROM locations
                 WHERE geocode_precision IN ('postal','city'))        AS precise
    """)).mappings().one()
    return dict(row)


@main_router.get("/studies", response_model=PaginatedStudies)
def get_studies(db=Depends(get_db), q: Optional[str] = None,
                status: Optional[str] = None, phase: Optional[str] = None,
                bbox: Optional[str] = None, page: int = 1, limit: int = 10):
    limit = max(1, min(limit, 200))
    page = max(1, page)
    offset = (page - 1) * limit

    clauses, params = build_filters(q, status, phase)
    env = parse_bbox(bbox)
    if env:
        # A semi-join driven by the GiST index, rather than the correlated EXISTS
        # that ran the whole join once per study row.
        clauses.append("""
            pd.nct_number IN (
                SELECT sl.study_nct_number
                FROM locations l
                JOIN study_locations sl ON sl.location_id = l.id
                WHERE l.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
            )
        """)
        params.update(dict(zip(("min_lon", "min_lat", "max_lon", "max_lat"), env)))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    total = db.execute(text(f"SELECT count(*) FROM processed_data pd {where}"), params).scalar_one()
    if total == 0:
        return {"total": 0, "page": page, "limit": limit, "data": []}

    ids = [r[0] for r in db.execute(text(f"""
        SELECT pd.nct_number FROM processed_data pd {where}
        ORDER BY pd.last_updated_date DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """), {**params, "limit": limit, "offset": offset}).fetchall()]
    if not ids:
        return {"total": total, "page": page, "limit": limit, "data": []}

    rows = db.execute(text("""
        SELECT pd.nct_number, pd.study_title, pd.study_url, pd.acronym, pd.status,
               pd.conditions, pd.phase, pd.sponsor, pd.enrollment_count,
               te.min_age_years, te.max_age_years, te.sex, te.healthy_volunteers,
               l.location_text, l.city, l.country_code,
               ST_Y(l.geom) AS latitude, ST_X(l.geom) AS longitude,
               l.geocode_precision
        FROM processed_data pd
        LEFT JOIN trial_eligibility te ON te.nct_number = pd.nct_number
        LEFT JOIN study_locations sl ON pd.nct_number = sl.study_nct_number
        LEFT JOIN locations l ON sl.location_id = l.id
        WHERE pd.nct_number = ANY(:ids)
          AND (l.geom IS NOT NULL OR l.id IS NULL)
    """), {"ids": ids}).mappings().all()

    studies: dict[str, dict] = {}
    for r in rows:
        s = studies.setdefault(r["nct_number"], {
            "nct_number": r["nct_number"], "study_title": r["study_title"],
            "study_url": r["study_url"], "acronym": r["acronym"], "status": r["status"],
            "conditions": r["conditions"], "phase": r["phase"], "sponsor": r["sponsor"],
            "enrollment_count": r["enrollment_count"],
            "min_age_years": r["min_age_years"], "max_age_years": r["max_age_years"],
            "sex": r["sex"], "healthy_volunteers": r["healthy_volunteers"],
            "locations": [],
        })
        if r["location_text"] and r["latitude"] is not None:
            s["locations"].append({
                "location_text": r["location_text"],
                "city": r["city"], "country_code": r["country_code"],
                "latitude": r["latitude"], "longitude": r["longitude"],
                "precision": r["geocode_precision"],
            })

    ordered = sorted(studies.values(), key=lambda s: ids.index(s["nct_number"]))
    return {"total": total, "page": page, "limit": limit,
            "data": [Study.model_validate(s) for s in ordered]}


@main_router.get("/studies/map-data", response_model=MapResponse)
def get_map_data(db=Depends(get_db),
                 bbox: str = Query(..., description="min_lon,min_lat,max_lon,max_lat"),
                 zoom: int = Query(..., description="Current map zoom level"),
                 q: Optional[str] = None, status: Optional[str] = None,
                 phase: Optional[str] = None):
    env = parse_bbox(bbox)
    assert env is not None  # bbox is required, so parse_bbox never returns None
    min_lon, min_lat, max_lon, max_lat = env

    clauses, params = build_filters(q, status, phase)
    filtered = bool(clauses)          # captured before the spatial clause is added
    params.update(dict(zip(("min_lon", "min_lat", "max_lon", "max_lat"), env)))

    # `&&` is the index operator; ST_Intersects would add a full geometry test
    # that a point-in-envelope check does not need.
    clauses.append("l.geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)")
    clauses.append("l.geom IS NOT NULL")
    where = "WHERE " + " AND ".join(clauses)

    if zoom < CLUSTER_THRESHOLD:
        # ── grid resolution ──────────────────────────────────────────────────
        #
        # Cells across the viewport double per zoom level instead of staying at a
        # fixed 64. A fixed grid meant zooming in only re-stamped the same lattice
        # over a smaller area, so the map appeared to stop revealing structure --
        # the opposite of what zooming is for. Doubling keeps the detail growing
        # with the zoom and makes the last few levels before the point threshold
        # essentially one cell per location.
        #
        # The cap is the payload: every cluster becomes a marker in the browser,
        # and the densest viewport holds ~50k locations. Raise CLUSTER_GRID_MAX if
        # the client can take more, lower it if the map feels sluggish.
        cells = min(CLUSTER_GRID_MAX, CLUSTER_GRID_BASE * (2 ** max(0, zoom - 3)))
        sx = (max_lon - min_lon) / cells
        sy = (max_lat - min_lat) / cells

        # ST_SnapToGrid returns the cell's lower-left corner, so the result is
        # translated by half a cell -- otherwise every cluster renders half a cell
        # away from the points it represents. The original passed five arguments
        # here, which PostGIS reads as (geom, size_x, size_y, size_z, size_m) and
        # which collapsed every point into one cluster.
        lat_expr = "ST_Y(ST_Translate(ST_SnapToGrid(l.geom, :sx, :sy), :sx / 2.0, :sy / 2.0))"
        lon_expr = "ST_X(ST_Translate(ST_SnapToGrid(l.geom, :sx, :sy), :sx / 2.0, :sy / 2.0))"

        # ── why the join is conditional ──────────────────────────────────────
        #
        # Joining through study_locations and processed_data exists only to
        # support the text/status/phase filters. Without them it contributes
        # nothing, and it costs a great deal: at the national view the planner
        # gives up on the index and sequentially scans all 3.4M rows of
        # study_locations, measured at 933 ms. Dropping the join when there is
        # nothing to filter on makes the same request index-only against
        # locations.geom, measured at 106 ms.
        join_sql = (
            "JOIN study_locations sl ON sl.location_id = l.id "
            "JOIN processed_data pd ON pd.nct_number = sl.study_nct_number"
            if filtered else ""
        )

        rows = db.execute(text(f"""
            SELECT count(*) AS point_count,
                   {lat_expr} AS latitude,
                   {lon_expr} AS longitude,
                   (array_agg(l.id))[1] AS sample_location_id
            FROM locations l
            {join_sql}
            {where}
            GROUP BY 2, 3
        """), {**params, "sx": sx, "sy": sy}).mappings().all()

        # A cell holding exactly one location is not a cluster -- it is a pin, and
        # returning it as a cluster meant a circle with a "1" in it, which is both
        # ugly and unclickable in any useful way. Those cells are resolved to the
        # trial behind them, so the client can render a real marker and open a
        # popup for it.
        singletons = [r["sample_location_id"] for r in rows
                      if r["point_count"] == 1 and r["sample_location_id"] is not None]
        by_location: dict[int, list[dict]] = {}
        location_total: dict[int, int] = {}
        if singletons:
            # ROW_NUMBER rather than DISTINCT ON: a site can host thousands of
            # studies and the UI wants the five most recent, not an arbitrary one.
            # The window count gives the true total without a second query.
            detail = db.execute(text("""
                WITH ranked AS (
                    SELECT l.id AS location_id, l.location_text,
                           ST_Y(l.geom) AS latitude, ST_X(l.geom) AS longitude,
                           l.geocode_precision, l.city, l.country_code,
                           pd.nct_number, pd.study_title, pd.study_url, pd.acronym,
                           pd.status, pd.conditions, pd.phase, pd.sponsor,
                           pd.enrollment_count,
                           te.min_age_years, te.max_age_years, te.sex,
                           te.healthy_volunteers,
                           count(*) OVER (PARTITION BY l.id) AS total_at_site,
                           row_number() OVER (PARTITION BY l.id
                                              ORDER BY pd.last_updated_date DESC NULLS LAST,
                                                       pd.nct_number) AS rn
                    FROM locations l
                    JOIN study_locations sl ON sl.location_id = l.id
                    JOIN processed_data pd ON pd.nct_number = sl.study_nct_number
                    LEFT JOIN trial_eligibility te ON te.nct_number = pd.nct_number
                    WHERE l.id = ANY(:ids)
                )
                SELECT * FROM ranked WHERE rn <= 5
            """), {"ids": singletons}).mappings().all()
            for d in detail:
                by_location.setdefault(d["location_id"], []).append(dict(d))
                location_total[d["location_id"]] = d["total_at_site"]

        def as_study(d: dict) -> Study:
            return Study.model_validate({
                "nct_number": d["nct_number"],
                "study_title": d["study_title"],
                "study_url": d["study_url"],
                "acronym": d["acronym"],
                "status": d["status"],
                "conditions": d["conditions"],
                "phase": d["phase"],
                "sponsor": d["sponsor"],
                "enrollment_count": d["enrollment_count"],
                "min_age_years": d["min_age_years"],
                "max_age_years": d["max_age_years"],
                "sex": d["sex"],
                "healthy_volunteers": d["healthy_volunteers"],
                "locations": [{
                    "location_text": d["location_text"],
                    "city": d["city"], "country_code": d["country_code"],
                    "latitude": d["latitude"],
                    "longitude": d["longitude"],
                    "precision": d["geocode_precision"],
                }],
            })

        out: list = []
        for r in rows:
            if r["point_count"] == 1:
                loc_id = r["sample_location_id"]
                found = by_location.get(loc_id) or []
                if found:
                    total = location_total.get(loc_id, len(found))
                    # Coordinates come from the location itself, not the grid
                    # cell centre, which is offset by half a cell.
                    lat, lon = found[0]["latitude"], found[0]["longitude"]
                    if total > 1:
                        out.append(Site(
                            point_count=total,
                            latitude=lat,
                            longitude=lon,
                            trials=[as_study(d) for d in found],
                        ))
                    else:
                        out.append(as_study(found[0]))
                    continue
            out.append(Cluster(**{k: r[k] for k in
                                  ("point_count", "latitude", "longitude")}))
        return out

    rows = db.execute(text(f"""
        SELECT pd.nct_number, pd.study_title, pd.study_url, pd.acronym, pd.status,
               pd.conditions, pd.phase, pd.sponsor, pd.enrollment_count,
               te.min_age_years, te.max_age_years, te.sex, te.healthy_volunteers,
               l.location_text, l.city, l.country_code,
               ST_Y(l.geom) AS latitude, ST_X(l.geom) AS longitude,
               l.geocode_precision
        FROM locations l
        JOIN study_locations sl ON sl.location_id = l.id
        JOIN processed_data pd ON pd.nct_number = sl.study_nct_number
        LEFT JOIN trial_eligibility te ON te.nct_number = pd.nct_number
        {where}
    """), params).mappings().all()

    studies: dict[str, dict] = {}
    for r in rows:
        s = studies.setdefault(r["nct_number"], {
            "nct_number": r["nct_number"], "study_title": r["study_title"],
            "study_url": r["study_url"], "acronym": r["acronym"], "status": r["status"],
            "conditions": r["conditions"], "phase": r["phase"], "sponsor": r["sponsor"],
            "enrollment_count": r["enrollment_count"],
            "min_age_years": r["min_age_years"], "max_age_years": r["max_age_years"],
            "sex": r["sex"], "healthy_volunteers": r["healthy_volunteers"],
            "locations": [],
        })
        s["locations"].append({
            "location_text": r["location_text"],
            "latitude": r["latitude"], "longitude": r["longitude"],
            "precision": r["geocode_precision"],
        })
    return [Study.model_validate(s) for s in studies.values()]


# ── sessions ────────────────────────────────────────────────────────────────

session_router = APIRouter(prefix="/api/session", tags=["Session"])


@session_router.post("", response_model=SessionInfo)
@session_router.post("/", response_model=SessionInfo, include_in_schema=False)
def create_session(request: Request, response: Response):
    conn = get_conn(request)
    try:
        _sid, token = sessions_mod.issue(conn)
        conn.commit()
    finally:
        conn.close()
    sessions_mod.set_cookie(response, token)
    return {"display_name": "guest"}


@session_router.get("", response_model=SessionInfo)
@session_router.get("/", response_model=SessionInfo, include_in_schema=False)
def read_session(request: Request, response: Response):
    """Return the current session, minting one if the caller has none."""
    conn = get_conn(request)
    try:
        session = sessions_mod.resolve(
            conn, request.cookies.get(sessions_mod.COOKIE_NAME))
        if session is None:
            _sid, token = sessions_mod.issue(conn)
            conn.commit()
            sessions_mod.set_cookie(response, token)
            return {"display_name": "guest"}
        return {"display_name": session["display_name"]}
    finally:
        conn.close()


# ── chat ────────────────────────────────────────────────────────────────────

chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])


def _require_session(request: Request):
    conn = get_conn(request)
    try:
        session = sessions_mod.resolve(
            conn, request.cookies.get(sessions_mod.COOKIE_NAME))
    finally:
        conn.close()
    if session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "no valid session")
    return session


def _owns(db, conversation_id: str, session_id: str) -> bool:
    return db.execute(text(
        "SELECT 1 FROM conversations WHERE id = :c AND session_id = :s"
    ), {"c": conversation_id, "s": session_id}).first() is not None


@chat_router.get("/conversations", response_model=List[Conversation])
def list_conversations(request: Request, db=Depends(get_db)):
    session = _require_session(request)
    rows = db.execute(text("""
        SELECT id, created_at, updated_at FROM conversations
        WHERE session_id = :s ORDER BY updated_at DESC
    """), {"s": session["id"]}).mappings().all()
    return [{k: str(v) for k, v in r.items()} for r in rows]


@chat_router.post("/conversations", response_model=Conversation, status_code=201)
def create_conversation(request: Request, db=Depends(get_db)):
    session = _require_session(request)
    with db.begin():
        row = db.execute(text(
            "INSERT INTO conversations (session_id) VALUES (:s) "
            "RETURNING id, created_at, updated_at"
        ), {"s": session["id"]}).mappings().one()
    return {k: str(v) for k, v in row.items()}


@chat_router.delete("/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, request: Request, db=Depends(get_db)):
    session = _require_session(request)
    with db.begin():
        if not _owns(db, conv_id, session["id"]):
            # 404 rather than 403 for someone else's conversation: a 403 confirms
            # the id exists, which is a small disclosure.
            raise HTTPException(404, "conversation not found")
        db.execute(text("DELETE FROM conversations WHERE id = :c"), {"c": conv_id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@chat_router.get("/conversations/{conv_id}/messages", response_model=List[Message])
def get_messages(conv_id: str, request: Request, db=Depends(get_db)):
    session = _require_session(request)
    if not _owns(db, conv_id, session["id"]):
        raise HTTPException(404, "conversation not found")
    rows = db.execute(text("""
        SELECT id, conversation_id, sender_id, content, sent_at
        FROM messages WHERE conversation_id = :c
        ORDER BY sent_at ASC, id ASC
    """), {"c": conv_id}).mappings().all()
    return [{k: str(v) for k, v in r.items()} for r in rows]


def _search_trials(db, q=None, status=None, phase=None, limit=5):
    """The model's only tool.

    Returns rows straight out of the database, so every NCT number in a reply is
    a real record. The model never supplies a trial it was not handed.
    """
    clauses, params = build_filters(q, status, phase)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    try:
        params["limit"] = max(1, min(int(limit or 5), 25))
    except (TypeError, ValueError):
        params["limit"] = 5
    rows = db.execute(text(f"""
        SELECT nct_number, study_title, status, phase, conditions, sponsor
        FROM processed_data pd {where}
        ORDER BY last_updated_date DESC NULLS LAST
        LIMIT :limit
    """), params).mappings().all()
    return {"studies": [dict(r) for r in rows]}


def _list_protocols(db):
    """Which criteria sets the assistant can screen against."""
    return {"criteria": qualification.list_criteria(),
            "matcher": qualification.describe_matcher()}


def _screen_eligibility(db, criteria_id=None, answers=None):
    """Run one round of eligibility screening and return the next questions.

    Deliberately stateless: the assistant holds the accumulated answers in the
    conversation and passes them back each turn. That means the screening can be
    resumed after a dropped connection, corrected when a patient changes an
    answer, and reproduced from an exported transcript.
    """
    c = qualification.get_criteria(criteria_id or "")
    if not c:
        return {"error": f"unknown criteria_id {criteria_id!r}",
                "available": [x["id"] for x in qualification.list_criteria()]}
    if not isinstance(answers, dict):
        answers = {}
    return qualification.load_matcher().evaluate(c, answers).as_dict()


def _narrow_trials(db, condition=None, status=None, answers=None):
    """Pick the next question that best splits the remaining trials.

    Pure arithmetic over the candidate pool (entropy of the partition), not a
    model call: it is free, deterministic, and cannot invent a question whose
    answer means nothing.
    """
    conn = db.get_bind().raw_connection()
    try:
        cur = conn.cursor()
        return qualification.segmenter.next_question(
            cur, condition, status, answers or {})
    finally:
        conn.close()


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_clinical_trials",
            "description": "Search the clinical trials database by condition, "
                           "recruitment status, and phase. Always use this rather "
                           "than answering from memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Condition or keywords, e.g. 'melanoma'"},
                    "status": {"type": "string", "enum": sorted(KNOWN_STATUSES)},
                    "phase": {"type": "string",
                              "description": "PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1"},
                    "limit": {"type": "integer", "description": "Max results, default 5"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_qualification_protocols",
            "description": "List the eligibility criteria sets that can be "
                           "screened against conversationally.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_eligibility",
            "description": (
                "Screen a person against a protocol's inclusion/exclusion criteria. "
                "Pass every answer collected so far; the call is stateless. It "
                "returns a verdict (likely_eligible, ineligible, needs_info), the "
                "criteria that blocked, and the next questions to ask. Never ask "
                "for information it does not request, and never decide eligibility "
                "yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria_id": {"type": "string",
                                    "description": "from list_qualification_protocols"},
                    "answers": {
                        "type": "object",
                        "description": (
                            "Answers collected so far, keyed by field name from the "
                            "question objects. Booleans as true/false, numbers as "
                            "integers, enums as their option string."
                        ),
                    },
                },
                "required": ["criteria_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "narrow_trials",
            "description": (
                "Choose the single most informative next question for narrowing a "
                "set of candidate trials. Returns a question, how many trials "
                "remain, and how many a yes or a no would remove. Use this instead "
                "of inventing questions. Collect age, sex and healthy-volunteer "
                "status as form fields first; this covers everything else."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "e.g. 'melanoma'"},
                    "status": {"type": "string", "enum": sorted(KNOWN_STATUSES)},
                    "answers": {
                        "type": "object",
                        "description": (
                            "Everything answered so far. age (integer), sex "
                            "('female'|'male'), healthy_volunteer (boolean), and "
                            "facts as {\"prior_therapy:chemotherapy\": true}. "
                            "Answer keys come from previous responses."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
]

# name -> (callable, accepted argument names). The filter matters: the model picks
# argument names, and a stray key is a TypeError that would otherwise escape as a
# 500 -- which a browser reports as a CORS failure.
TOOL_REGISTRY = {
    "search_clinical_trials": (_search_trials, {"q", "status", "phase", "limit"}),
    "list_qualification_protocols": (_list_protocols, set()),
    "screen_eligibility": (_screen_eligibility, {"criteria_id", "answers"}),
    "narrow_trials": (_narrow_trials, {"condition", "status", "answers"}),
}

SYSTEM_PROMPT = (
    "You are a clinical trial discovery assistant. You help people find trials and "
    "work out whether a trial's stated criteria could apply to them.\n\n"
    "HOW TO WORK, IN ORDER:\n"
    "1. Never state a fact about trials from memory. Call search_clinical_trials "
    "first, and cite every trial by its NCT number verbatim. If a tool returns "
    "nothing, say so plainly -- a fabricated NCT number is worse than no answer.\n"
    "2. Collect age, sex and healthy-volunteer status ONCE, together, as a short "
    "form. Do not drip these out one per turn; they are the same questions every "
    "form already asks and making someone answer them in chat is tedious.\n"
    "3. Then narrow. Call narrow_trials with the accumulated answers. It returns "
    "the single most informative question, already phrased for a patient, plus "
    "how many trials remain. Ask exactly that question, one per turn, and pass "
    "every answer back next time. Never invent your own question.\n"
    "4. When narrow_trials returns no question, the pool is as small as it gets. "
    "Call search_clinical_trials and present the shortlist with NCT numbers.\n"
    "5. If the person names a specific protocol from "
    "list_qualification_protocols, switch to screen_eligibility and ask its "
    "`next_questions` the same way.\n\n"
    "BOUNDARIES:\n"
    "6. You are screening, not deciding. Never say someone is eligible or "
    "ineligible -- say which criteria they appear to meet or not meet, and that "
    "the study team confirms it with tests.\n"
    "7. If someone declines a question, move on. It is reported as unknown and "
    "never assumed either way.\n"
    "8. Be conversational and brief. No questionnaires, no walls of text.\n"
    "9. This is health information, not medical advice. Encourage them to raise "
    "anything concerning with their care team."
)


@chat_router.post("/conversations/{conv_id}/messages", response_model=Message, status_code=201)
def post_message(conv_id: str, body: dict, request: Request, db=Depends(get_db)):
    session = _require_session(request)
    content = (body or {}).get("content")
    if not content or not str(content).strip():
        raise HTTPException(400, "message content cannot be empty")
    if not _owns(db, conv_id, session["id"]):
        raise HTTPException(404, "conversation not found")

    # Explicit commits rather than `with db.begin():`.
    #
    # `_run_llm` runs SELECTs through this same Session, and SQLAlchemy
    # autobegins a transaction on the first execute. A subsequent `db.begin()`
    # then raises "A transaction is already begun on this Session" -- which
    # escaped as an unhandled 500, which Starlette turned into a response with
    # no CORS headers, which the browser reported as a CORS failure. The real
    # error was never CORS.
    db.execute(text("""
        INSERT INTO messages (conversation_id, sender_id, content)
        VALUES (:c, :s, :m)
    """), {"c": conv_id, "s": session["id"], "m": content})
    db.commit()

    history = db.execute(text("""
        SELECT sender_id, content FROM messages
        WHERE conversation_id = :c ORDER BY sent_at DESC, id DESC LIMIT 10
    """), {"c": conv_id}).mappings().all()

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in reversed(history):
        msgs.append({
            "role": "user" if h["sender_id"] == session["id"] else "assistant",
            "content": h["content"],
        })

    reply = _run_llm(db, msgs)

    # Close whatever transaction the tool queries left open before writing.
    db.commit()

    row = db.execute(text("""
        INSERT INTO messages (conversation_id, sender_id, content)
        VALUES (:c, :s, :m)
        RETURNING id, conversation_id, sender_id, content, sent_at
    """), {"c": conv_id, "s": sessions_mod.AI_SENDER_ID, "m": reply}).mappings().one()
    db.execute(text("UPDATE conversations SET updated_at = now() WHERE id = :c"),
               {"c": conv_id})
    db.commit()

    return {k: str(v) for k, v in row.items()}


def _run_llm(db, msgs: list[dict]) -> str:
    """One tool round-trip, then a final answer.

    Degrades to an explanation rather than a 500 when the model is unreachable,
    because a chat failure should not look like a platform outage.
    """
    if not LLM_API_KEY:
        return ("Chat is not configured on this deployment (no model API key). "
                "Trial search and the map are unaffected.")

    try:
        from openai import OpenAI
    except ImportError:
        return "Chat is unavailable: the `openai` package is not installed."

    client = (OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_BASE_URL
              else OpenAI(api_key=LLM_API_KEY))

    try:
        first = client.chat.completions.create(
            model=CHAT_MODEL, messages=msgs, tools=TOOL_SCHEMA, tool_choice="auto")
    except Exception as exc:  # noqa: BLE001
        return f"The model is unreachable right now ({type(exc).__name__})."

    # An empty `choices` list is legal in some OpenAI-compatible implementations
    # and would be an IndexError otherwise.
    if not getattr(first, "choices", None):
        return "The model returned no answer."

    choice = first.choices[0].message
    calls = getattr(choice, "tool_calls", None)
    if not calls:
        return choice.content or "No answer was produced."

    msgs.append(choice)

    for call in calls:
        name = call.function.name
        # The model chooses the argument names, so they are filtered against the
        # function's real signature. A stray key would be a TypeError, which used
        # to escape as a 500 -- and a 500 without CORS headers reads, in a
        # browser, as a CORS failure.
        fn, allowed = TOOL_REGISTRY.get(name, (None, set()))

        try:
            raw_args = json.loads(call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            raw_args = {}
        if not isinstance(raw_args, dict):
            raw_args = {}

        if fn is None:
            result = {"error": f"unknown tool {name!r}",
                      "available": sorted(TOOL_REGISTRY)}
        else:
            args = {k: v for k, v in raw_args.items() if k in allowed}
            try:
                result = fn(db, **args)
            except Exception as exc:  # noqa: BLE001
                # A failed tool should become a result the model can explain, not
                # an exception that takes down the request.
                db.rollback()
                result = {"error": f"{name} failed: {type(exc).__name__}"}

        msgs.append({"role": "tool", "tool_call_id": call.id,
                     "name": name, "content": json.dumps(result, default=str)})

    try:
        second = client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
        if not getattr(second, "choices", None):
            return "The model returned no answer."
        return second.choices[0].message.content or "No answer was produced."
    except Exception as exc:  # noqa: BLE001
        return f"The model is unreachable right now ({type(exc).__name__})."


# ── routers ─────────────────────────────────────────────────────────────────

app.include_router(main_router)
app.include_router(session_router)
app.include_router(boundaries_router)
app.include_router(qualification_router)

# Mounted at both spellings: the router declares prefix "/api/correlation" with a
# "/" route, so only "/api/correlation/" is a real path and "/api/correlation"
# 307-redirects -- which a browser reports as a CORS failure rather than as a
# redirect. Registering both removes the redirect entirely.
app.include_router(county_data_router)
app.include_router(county_data_router, prefix="/api")

if ENABLE_CHAT:
    app.include_router(chat_router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"starting on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
