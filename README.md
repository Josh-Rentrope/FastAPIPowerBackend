# Augmented Clinical Trials API

A FastAPI service over a PostgreSQL/PostGIS database of the full
ClinicalTrials.gov registry, with a geocoded map view and a county-level
correlation layer. The front end is
[`Josh-Rentrope/health-trials-app`](https://github.com/Josh-Rentrope/health-trials-app).

This was originally a working demo deployed in 2025 and then lost. It has been
rebuilt so that the whole thing is reproducible from this repository plus the
source CSV, which it previously was not.

## The dataset

| | |
|---|---|
| Studies | **607,160** (ClinicalTrials.gov API v2 export, 1.9 GB, 30 columns) |
| Location entries | **3,546,381** — free text, 262 countries |
| Distilled locations | **448,855** after city-level deduplication |
| Study-location pairs | **3,405,775** |
| Placed on a map | **99.8%** (447,761 of 448,855) |
| US counties | 3,235 boundaries, 3,152 with CHR measures |

## Pipeline

Seven stages, each independently re-runnable and idempotent. Run them from the
repository root with `uv run python <stage>`.

| Stage | What it does |
|---|---|
| `db/00_setup_database.py` | Extensions, tables, migrations, indexes. Three ordered phases — see below. |
| `lookups/00_fetch_reference_data.py` | Downloads GeoNames, Census ZCTA + TIGER, CHR&R (~125 MB, once) |
| `lookups/01_load_gazetteers.py` | Loads them into `geo_*` tables |
| `etl/01_load_raw.py` | 30-column CSV → `raw_data` via `COPY`, idempotent by SHA-256 |
| `etl/02_build_processed.py` | `raw_data` → `processed_data` (the transform) |
| `etl/03_extract_locations.py` | Parses 3.5M location strings → `locations` + `study_locations` |
| `etl/04_geocode_locations.py` | The tiered offline geocoder |
| `etl/05_load_counties.py` | TIGER county boundaries + County Health Rankings, nationally |
| `etl/06_verify.py` | Verification gate + `EXPLAIN ANALYZE`; exits non-zero on failure |

The API is `FastAPIPostgresServer.py`; chat identity is `sessions.py`; the
correlation and boundary endpoints are in `api/`.

### Setup order matters

`db/00_setup_database.py` runs in three phases — **tables, then migrations, then
indexes** — and the order is load-bearing. `CREATE TABLE IF NOT EXISTS` no-ops on
an existing table, so a column added in a later revision is absent on an existing
database. If indexes were created alongside their tables, an index naming that new
column would fail before the `ALTER` that adds it had run, aborting the update.
This is what makes the script safe to point at a database built by the previous
version.

## Geocoding

Fully offline, $0, and no API calls at query time. The alternative was ~3.5M
requests: about $17k at typical rates, or ~40 days rate-limited.

Parsing works **from the right**, because facility names contain commas
(`"University of California, Los Angeles, Los Angeles, California, 90095, United
States"`). Two shapes are genuinely ambiguous by shape alone — a three-field
string is either `(facility, city, country)` or `(city, state, country)` — and are
resolved by asking whether the candidate field is a **real administrative
division for that country**, which is why the gazetteers load first.

Every point records which tier produced it:

| tier | method | precision | confidence |
|---|---|---|---|
| 0 | US postal code → Census ZCTA centroid | `postal` | 95 |
| 1 | city + admin1 → GeoNames populated place | `city` | 90 |
| 2 | city → GeoNames populated place | `city` | 65 |
| 3 | admin1 centroid | `admin1` | 45 |
| 4 | country centroid | `country` | 20 |
| 5 | nothing matched | `none` | 0 |

Actual distribution: **city 24.9%, postal 6.9%, admin1 53.9%, country 14.1%,
none 0.2%**.

`geocode_precision` is not decoration. A country centroid is a real position for
"somewhere in Chile" but it is not a location, so the API returns it and the
county-correlation join excludes anything below `city`.

### Normalisation is where the accuracy is

Four things had to be right, and each was invisible until measured:

- **Both spellings of every place name.** GeoNames stores a local name and an
  ASCII transliteration (Roma/Rome, München/Munich, Köln/Cologne). Indexing only
  the ASCII form silently lost every city whose English name differs, and the same
  bug in `geo_admin1` lost "Masovian Voivodeship".
- **Postal codes that the country pattern does not accept.** Sweden's pattern
  requires a space (`549 49`), Norway and Hungary prefix theirs (`N-3515`,
  `H-1027`), Brazil's expects a hyphen, and US zips arrive as nine bare digits.
- **Region suffixes.** "Porto District" vs the gazetteer's "Porto"; "Beijing
  Municipality" vs "Beijing". Stripping the suffix from *both* sides is what makes
  them meet.
- **Raised statistics.** With default statistics the planner estimated the spatial
  semi-join badly enough to prefer a parallel sequential scan of all 3.4M
  `study_locations` rows over an index scan touching 379 of them: 310 ms vs 150 ms.

## Measured performance

Against the full 607k-study database, local PostgreSQL 16, 200 requests each:

| Query | p50 | p95 |
|---|---|---|
| `/studies/map-data` zoom 6 (clustered) | **15.8 ms** | 26.9 ms |
| `/studies` list, 10 rows | **50.7 ms** | 67.2 ms |
| `/studies/map-data` zoom 3 (whole US) | **111.5 ms** | 147.7 ms |
| `/studies?q=melanoma` (trigram) | **26.4 ms** | 46.7 ms |
| `/studies` with a bbox filter | 248.3 ms | 265.2 ms |
| `/studies/map-data` zoom 14 (points, SF) | 498.2 ms | 550.3 ms |

All plans were confirmed with `EXPLAIN (ANALYZE, BUFFERS)` to use the intended
index — `idx_locations_geom` for spatial, `idx_pd_title_trgm` for text,
`idx_pd_status_updated` for status. `etl/06_verify.py` gates on this and exits
non-zero if a plan regresses to a sequential scan.

## Bugs fixed in the rewrite

Each of these was live, and several were silent:

- **The map clustering was wrong.** The cluster query called `ST_SnapToGrid` with
  five arguments. PostGIS reads that as `(geom, size_x, size_y, size_z, size_m)` —
  *not* an origin form — so `min_lon`/`min_lat` became the x/y cell sizes (hundreds
  of degrees) and every point in the viewport collapsed into one cluster. It looked
  plausible because a single large cluster is not obviously wrong on a map.
- **The bbox filter was a correlated subquery per row**, running the join for all
  607k studies. Now a semi-join driven by the GiST index, using `&&` (index-only)
  rather than `ST_Intersects`.
- **A missing geometry crashed the response.** `latitude`/`longitude` are
  non-nullable but `geom` can be NULL, so one ungeocoded location was a Pydantic
  500 away.
- **`phase ILIKE '%PHASE1%'` also matched `EARLY_PHASE1`** — a wrong answer, not a
  slow one. Now token membership.
- **The status filter knew 3 of the 13 real statuses**, so TERMINATED or SUSPENDED
  silently returned nothing.
- **`/api/correlation` redirected** (307), which a browser reports as a CORS
  failure rather than as a redirect.
- **SQL injection in the correlation endpoint.** The guard was
  `if not v.isalnum() and '_' not in v` — inverted in effect, since the intended
  value `v001_raw_value` contains an underscore and so was never rejected. A value
  like `x_1) OR 1=1--` passed. Now an allowlist; verified 400.
- **Chat identity was a request header.** `X-User-Email`, which the shipped front
  end hardcoded to a single UUID while ignoring the value it was passed — so every
  visitor shared one identity and could read every other visitor's conversations.
  Replaced with server-issued sessions; only the token's SHA-256 is stored.
  Verified: no cookie → 401, and a second session reading the first's conversation
  → 404.
- **Three CHR measure codes had drifted** in the 2025 release (v021→v053,
  v043→v085, v057→v024), resolved by label rather than by code.
- **`CHRNRSetup.py` hardcoded a North Carolina FIPS prefix** (`f"37{...}"`), so
  every county outside NC would have been written with a fabricated FIPS.
- **30 of 3,235 county polygons were invalid** (nested shells from naive ring
  assembly); resolved with `ST_MakeValid`.

## Configuration

All environment variables; nothing is hardcoded.

```
DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD DB_SSLMODE   (sslmode defaults to require)
CORS_ORIGINS                 comma-separated exact origins, no wildcards
CT_SESSION_COOKIE_SECURE     "0" only for plain-http local development
ENABLE_CHAT                  "1" to mount the chat router
OPENAI_API_KEY
OPENAI_BASE_URL              any OpenAI-compatible endpoint; empty means OpenAI
CHAT_MODEL
```

`OPENAI_BASE_URL` exists so the chat model can be DeepSeek or a local Ollama. A
local model is the only configuration in which the questions asked about trials do
not leave the operator's network.

The chat tool loop calls exactly one function, `search_clinical_trials`, which runs
parameterised SQL. Every NCT number in a reply is a row the database returned, so
a fabricated trial cannot reach a user — the failure mode that matters most for
clinical content.

## Running locally

```bash
docker run -d --name ct-pg \
  -e POSTGRES_PASSWORD=dev -e POSTGRES_DB=clinical_trials -e POSTGRES_USER=postgres \
  -p 5432:5432 postgis/postgis:16-3.4

uv sync --python 3.12
export DB_HOST=localhost DB_PORT=5432 DB_NAME=clinical_trials DB_USER=postgres DB_PASSWORD=dev

uv run python db/00_setup_database.py --sslmode prefer
# ...then the stages in order. Every stage takes --sslmode prefer for local use.
uvicorn FastAPIPostgresServer:app --port 8000
```

## Attribution

Place data is from [GeoNames](https://www.geonames.org/), licensed CC BY 4.0 —
attribution is required. Postal centroids and county boundaries are US Census
public domain. County Health Rankings & Roadmaps measures are redistributed here
without their terms having been checked; **confirm those before publishing this**.
