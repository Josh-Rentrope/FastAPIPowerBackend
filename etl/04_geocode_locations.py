#!/usr/bin/env python3
"""Resolve `locations.geom` offline, in tiers, recording how each point was found.

## Why not a geocoding service

The input is **3,546,381 location entries** collapsing to a few hundred thousand
distinct places. At the usual ~$5 per 1,000 lookups that is five figures, and a
free service is rate-limited to roughly one request per second -- about 40 days of
wall clock. Neither is acceptable for a demo, so everything here runs against
local reference data.

## The tier ladder

Each tier runs only over rows still lacking a geometry, so they cannot overwrite
one another. Every row records the tier that produced it:

    0  US postal code -> Census ZCTA centroid      census_zcta       postal   conf 95
    1  city + admin1  -> GeoNames populated place  geonames_city     city     conf 90
    2  city only      -> GeoNames populated place  geonames_city     city     conf 65
    3  admin1 centroid                             geonames_admin1   admin1   conf 45
    4  country centroid                            geonames_country  country  conf 20
    5  nothing matched                             --                none     conf  0

Tier 0 is the only one giving street-level accuracy, and it is US-only because the
Census publishes a ZCTA centroid for every ZIP and nobody publishes an equivalent
for the world. Tiers 3 and 4 are deliberately coarse: a country centroid is a real
position for "somewhere in Chile" but it is not a location, which is exactly why
`geocode_precision` is stored rather than assumed. Consumers can and should filter
on it -- the county-correlation join excludes anything below `city`.

Tier 1 beats tier 2 on a genuine signal (the admin1 division agreed), and where a
country has many same-named places the more populous one wins, which is right far
more often than not.

Usage:
    uv run python etl/04_geocode_locations.py
    uv run python etl/04_geocode_locations.py --force     # re-geocode everything
"""

from __future__ import annotations

import argparse
import os
import time

import psycopg2

# Best populated place per (country, admin1, city) and per (country, city).
# DISTINCT ON with a population-ordered sort makes the choice deterministic, so
# re-running produces identical results rather than a different "Portland".
BUILD_MATCH_TABLES = """
-- Every searchable spelling of a populated place, with a priority so an exact
-- name always beats a suffix-stripped one.
--
-- * prio 0 -- the ASCII name and the local name. GeoNames gives them separately
--   and they differ often (Roma/Rome, Munchen/Munich, Koln/Cologne). Matching on
--   the ASCII form alone was silently losing every city with an English name.
-- * prio 1 -- the same names with a trailing "City" removed, so "Kaohsiung City"
--   finds Kaohsiung. Lower priority because "Kansas City" must not become
--   "Kansas" when something genuinely means Kansas.
CREATE TEMP TABLE city_names AS
SELECT geonameid, country_code, admin1_code, name, latitude, longitude, population,
       normalized_name AS nname, 0 AS prio
FROM geo_cities
UNION ALL
SELECT geonameid, country_code, admin1_code, name, latitude, longitude, population,
       normalized_name_local, 0
FROM geo_cities
WHERE normalized_name_local IS NOT NULL AND normalized_name_local <> normalized_name
UNION ALL
SELECT geonameid, country_code, admin1_code, name, latitude, longitude, population,
       regexp_replace(normalized_name, '\\s+(city|ciudad|stadt|ville)$', ''), 1
FROM geo_cities
WHERE normalized_name ~ '\\s+(city|ciudad|stadt|ville)$'
  AND length(regexp_replace(normalized_name, '\\s+(city|ciudad|stadt|ville)$', '')) >= 4;

CREATE TEMP TABLE city_by_admin1 AS
SELECT DISTINCT ON (country_code, admin1_code, nname)
       country_code, admin1_code, nname, name, geonameid, latitude, longitude
FROM city_names
WHERE country_code IS NOT NULL AND admin1_code IS NOT NULL
ORDER BY country_code, admin1_code, nname, prio, population DESC NULLS LAST;

CREATE INDEX ON city_by_admin1 (country_code, admin1_code, nname);

CREATE TEMP TABLE city_by_country AS
SELECT DISTINCT ON (country_code, nname)
       country_code, nname, name, geonameid, latitude, longitude
FROM city_names
WHERE country_code IS NOT NULL
ORDER BY country_code, nname, prio, population DESC NULLS LAST;

CREATE INDEX ON city_by_country (country_code, nname);

-- admin1 name -> canonical code.
--
-- admin2 rows ('#2' codes) ARE included, at lower priority. The parser recognises
-- admin2 names -- the export's middle field is sometimes a sub-region such as the
-- UK's "East Midlands" -- so excluding them here meant those rows were recognised
-- at parse time and then silently failed to match, landing on a country centroid.
CREATE TEMP TABLE admin1_name AS
SELECT DISTINCT ON (country_code, nname)
       country_code, nname, admin1_code, latitude, longitude
FROM (
    SELECT country_code, admin1_code, normalized_name AS nname, latitude, longitude, 0 AS prio
    FROM geo_admin1 WHERE admin1_code NOT LIKE '%#%'
    UNION ALL
    SELECT country_code, admin1_code, normalized_name_local, latitude, longitude, 0
    FROM geo_admin1
    WHERE admin1_code NOT LIKE '%#%' AND normalized_name_local IS NOT NULL
      AND normalized_name_local <> normalized_name
    UNION ALL
    SELECT country_code, admin1_code, normalized_name, latitude, longitude, 1
    FROM geo_admin1
    WHERE admin1_code LIKE '%#2%' AND latitude IS NOT NULL
) x
WHERE nname IS NOT NULL
ORDER BY country_code, nname, prio, admin1_code;

CREATE INDEX ON admin1_name (country_code, nname);
"""

# Tier 0. `left(regexp_replace(postal_code,'\\D','','g'),5)` handles "91436" and
# "91436-1234" alike. A shorter-than-5-digit value simply does not match a ZCTA,
# so there is no false positive from padding.
TIER0 = """
UPDATE locations l SET
    geom = ST_SetSRID(ST_MakePoint(gp.longitude, gp.latitude), 4326),
    geocode_source = 'census_zcta',
    geocode_precision = 'postal',
    geocode_confidence = 95,
    matched_name = l.postal_code,
    matched_gazetteer_id = 'zcta:' || gp.postal_code,
    last_geocoded_at = now()
FROM geo_postal gp
WHERE l.geom IS NULL
  AND l.country_code = 'US'
  AND l.postal_code IS NOT NULL
  AND gp.country_code = 'US'
  AND gp.postal_code = left(regexp_replace(l.postal_code, '\\D', '', 'g'), 5)
"""

# Tier 1. Joins through admin1_name so the location's free-text state name is
# compared to the gazetteer's, then demands the city agree on the same division.
TIER1 = """
UPDATE locations l SET
    geom = ST_SetSRID(ST_MakePoint(m.longitude, m.latitude), 4326),
    geocode_source = 'geonames_city',
    geocode_precision = 'city',
    geocode_confidence = 90,
    matched_name = m.name,
    matched_gazetteer_id = 'geonameid:' || m.geonameid,
    last_geocoded_at = now()
FROM city_by_admin1 m
JOIN admin1_name a
  ON a.country_code = m.country_code AND a.admin1_code = m.admin1_code
WHERE l.geom IS NULL
  AND l.country_code IS NOT NULL
  AND l.normalized_city IS NOT NULL
  AND l.normalized_admin1 IS NOT NULL
  AND l.country_code = m.country_code
  AND (l.normalized_city = m.nname OR l.normalized_city_core = m.nname)
  AND l.normalized_admin1 = a.nname
"""

TIER2 = """
UPDATE locations l SET
    geom = ST_SetSRID(ST_MakePoint(m.longitude, m.latitude), 4326),
    geocode_source = 'geonames_city',
    geocode_precision = 'city',
    geocode_confidence = 65,
    matched_name = m.name,
    matched_gazetteer_id = 'geonameid:' || m.geonameid,
    last_geocoded_at = now()
FROM city_by_country m
WHERE l.geom IS NULL
  AND l.country_code IS NOT NULL
  AND l.normalized_city IS NOT NULL
  AND l.country_code = m.country_code
  AND (l.normalized_city = m.nname OR l.normalized_city_core = m.nname)
"""

TIER3 = """
UPDATE locations l SET
    geom = ST_SetSRID(ST_MakePoint(a.longitude, a.latitude), 4326),
    geocode_source = 'geonames_admin1',
    geocode_precision = 'admin1',
    geocode_confidence = 45,
    matched_name = a.admin1_code,
    matched_gazetteer_id = 'admin1:' || a.country_code || '.' || a.admin1_code,
    last_geocoded_at = now()
FROM admin1_name a
WHERE l.geom IS NULL
  AND l.country_code IS NOT NULL
  AND l.normalized_admin1 IS NOT NULL
  AND l.country_code = a.country_code
  AND l.normalized_admin1 = a.nname
  AND a.latitude IS NOT NULL
"""

TIER4 = """
UPDATE locations l SET
    geom = ST_SetSRID(ST_MakePoint(gc.longitude, gc.latitude), 4326),
    geocode_source = 'geonames_country',
    geocode_precision = 'country',
    geocode_confidence = 20,
    matched_name = gc.name,
    matched_gazetteer_id = 'country:' || gc.iso2,
    last_geocoded_at = now()
FROM geo_country gc
WHERE l.geom IS NULL
  AND l.country_code IS NOT NULL
  AND l.country_code = gc.iso2
  AND gc.latitude IS NOT NULL
"""

TIER5 = """
UPDATE locations SET
    geocode_source = 'none',
    geocode_precision = 'none',
    geocode_confidence = 0,
    last_geocoded_at = now()
WHERE geom IS NULL AND (geocode_precision IS NULL OR geocode_precision <> 'none')
"""

TIERS = [
    ("0  US postal -> Census ZCTA", TIER0),
    ("1  city + admin1 -> GeoNames", TIER1),
    ("2  city -> GeoNames", TIER2),
    ("3  admin1 centroid", TIER3),
    ("4  country centroid", TIER4),
    ("5  unmatched", TIER5),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    p.add_argument("--force", action="store_true",
                   help="clear every geometry first and redo the whole ladder")
    args = p.parse_args()

    conn = psycopg2.connect(
        dbname=args.dbname or os.environ.get("DB_NAME"),
        user=args.user or os.environ.get("DB_USER"),
        password=args.password or os.environ.get("DB_PASSWORD"),
        host=args.host, port=args.port, sslmode=args.sslmode,
    )
    conn.autocommit = True
    cur = conn.cursor()

    t0 = time.time()
    if args.force:
        cur.execute("""
            UPDATE locations SET geom = NULL, geocode_source = NULL,
                geocode_precision = NULL, geocode_confidence = NULL,
                matched_name = NULL, matched_gazetteer_id = NULL, last_geocoded_at = NULL
        """)
        print(f"cleared {cur.rowcount:,} geometries for a full re-run")

    cur.execute("SELECT count(*) FROM locations WHERE geom IS NULL AND geocode_precision IS DISTINCT FROM 'none'")
    todo = cur.fetchone()[0]
    print(f"worklist: {todo:,} locations to resolve")

    print("\n-- building match tables --")
    t1 = time.time()
    cur.execute(BUILD_MATCH_TABLES)
    print(f"   done in {time.time()-t1:.1f}s")

    print("\n-- running tiers --")
    for label, sql in TIERS:
        t1 = time.time()
        cur.execute(sql)
        n = cur.rowcount
        print(f"   tier {label:<34} {n:>9,} rows  ({time.time()-t1:.1f}s)")

    print("\n-- result --")
    cur.execute("""
        SELECT geocode_precision, geocode_source, count(*),
               round(100.0 * count(*) / (SELECT count(*) FROM locations), 1) AS pct
        FROM locations
        GROUP BY 1, 2
        ORDER BY 3 DESC
    """)
    print(f"   {'precision':<10} {'source':<18} {'count':>10}  {'pct':>6}")
    for prec, src, n, pct in cur.fetchall():
        print(f"   {str(prec):<10} {str(src):<18} {n:>10,}  {pct:>5}%")

    cur.execute("SELECT count(*) FROM locations WHERE geom IS NOT NULL")
    with_geom = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM locations")
    total = cur.fetchone()[0]
    print(f"\n   placed {with_geom:,} / {total:,} ({100*with_geom/max(total,1):.1f}%)")

    print("\n-- studies with at least one placed location --")
    cur.execute("""
        SELECT count(DISTINCT sl.study_nct_number)
        FROM study_locations sl JOIN locations l ON l.id = sl.location_id
        WHERE l.geom IS NOT NULL
    """)
    placed_studies = cur.fetchone()[0]
    cur.execute("SELECT count(DISTINCT study_nct_number) FROM study_locations")
    all_studies = cur.fetchone()[0]
    print(f"   {placed_studies:,} of {all_studies:,} "
          f"({100*placed_studies/max(all_studies,1):.1f}%)")

    # A sanity check on lat/lon ordering: swapping them is the classic geocoding
    # bug and it is invisible until someone looks at the map.
    print("\n-- geometry sanity --")
    cur.execute("""
        SELECT count(*) FROM locations
        WHERE geom IS NOT NULL
          AND (ST_X(geom) NOT BETWEEN -180 AND 180 OR ST_Y(geom) NOT BETWEEN -90 AND 90)
    """)
    print(f"   out-of-range coordinates: {cur.fetchone()[0]:,}")

    cur.close()
    conn.close()
    print(f"\ndone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
