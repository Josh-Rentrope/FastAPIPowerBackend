#!/usr/bin/env python3
"""Verify a completed load, and measure the queries the API actually runs.

Exits non-zero if any invariant fails, so it can gate a deployment rather than
being something someone reads and shrugs at.

Two halves:

* **Invariants** -- counts agree, no orphan rows, no geometry outside the valid
  range, every location carries a precision label. These catch the failures that
  are otherwise silent: a half-finished stage, a join that matched nothing, a
  lat/lon pair swapped.
* **EXPLAIN ANALYZE** on the two hot queries and the text search. The point is not
  the timing, it is the plan: a sequential scan over 607k rows where an index was
  expected means an index is missing or does not match the query's shape, and that
  is a correctness bug about the schema rather than a slow query.

Usage:
    uv run python etl/06_verify.py --expect-raw 607160
    uv run python etl/06_verify.py --skip-explain      # fast pass
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import urllib.request

import psycopg2

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"   {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(label)


def connect(args):
    dsn = {
        "dbname": args.dbname or os.environ.get("DB_NAME"),
        "user": args.user or os.environ.get("DB_USER"),
        "password": args.password or os.environ.get("DB_PASSWORD"),
        "host": args.host or os.environ.get("DB_HOST", "localhost"),
        "port": args.port or os.environ.get("DB_PORT", "5432"),
        "sslmode": args.sslmode,
    }
    return psycopg2.connect(**dsn)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    p.add_argument("--expect-raw", type=int,
                   help="expected raw_data row count (the full export is 607,160)")
    p.add_argument("--skip-explain", action="store_true")
    p.add_argument("--base-url", help="also time the live API's map endpoint")
    args = p.parse_args()

    conn = connect(args)
    conn.autocommit = True
    cur = conn.cursor()

    print("== counts ==")
    cur.execute("""
        SELECT
          (SELECT count(*) FROM raw_data)                                 AS raw,
          (SELECT count(*) FROM processed_data)                           AS processed,
          (SELECT count(DISTINCT study_nct_number) FROM study_locations)   AS studies_with_site,
          (SELECT count(*) FROM study_locations)                          AS study_locations,
          (SELECT count(*) FROM locations)                                AS locations
    """)
    raw, processed, studies_with_site, study_locations, locations = cur.fetchone()
    for label, value in (("raw_data", raw), ("processed_data", processed),
                         ("locations", locations), ("study_locations", study_locations),
                         ("studies_with_site", studies_with_site)):
        print(f"   {label:<20} {value:>12,}")

    if args.expect_raw:
        check(raw == args.expect_raw, "raw_data row count matches the source file",
              f"({raw:,} vs expected {args.expect_raw:,})")
    # The transform drops rows with no NCT number, so processed should equal raw.
    # A shortfall means some rows were silently unusable.
    check(raw == processed, "every raw row became a processed row",
          f"({raw:,} -> {processed:,})")

    print("\n== orphans ==")
    cur.execute("""
        SELECT count(*) FROM study_locations sl
        LEFT JOIN processed_data pd ON pd.nct_number = sl.study_nct_number
        WHERE pd.nct_number IS NULL
    """)
    check(cur.fetchone()[0] == 0, "no study_locations pointing at a missing study")
    cur.execute("""
        SELECT count(*) FROM study_locations sl
        LEFT JOIN locations l ON l.id = sl.location_id
        WHERE l.id IS NULL
    """)
    check(cur.fetchone()[0] == 0, "no study_locations pointing at a missing location")

    print("\n== geocoding ==")
    cur.execute("SELECT count(*) FROM locations WHERE geocode_precision IS NULL")
    check(cur.fetchone()[0] == 0, "every location carries a precision label",
          "(NULL means the geocode ladder never ran over it)")

    cur.execute("""
        SELECT count(*) FROM locations WHERE geom IS NOT NULL
          AND (ST_SRID(geom) <> 4326
               OR ST_X(geom) NOT BETWEEN -180 AND 180
               OR ST_Y(geom) NOT BETWEEN -90 AND 90)
    """)
    check(cur.fetchone()[0] == 0, "all geometry is SRID 4326 and in range",
          "(a failure here usually means lat/lon were swapped)")

    cur.execute("SELECT count(*) FROM locations")
    total = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM locations WHERE geom IS NOT NULL")
    placed = cur.fetchone()[0]
    check(total == 0 or placed / total > 0.99,
          "at least 99% of locations have a point",
          f"({placed:,}/{total:,} = {100*placed/max(total,1):.1f}%)")

    print("\n   precision distribution:")
    cur.execute("""
        SELECT geocode_precision, count(*),
               round(100.0*count(*)/(SELECT count(*) FROM locations), 1)
        FROM locations GROUP BY 1 ORDER BY 2 DESC
    """)
    for prec, n, pct in cur.fetchall():
        print(f"      {str(prec):<10} {n:>10,}  {pct:>5}%")

    cur.execute("""
        SELECT count(*) FROM locations
        WHERE geocode_precision IN ('postal','city')
    """)
    good = cur.fetchone()[0]
    print(f"\n   city-or-better: {good:,}/{total:,} ({100*good/max(total,1):.1f}%)")

    print("\n== counties ==")
    cur.execute("SELECT count(*) FROM county_boundaries")
    cb = cur.fetchone()[0]
    check(cb > 3000, "county_boundaries looks like the full US", f"({cb:,} rows)")
    cur.execute("""
        SELECT count(*) FROM county_boundaries
        WHERE geom IS NULL OR NOT ST_IsValid(geom)
    """)
    check(cur.fetchone()[0] == 0, "all county geometry is valid")
    cur.execute("SELECT count(*) FROM county_health_rankings")
    chr_rows = cur.fetchone()[0]
    check(chr_rows > 3000, "county_health_rankings loaded nationally", f"({chr_rows:,} rows)")
    cur.execute("""
        SELECT count(*) FROM county_health_rankings
        WHERE v001_raw_value IS NULL AND v011_raw_value IS NULL
          AND v009_raw_value IS NULL AND v021_raw_value IS NULL
          AND v043_raw_value IS NULL AND v057_raw_value IS NULL
    """)
    check(cur.fetchone()[0] == 0, "every CHR row carries at least one measure",
          "(all six NULL means the column mapping is wrong)")

    if args.skip_explain:
        print("\n(skipping EXPLAIN)")
        cur.close(); conn.close()
        return 1 if FAILURES else 0

    print("\n== query plans ==")
    plans = [
        ("/studies list (status + order)",
         """EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
            SELECT nct_number FROM processed_data
            WHERE status = 'RECRUITING'
            ORDER BY last_updated_date DESC NULLS LAST LIMIT 10""",
         "idx_pd_status_updated"),
        ("/studies bbox filter",
         """EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
            SELECT pd.nct_number FROM processed_data pd
            WHERE pd.nct_number IN (
                SELECT sl.study_nct_number FROM study_locations sl
                JOIN locations l ON l.id = sl.location_id
                WHERE l.geom && ST_MakeEnvelope(-122.6, 37.6, -122.2, 37.9, 4326))
            ORDER BY pd.last_updated_date DESC NULLS LAST LIMIT 10""",
         "idx_locations_geom"),
        ("/studies/map-data cluster",
         """EXPLAIN (ANALYZE, BUFFERS, COSTS OFF)
            SELECT count(*), ST_SnapToGrid(l.geom, 0.5, 0.5)
            FROM locations l
            JOIN study_locations sl ON sl.location_id = l.id
            WHERE l.geom && ST_MakeEnvelope(-125, 32, -114, 42, 4326)
            GROUP BY 2""",
         "idx_locations_geom"),
        ("text search (trigram)",
         "EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) "
         "SELECT nct_number FROM processed_data WHERE study_title ILIKE '%melanoma%' LIMIT 10",
         "idx_pd_title_trgm"),
    ]
    for label, sql, want_index in plans:
        print(f"\n   -- {label}")
        cur.execute(sql)
        plan = "\n".join(r[0] for r in cur.fetchall())
        for line in plan.splitlines():
            if any(k in line for k in ("Scan", "Sort", "Aggregate", "Execution Time", "Planning Time")):
                print(f"      {line.strip()}")
        if want_index in plan:
            print(f"      PASS  uses {want_index}")
        elif "Seq Scan on processed_data" in plan:
            print(f"      FAIL  seq scan on processed_data; expected {want_index}")
            FAILURES.append(f"{label}: seq scan")
        else:
            print(f"      NOTE  {want_index} not named in the plan (may still be fine)")

    if args.base_url:
        print("\n== live latency ==")
        for zoom, bbox in ((3, "-125,25,-66,49"), (14, "-122.5,37.7,-122.3,37.85")):
            url = (f"{args.base_url}/studies/map-data?bbox={bbox}&zoom={zoom}")
            samples, codes = [], {}
            for _ in range(60):
                t0 = time.perf_counter()
                try:
                    with urllib.request.urlopen(url, timeout=30) as r:
                        r.read()
                        codes[r.status] = codes.get(r.status, 0) + 1
                except Exception as exc:  # noqa: BLE001
                    codes[getattr(exc, "code", "err")] = codes.get(getattr(exc, "code", "err"), 0) + 1
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            p50 = statistics.median(samples)
            p95 = samples[int(len(samples) * 0.95) - 1]
            print(f"   zoom {zoom:>2}: p50 {p50:7.1f} ms   p95 {p95:7.1f} ms   {codes}")

    cur.close()
    conn.close()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:", file=sys.stderr)
        for f in FAILURES:
            print(f"   - {f}", file=sys.stderr)
        return 1
    print("all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
