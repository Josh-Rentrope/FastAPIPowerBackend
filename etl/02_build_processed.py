#!/usr/bin/env python3
"""Build `processed_data` from `raw_data`.

This is the transform that never existed. In the original repository nothing
populated `processed_data` at all -- only `PostGresTableInspector.py`, which just
prints a schema. The API reads exclusively from `processed_data`, so the pipeline
was not runnable from the committed scripts.

Everything below was checked against the actual v2 export rather than assumed,
and two assumptions that looked safe turned out to be wrong:

  * **Status does NOT need remapping.** The v2 export already emits the
    underscored uppercase form the API compares against (`ACTIVE_NOT_RECRUITING`,
    `NOT_YET_RECRUITING`, `ENROLLING_BY_INVITATION`, `NO_LONGER_AVAILABLE`). The
    worry was that it would say "Active, not recruiting" and every status filter
    would silently return zero rows. It does not. The only work is trimming.
    Verified with: SELECT study_status, count(*) FROM raw_data GROUP BY 1;

  * **`phases` uses the literal string `NA`** for "not applicable" (~39% of rows),
    which is not a phase and must become NULL rather than being passed through.

Dates arrive in two shapes -- `YYYY-MM-DD` and `YYYY-MM` -- and nothing else, so
the parser handles both (plus a bare year, defensively) and NULLs anything else
rather than guessing.

Usage:
    uv run python etl/02_build_processed.py --file-name subset5000.csv
    uv run python etl/02_build_processed.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg2

# A date parser used by the INSERT below. IMMUTABLE so it can be inlined.
DATE_FN = """
CREATE OR REPLACE FUNCTION ct_loose_date(txt TEXT) RETURNS DATE AS $$
    SELECT CASE
        WHEN txt IS NULL OR btrim(txt) = ''            THEN NULL
        WHEN btrim(txt) ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN to_date(btrim(txt), 'YYYY-MM-DD')
        WHEN btrim(txt) ~ '^[0-9]{4}-[0-9]{2}$'          THEN to_date(btrim(txt) || '-01', 'YYYY-MM-DD')
        WHEN btrim(txt) ~ '^[0-9]{4}$'                   THEN to_date(btrim(txt) || '-01-01', 'YYYY-MM-DD')
        ELSE NULL
    END
$$ LANGUAGE sql IMMUTABLE;
"""

# `enrollment` is a plain integer throughout this export, but ranges like "10-20"
# appear in other ClinicalTrials.gov shapes, and stripping non-digits would turn
# "10-20" into 1020. Take the first integer group only. The LEAST is guarded by a
# CASE because PostgreSQL's LEAST() ignores NULLs -- LEAST(NULL, x) is x, which
# would silently turn "unknown enrollment" into 2147483647.
ENROLLMENT = """
    CASE WHEN (regexp_match(enrollment, '[0-9]+'))[1] IS NULL THEN NULL
         ELSE LEAST((regexp_match(enrollment, '[0-9]+'))[1]::bigint, 2147483647)::integer
    END
"""

TRANSFORM = f"""
INSERT INTO processed_data (
    nct_number, study_title, study_url, acronym, status, conditions,
    interventions, sponsor, sex, age_group, phase, enrollment_count,
    study_type, start_date, completion_date, first_posted_date,
    last_updated_date, source_file_name
)
SELECT
    btrim(nct_number),
    nullif(btrim(study_title), ''),
    nullif(btrim(study_url), ''),
    nullif(btrim(acronym), ''),
    nullif(upper(btrim(study_status)), ''),
    nullif(btrim(conditions), ''),
    nullif(btrim(interventions), ''),
    nullif(btrim(sponsor), ''),
    nullif(upper(btrim(sex)), ''),
    nullif(btrim(age), ''),
    -- 'NA' is the export's "not applicable", not a phase.
    nullif(nullif(upper(btrim(phases)), 'NA'), ''),
    {ENROLLMENT},
    nullif(upper(btrim(study_type)), ''),
    ct_loose_date(start_date),
    ct_loose_date(completion_date),
    ct_loose_date(first_posted),
    ct_loose_date(last_update_posted),
    source_file_name
FROM raw_data
WHERE source_file_name = %(file_name)s
  AND nullif(btrim(nct_number), '') IS NOT NULL
ON CONFLICT (nct_number) DO UPDATE SET
    study_title       = EXCLUDED.study_title,
    study_url         = EXCLUDED.study_url,
    acronym           = EXCLUDED.acronym,
    status            = EXCLUDED.status,
    conditions        = EXCLUDED.conditions,
    interventions     = EXCLUDED.interventions,
    sponsor           = EXCLUDED.sponsor,
    sex               = EXCLUDED.sex,
    age_group         = EXCLUDED.age_group,
    phase             = EXCLUDED.phase,
    enrollment_count  = EXCLUDED.enrollment_count,
    study_type        = EXCLUDED.study_type,
    start_date        = EXCLUDED.start_date,
    completion_date   = EXCLUDED.completion_date,
    first_posted_date = EXCLUDED.first_posted_date,
    last_updated_date = EXCLUDED.last_updated_date,
    source_file_name  = EXCLUDED.source_file_name;
"""


def connect(args):
    dsn = {
        "dbname": args.dbname or os.environ.get("DB_NAME"),
        "user": args.user or os.environ.get("DB_USER"),
        "password": args.password or os.environ.get("DB_PASSWORD"),
        "host": args.host or os.environ.get("DB_HOST", "localhost"),
        "port": args.port or os.environ.get("DB_PORT", "5432"),
        "sslmode": args.sslmode,
    }
    missing = [k for k in ("dbname", "user", "password") if not dsn[k]]
    if missing:
        raise SystemExit(f"missing connection details: {', '.join(missing)}")
    return psycopg2.connect(**dsn)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file-name", help="source_file_name to transform (as recorded in raw_data)")
    p.add_argument("--all", action="store_true", help="transform every distinct source file")
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    args = p.parse_args()

    conn = connect(args)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(DATE_FN)

    if args.all:
        cur.execute("SELECT DISTINCT source_file_name FROM raw_data ORDER BY 1")
        files = [r[0] for r in cur.fetchall()]
    elif args.file_name:
        files = [args.file_name]
    else:
        raise SystemExit("pass --file-name or --all")

    if not files:
        print("nothing in raw_data to transform.")
        return 0

    for fname in files:
        print(f"\n-- {fname} --")
        # Scoped delete rather than TRUNCATE so that re-running one file does not
        # discard another. The FK to study_locations cascades, which is correct:
        # those pairs are meaningless once the study row is gone, and stage 03
        # rebuilds them.
        t0 = time.time()
        cur.execute("DELETE FROM processed_data WHERE source_file_name = %s", (fname,))
        deleted = cur.rowcount

        cur.execute(TRANSFORM, {"file_name": fname})
        written = cur.rowcount
        conn.commit()
        print(f"   removed {deleted:,} previous rows")
        print(f"   wrote   {written:,} rows in {time.time() - t0:.1f}s")

    print("\n-- result --")
    cur.execute("""
        SELECT
            count(*)                                            AS total,
            count(*) FILTER (WHERE status IS NULL)               AS no_status,
            count(*) FILTER (WHERE phase IS NULL)                AS no_phase,
            count(*) FILTER (WHERE enrollment_count IS NULL)     AS no_enrollment,
            count(*) FILTER (WHERE last_updated_date IS NULL)    AS no_last_updated,
            count(*) FILTER (WHERE sponsor IS NULL)              AS no_sponsor,
            count(*) FILTER (WHERE acronym IS NULL)              AS no_acronym
        FROM processed_data
    """)
    cols = ["total", "no_status", "no_phase", "no_enrollment", "no_last_updated", "no_sponsor", "no_acronym"]
    for k, v in zip(cols, cur.fetchone()):
        print(f"   {k:<18} {v:>10,}")

    print("\n-- status distribution (what the API filters on) --")
    cur.execute("SELECT status, count(*) FROM processed_data GROUP BY 1 ORDER BY 2 DESC")
    for status, n in cur.fetchall():
        print(f"   {str(status):<26} {n:>10,}")

    print("\n-- phase distribution --")
    cur.execute("SELECT phase, count(*) FROM processed_data GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT 12")
    for phase, n in cur.fetchall():
        print(f"   {str(phase):<26} {n:>10,}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
