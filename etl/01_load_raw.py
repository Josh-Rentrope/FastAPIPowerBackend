#!/usr/bin/env python3
"""Load a ClinicalTrials.gov API v2 CSV export into `raw_data`.

Rewritten from CSVDataIngestion.py, which could not run against the current
export. That version built a 23-column temp table matching the *old* format and
then ran `COPY tmp FROM STDIN`; the v2 export has 30 columns, so the copy fails
with "extra data after last expected column".

The staging table below therefore mirrors **all 30 columns in file order**, and
the choice of what to persist is made explicitly in the INSERT that follows it.
That split matters: it means the COPY is positional and needs no parsing, and
dropping a column later is a one-line edit rather than a re-encoding of the file.

What is deliberately NOT persisted: the three outcome-measure columns and
`study_documents`. The application reads none of them, and one row's Secondary
Outcome Measures field is 546 KB -- together they are the whole reason this
database would otherwise exceed 10 GB instead of ~4 GB.

Idempotency is by content hash in `ingest_files`, not by filename. The original
orchestrator used `SELECT DISTINCT source_file_name FROM raw_data`, which cannot
distinguish a finished load from one that died halfway.

Usage:
    uv run python etl/01_load_raw.py --file "C:/Users/.../ctg-studies.csv"
    uv run python etl/01_load_raw.py --file ... --force
    uv run python etl/01_load_raw.py --make-subset 5000 --out small.csv   # dev slice
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time

import psycopg2

# ── the export, in file order ───────────────────────────────────────────────
#
# Positional. If ClinicalTrials.gov adds a column this list must be updated in
# lockstep with the CREATE TEMP TABLE below, which is why they sit adjacent.

CSV_COLUMNS = [
    ("NCT Number", "nct_number"),
    ("Study Title", "study_title"),
    ("Study URL", "study_url"),
    ("Acronym", "acronym"),
    ("Study Status", "study_status"),
    ("Brief Summary", "brief_summary"),
    ("Study Results", "study_results"),
    ("Conditions", "conditions"),
    ("Interventions", "interventions"),
    ("Primary Outcome Measures", "primary_outcome_measures"),
    ("Secondary Outcome Measures", "secondary_outcome_measures"),
    ("Other Outcome Measures", "other_outcome_measures"),
    ("Sponsor", "sponsor"),
    ("Collaborators", "collaborators"),
    ("Sex", "sex"),
    ("Age", "age"),
    ("Phases", "phases"),
    ("Enrollment", "enrollment"),
    ("Funder Type", "funder_type"),
    ("Study Type", "study_type"),
    ("Study Design", "study_design"),
    ("Other IDs", "other_ids"),
    ("Start Date", "start_date"),
    ("Primary Completion Date", "primary_completion_date"),
    ("Completion Date", "completion_date"),
    ("First Posted", "first_posted"),
    ("Results First Posted", "results_first_posted"),
    ("Last Update Posted", "last_update_posted"),
    ("Locations", "locations"),
    ("Study Documents", "study_documents"),
]

# Columns copied from staging into `raw_data`. Everything else is dropped.
KEEP = [
    "nct_number", "study_title", "study_url", "acronym", "study_status",
    "brief_summary", "study_results", "conditions", "interventions",
    "sponsor", "collaborators", "sex", "age", "phases", "enrollment",
    "funder_type", "study_type", "study_design", "other_ids",
    "start_date", "primary_completion_date", "completion_date",
    "first_posted", "results_first_posted", "last_update_posted", "locations",
]


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


def sha256_of(path: str, label: str = "") -> tuple[str, int]:
    """Hash the file, reporting progress: this is ~2 GB and takes a moment."""
    h = hashlib.sha256()
    size = 0
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
            if label and size % (256 * 1024 * 1024) < 8 * 1024 * 1024:
                pct = 100 * size / os.path.getsize(path)
                print(f"\r   hashing {label} {pct:5.1f}%", end="", flush=True)
    if label:
        print(f"\r   hashed {label} in {time.time() - t0:.1f}s" + " " * 20)
    return h.hexdigest(), size


def make_subset(src: str, n: int, out: str) -> int:
    """Write the first `n` records to a clean CSV.

    Uses the csv module rather than `head`, because quoted fields in this export
    contain embedded newlines -- a line-based slice would cut records in half and
    produce a file that loads without error but with corrupted rows.
    """
    written = 0
    with open(src, encoding="utf-8", newline="") as fin, \
         open(out, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(next(reader))
        for row in reader:
            writer.writerow(row)
            written += 1
            if written >= n:
                break
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="path to the ClinicalTrials.gov v2 CSV export")
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    p.add_argument("--force", action="store_true",
                   help="re-ingest even if this file's hash is already in ingest_files")
    p.add_argument("--make-subset", type=int, metavar="N",
                   help="write the first N records to --out and exit (dev slicing)")
    p.add_argument("--out", help="destination for --make-subset")
    args = p.parse_args()

    if args.make_subset:
        if not args.file or not args.out:
            raise SystemExit("--make-subset needs both --file and --out")
        n = make_subset(args.file, args.make_subset, args.out)
        print(f"wrote {n} records to {args.out}")
        return 0

    if not args.file:
        raise SystemExit("--file is required")
    if not os.path.exists(args.file):
        raise SystemExit(f"no such file: {args.file}")

    size_mb = os.path.getsize(args.file) / (1024 * 1024)
    print(f"source: {args.file}  ({size_mb:,.1f} MB)")

    # Verify the header before hashing 2 GB -- a mismatched export should fail in
    # the first second, not four minutes in.
    with open(args.file, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    expected = [c for c, _ in CSV_COLUMNS]
    if header != expected:
        print("header does not match the expected v2 export.", file=sys.stderr)
        print(f"  expected {len(expected)} columns, found {len(header)}", file=sys.stderr)
        for i, (got, want) in enumerate(zip(header, expected)):
            if got != want:
                print(f"  col {i}: expected {want!r}, found {got!r}", file=sys.stderr)
        if len(header) != len(expected):
            print(f"  extra/missing: {set(header) ^ set(expected)}", file=sys.stderr)
        return 2
    print(f"header verified: {len(header)} columns as expected")

    digest, byte_size = sha256_of(args.file, label=os.path.basename(args.file))
    print(f"sha256: {digest}")

    conn = connect(args)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM ingest_files WHERE file_sha256 = %s AND stage = 'raw'", (digest,))
    if cur.fetchone() and not args.force:
        print("this exact file is already loaded (stage='raw'). Use --force to reload.")
        cur.close(); conn.close()
        return 0

    if args.force:
        cur.execute("DELETE FROM raw_data WHERE source_file_name = %s", (os.path.basename(args.file),))

    # Staging mirrors the file exactly. TEMP + ON COMMIT DROP means the 2 GB of
    # text never reaches WAL and is gone the moment the transaction ends.
    cols_ddl = ",\n            ".join(f"{name} TEXT" for _, name in CSV_COLUMNS)
    try:
        print("\n-- staging table --")
        cur.execute(f"""
            CREATE TEMP TABLE raw_data_stg (
            {cols_ddl}
            ) ON COMMIT DROP
        """)

        print("-- COPY from stdin --")
        t0 = time.time()
        with open(args.file, encoding="utf-8", newline="") as f:
            # HEADER true ignores the first line; the positional order is already
            # verified above. 1 MB read buffer.
            cur.copy_expert(
                "COPY raw_data_stg FROM STDIN WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')",
                f, size=1024 * 1024,
            )
        print(f"   copied in {time.time() - t0:.1f}s")

        cur.execute("SELECT count(*) FROM raw_data_stg")
        staged = cur.fetchone()[0]
        print(f"   staged rows: {staged:,}")

        print("-- inserting into raw_data --")
        t0 = time.time()
        collist = ", ".join(KEEP)
        cur.execute(f"""
            INSERT INTO raw_data ({collist}, source_file_name)
            SELECT {collist}, %s FROM raw_data_stg
        """, (os.path.basename(args.file),))
        inserted = cur.rowcount
        print(f"   inserted {inserted:,} rows in {time.time() - t0:.1f}s")

        cur.execute("""
            INSERT INTO ingest_files (file_sha256, file_name, byte_size, row_count, stage)
            VALUES (%s, %s, %s, %s, 'raw')
            ON CONFLICT (file_sha256) DO UPDATE
                SET row_count = EXCLUDED.row_count, loaded_at = now()
        """, (digest, os.path.basename(args.file), byte_size, inserted))

        conn.commit()
        print("committed.")

    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        print(f"\nFAILED, rolled back: {exc}", file=sys.stderr)
        cur.close(); conn.close()
        return 1

    print("\n-- post-load --")
    cur.execute("SELECT count(*), count(DISTINCT nct_number) FROM raw_data")
    total, distinct = cur.fetchone()
    print(f"   raw_data rows          : {total:,}")
    print(f"   distinct NCT numbers   : {distinct:,}")
    cur.execute("SELECT count(*) FROM raw_data WHERE locations IS NOT NULL AND locations <> ''")
    print(f"   rows carrying locations: {cur.fetchone()[0]:,}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
