#!/usr/bin/env python3
"""Create or update the schema for the clinical trials pipeline.

Rewritten from PostGresDatabaseSetup.py. The changes are not cosmetic:

* `processed_data.location_geom` is GONE, along with its GiST index. Nothing ever
  wrote it and nothing ever read it -- both map queries join through
  `study_locations` instead. A study averages ~6.5 locations, so a single point on
  the study row was an arbitrary representative that would have disagreed with the
  detail view. One source of truth: `locations.geom`.

* `study_locations` was rebuilt. It previously had NO primary key, NO index and a
  VARCHAR key, which at ~3.4M rows meant every join seq-scanned and every re-run of
  the load duplicated the whole table. It now has a composite PK (which also makes
  loading idempotent via ON CONFLICT DO NOTHING) and an index for the reverse
  direction.

* `processed_data` had no indexes at all, while every list request sorts 607k rows
  by `last_updated_date DESC NULLS LAST` and filters with `ILIKE '%...%'`. The
  index below matches the ORDER BY expression exactly -- Postgres will not use it
  otherwise -- and the trigram GIN indexes are the only thing that makes the text
  search not a guaranteed seq scan.

* Identity moved from an unauthenticated `X-User-Email` request header to
  server-issued sessions, so `users`/`participants` are replaced by `sessions`.

## Ordering matters

The script runs in three phases -- **tables, then migrations, then indexes** --
and the order is load-bearing. `CREATE TABLE IF NOT EXISTS` silently no-ops on a
table that already exists, so a column added in this revision is absent on an
existing database. If the indexes were created alongside their tables, an index
naming that new column would fail before the ALTER that adds it had a chance to
run, and the whole update would abort. Running the ALTERs in between is what makes
this script safe to point at a database built by the previous version.

Usage:
    uv run python db/00_setup_database.py            # create/update
    uv run python db/00_setup_database.py --drop     # destroy first (destructive)
    uv run python db/00_setup_database.py --clean-legacy   # also drop superseded objects
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg2

# ── connection ──────────────────────────────────────────────────────────────


def connect(args) -> "psycopg2.extensions.connection":
    """Connect from args, falling back to the PG* / DB_* environment.

    DigitalOcean requires TLS; `sslmode` defaults to `require` so that pointing
    this at the cluster cannot silently fall back to an unencrypted connection.
    """
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
        raise SystemExit(
            f"missing connection details: {', '.join(missing)}. "
            "Pass them as arguments or set DB_NAME/DB_USER/DB_PASSWORD."
        )
    conn = psycopg2.connect(**dsn)
    conn.autocommit = True
    return conn


# ── phase 1: tables ─────────────────────────────────────────────────────────

TABLES: list[tuple[str, str]] = [
    ("extensions", """
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
        CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- ILIKE '%...%' index support
    """),

    # Idempotency ledger. The old orchestrator asked
    # `SELECT DISTINCT source_file_name FROM raw_data`, which cannot tell a
    # completed load from a half-finished one. A content hash can.
    ("ingest_files", """
        CREATE TABLE IF NOT EXISTS ingest_files (
            file_sha256 TEXT PRIMARY KEY,
            file_name   TEXT NOT NULL,
            byte_size   BIGINT,
            row_count   INTEGER,
            stage       TEXT NOT NULL,
            loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),

    # Verbatim landing zone. This mirrors the columns we intend to keep, not all
    # 30 in the export: the three outcome-measure columns and Study Documents are
    # deliberately absent. One row's Secondary Outcome Measures field is 546 KB,
    # the application reads none of them, and they are the entire reason the
    # database would otherwise pass 10 GB.
    ("raw_data", """
        CREATE TABLE IF NOT EXISTS raw_data (
            id                      BIGSERIAL PRIMARY KEY,
            nct_number              TEXT,
            study_title             TEXT,
            study_url               TEXT,
            acronym                 TEXT,
            study_status            TEXT,
            brief_summary           TEXT,
            study_results           TEXT,
            conditions              TEXT,
            interventions           TEXT,
            sponsor                 TEXT,
            collaborators           TEXT,
            sex                     TEXT,
            age                     TEXT,
            phases                  TEXT,
            enrollment              TEXT,
            funder_type             TEXT,
            study_type              TEXT,
            study_design            TEXT,
            other_ids               TEXT,
            start_date              TEXT,
            primary_completion_date TEXT,
            completion_date         TEXT,
            first_posted            TEXT,
            results_first_posted    TEXT,
            last_update_posted      TEXT,
            locations               TEXT,
            ingested_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source_file_name        TEXT NOT NULL
        );
    """),

    ("processed_data", """
        CREATE TABLE IF NOT EXISTS processed_data (
            nct_number        TEXT PRIMARY KEY,
            study_title       TEXT,
            study_url         TEXT,
            acronym           TEXT,
            status            VARCHAR(50),
            conditions        TEXT,
            interventions     TEXT,
            sponsor           TEXT,
            sex               VARCHAR(20),
            age_group         VARCHAR(50),
            phase             VARCHAR(50),
            enrollment_count  INTEGER,
            study_type        VARCHAR(50),
            start_date        DATE,
            completion_date   DATE,
            first_posted_date DATE,
            last_updated_date DATE,
            source_file_name  TEXT
        );
    """),

    # Keyed on a hash of the normalised components, not on the raw text. The same
    # place recurs across thousands of studies and raw strings fragment on casing
    # and abbreviation, so they under-merge. `location_text` keeps the first-seen
    # spelling for display and audit.
    ("locations", """
        CREATE TABLE IF NOT EXISTS locations (
            id                   BIGSERIAL PRIMARY KEY,
            location_key         TEXT NOT NULL UNIQUE,
            location_text        TEXT NOT NULL,
            facility             TEXT,
            city                 TEXT,
            normalized_city      TEXT,
            -- normalized_city with a trailing "City"/"Ciudad" removed, matched as
            -- a lower-priority key by the geocoder.
            normalized_city_core TEXT,
            admin1               TEXT,
            normalized_admin1    TEXT,
            postal_code          TEXT,
            country_raw          TEXT,
            country_code         CHAR(2),
            geom                 GEOMETRY(Point, 4326),
            geocode_source       TEXT,
            geocode_precision    TEXT,
            geocode_confidence   SMALLINT,
            matched_name         TEXT,
            matched_gazetteer_id TEXT,
            last_geocoded_at     TIMESTAMPTZ
        );
    """),

    # The composite PK is what makes ON CONFLICT DO NOTHING possible, which is what
    # makes the location stage safely re-runnable.
    ("study_locations", """
        CREATE TABLE IF NOT EXISTS study_locations (
            study_nct_number TEXT   NOT NULL,
            location_id      BIGINT NOT NULL,
            PRIMARY KEY (study_nct_number, location_id),
            CONSTRAINT fk_sl_study FOREIGN KEY (study_nct_number)
                REFERENCES processed_data(nct_number) ON DELETE CASCADE,
            CONSTRAINT fk_sl_location FOREIGN KEY (location_id)
                REFERENCES locations(id) ON DELETE CASCADE
        );
    """),

    # Country names arrive as 262 distinct free-text variants, including
    # comma-bearing ones like "Korea, Republic of". Kept as a table rather than
    # code so the mapping can be corrected without a redeploy.
    ("country_aliases", """
        CREATE TABLE IF NOT EXISTS country_aliases (
            raw_country    TEXT PRIMARY KEY,
            iso2           CHAR(2) NOT NULL,
            canonical_name TEXT,
            centroid       GEOMETRY(Point, 4326)
        );
    """),

    # Per-country facts the parser needs. `postal_regex` comes from field 14 of
    # GeoNames countryInfo.txt, which is how postal codes get validated per country
    # instead of by a single guess-everything heuristic.
    ("geo_country", """
        CREATE TABLE IF NOT EXISTS geo_country (
            iso2            CHAR(2) PRIMARY KEY,
            iso3            CHAR(3),
            name            TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            postal_regex    TEXT,
            latitude        DOUBLE PRECISION,
            longitude       DOUBLE PRECISION
        );
    """),

    ("geo_cities", """
        CREATE TABLE IF NOT EXISTS geo_cities (
            geonameid       BIGINT PRIMARY KEY,
            name            TEXT NOT NULL,
            ascii_name      TEXT,
            normalized_name TEXT NOT NULL,
            -- GeoNames' `name` is the local spelling and `ascii_name` the
            -- transliteration: Roma/Rome, Munchen/Munich, Koln/Cologne. Indexing
            -- only the ASCII form means an export that says "Roma" never matches,
            -- which is exactly what happened before this column existed.
            normalized_name_local TEXT,
            country_code    CHAR(2),
            admin1_code     TEXT,
            latitude        DOUBLE PRECISION NOT NULL,
            longitude       DOUBLE PRECISION NOT NULL,
            population      BIGINT,
            feature_code    TEXT
        );
    """),

    ("geo_admin1", """
        CREATE TABLE IF NOT EXISTS geo_admin1 (
            country_code    CHAR(2) NOT NULL,
            admin1_code     TEXT    NOT NULL,
            name            TEXT    NOT NULL,
            normalized_name TEXT    NOT NULL,
            -- Same dual-name treatment as geo_cities: "Mazowieckie" and "Masovian
            -- Voivodeship" are the same division and the export uses both.
            normalized_name_local TEXT,
            latitude        DOUBLE PRECISION,
            longitude       DOUBLE PRECISION,
            PRIMARY KEY (country_code, admin1_code)
        );
    """),

    # US-only, from the Census ZCTA gazetteer. The highest-precision rung
    # available offline, and it hands us the county for free via ST_Contains.
    ("geo_postal", """
        CREATE TABLE IF NOT EXISTS geo_postal (
            postal_code  CHAR(5) NOT NULL,
            country_code CHAR(2) NOT NULL DEFAULT 'US',
            latitude     DOUBLE PRECISION NOT NULL,
            longitude    DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (country_code, postal_code)
        );
    """),

    ("county_boundaries", """
        CREATE TABLE IF NOT EXISTS county_boundaries (
            fipscode    TEXT PRIMARY KEY,
            county_name TEXT,
            state_fips  CHAR(2),
            state_name  TEXT,
            geom        GEOMETRY(MultiPolygon, 4326) NOT NULL
        );
    """),

    ("county_health_rankings", """
        CREATE TABLE IF NOT EXISTS county_health_rankings (
            id              SERIAL PRIMARY KEY,
            fipscode        TEXT UNIQUE NOT NULL,
            state_name      TEXT,
            county_name     TEXT,
            v001_raw_value  FLOAT,
            v011_raw_value  FLOAT,
            v009_raw_value  FLOAT,
            v057_raw_value  FLOAT,
            v043_raw_value  FLOAT,
            v021_raw_value  FLOAT,
            ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """),

    # ── chat, now behind a server-issued session ─────────────────────────────
    #
    # Only the SHA-256 of the token is stored. The raw value exists exactly once,
    # in the response that set the cookie, so a dump of this table does not yield
    # a usable credential.
    ("sessions", """
        CREATE TABLE IF NOT EXISTS sessions (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            token_hash   TEXT NOT NULL UNIQUE,
            display_name TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),

    ("conversations", """
        CREATE TABLE IF NOT EXISTS conversations (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),

    ("messages", """
        CREATE TABLE IF NOT EXISTS messages (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            sender_id       VARCHAR(64) NOT NULL,
            content         TEXT NOT NULL,
            sent_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),
]


# ── phase 2: migrations ─────────────────────────────────────────────────────
#
# Additive only, and they run BEFORE the indexes for the reason described in the
# module docstring. Postgres has had ADD COLUMN IF NOT EXISTS since 9.6.

MIGRATIONS: list[tuple[str, str]] = [
    ("locations.normalized_admin1",
     "ALTER TABLE locations ADD COLUMN IF NOT EXISTS normalized_admin1 TEXT"),
    ("locations.geocode_provenance", """
        ALTER TABLE locations ADD COLUMN IF NOT EXISTS geocode_source TEXT;
        ALTER TABLE locations ADD COLUMN IF NOT EXISTS geocode_precision TEXT;
        ALTER TABLE locations ADD COLUMN IF NOT EXISTS geocode_confidence SMALLINT;
        ALTER TABLE locations ADD COLUMN IF NOT EXISTS matched_name TEXT;
        ALTER TABLE locations ADD COLUMN IF NOT EXISTS matched_gazetteer_id TEXT;
    """),
    ("processed_data.acronym",
     "ALTER TABLE processed_data ADD COLUMN IF NOT EXISTS acronym TEXT"),
    ("raw_data.results_first_posted",
     "ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS results_first_posted TEXT"),
    ("raw_data.funder_and_design", """
        ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS funder_type TEXT;
        ALTER TABLE raw_data ADD COLUMN IF NOT EXISTS study_design TEXT;
    """),
    ("gazetteer.local_names", """
        ALTER TABLE geo_cities ADD COLUMN IF NOT EXISTS normalized_name_local TEXT;
        ALTER TABLE geo_admin1 ADD COLUMN IF NOT EXISTS normalized_name_local TEXT;
        ALTER TABLE locations ADD COLUMN IF NOT EXISTS normalized_city_core TEXT;
    """),
]


# ── phase 3: indexes ────────────────────────────────────────────────────────
#
# After the migrations, because some of these name migrated columns.

INDEXES: list[tuple[str, str]] = [
    ("raw_data", """
        CREATE INDEX IF NOT EXISTS idx_raw_data_source ON raw_data (source_file_name);
        CREATE INDEX IF NOT EXISTS idx_raw_data_nct ON raw_data (nct_number);
    """),

    ("processed_data", """
        -- Must match the ORDER BY in FastAPIPostgresServer.get_studies exactly,
        -- including NULLS LAST, or the planner will not use it.
        CREATE INDEX IF NOT EXISTS idx_pd_last_updated
            ON processed_data (last_updated_date DESC NULLS LAST);
        CREATE INDEX IF NOT EXISTS idx_pd_status_updated
            ON processed_data (status, last_updated_date DESC NULLS LAST);

        -- Without these, `study_title ILIKE '%melanoma%'` is a seq scan of 607k
        -- rows, in both /studies and the LLM tool call.
        CREATE INDEX IF NOT EXISTS idx_pd_title_trgm
            ON processed_data USING GIN (study_title gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_pd_cond_trgm
            ON processed_data USING GIN (conditions gin_trgm_ops);
    """),

    ("locations", """
        CREATE INDEX IF NOT EXISTS idx_locations_geom
            ON locations USING GIST (geom);
        CREATE INDEX IF NOT EXISTS idx_locations_norm
            ON locations (country_code, normalized_city, normalized_admin1, postal_code);
        -- Drives the geocoder's resumable worklist.
        CREATE INDEX IF NOT EXISTS idx_locations_worklist
            ON locations (id) WHERE geom IS NULL;
    """),

    ("study_locations", """
        CREATE INDEX IF NOT EXISTS idx_sl_location ON study_locations (location_id);

        -- Raised from the default 100. With default statistics the planner
        -- estimated the bbox semi-join badly enough to choose a parallel
        -- sequential scan of all 3.4M rows over an index scan that visits 379
        -- of them; measured at 310 ms versus 150 ms. This setting is what
        -- stopped that.
        ALTER TABLE study_locations ALTER COLUMN location_id SET STATISTICS 1000;
    """),

    ("gazetteers", """
        CREATE INDEX IF NOT EXISTS idx_geo_cities_match
            ON geo_cities (country_code, normalized_name);
        CREATE INDEX IF NOT EXISTS idx_geo_cities_match_admin1
            ON geo_cities (country_code, admin1_code, normalized_name);
        CREATE INDEX IF NOT EXISTS idx_geo_cities_local
            ON geo_cities (country_code, normalized_name_local);
        CREATE INDEX IF NOT EXISTS idx_geo_admin1_name
            ON geo_admin1 (country_code, normalized_name);
        CREATE INDEX IF NOT EXISTS idx_geo_admin1_local
            ON geo_admin1 (country_code, normalized_name_local);
        CREATE INDEX IF NOT EXISTS idx_geo_country_norm
            ON geo_country (normalized_name);
    """),

    ("counties", """
        CREATE INDEX IF NOT EXISTS idx_county_boundaries_geom
            ON county_boundaries USING GIST (geom);
    """),

    ("chat", """
        CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations (session_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages (conversation_id, sent_at DESC);
    """),
]


# Dropped only under --drop. Ordered children-first so FKs do not block.
DROP_ORDER = [
    "messages", "conversations", "sessions",
    "study_locations", "locations", "raw_data", "processed_data",
    "ingest_files", "county_boundaries", "county_health_rankings",
    "geo_cities", "geo_admin1", "geo_postal", "geo_country", "country_aliases",
]


def run_phase(cur, label: str, steps: list[tuple[str, str]]) -> bool:
    print(f"\n-- {label} --")
    ok = True
    for name, sql in steps:
        try:
            cur.execute(sql)
            print(f"   ok  {name}")
        except Exception as exc:  # noqa: BLE001 - report the object, keep going
            print(f"   FAILED {name}: {exc}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"),
                   help="libpq sslmode; 'require' is the default so a managed "
                        "cluster cannot be reached in the clear by accident")
    p.add_argument("--drop", action="store_true",
                   help="DROP the pipeline tables first. Destructive.")
    p.add_argument("--clean-legacy", action="store_true",
                   help="also drop the columns/tables this revision supersedes")
    args = p.parse_args()

    conn = connect(args)
    cur = conn.cursor()

    print(f"connected to {args.host}:{args.port}/{args.dbname}")

    # Report the environment; never assume it. This used to call
    # postgis_full_version() directly, which raises
    # "function postgis_full_version() does not exist" on a database where the
    # extension has not been enabled -- a confusing way to say "run the enable
    # step", and one that pointed at the script rather than at the database.
    cur.execute("SELECT version()")
    print(f"  {cur.fetchone()[0].split(',')[0]}")

    cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
    is_super = bool(cur.fetchone()[0])

    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'postgis'")
    row = cur.fetchone()
    if row:
        # postgis_full_version() returns a very long dependency dump; the version
        # inside the quotes is the useful part.
        cur.execute("SELECT postgis_full_version()")
        full = cur.fetchone()[0]
        version = full.split('"')[1] if '"' in full else full[:40]
        print(f"  PostGIS {version}")
    elif is_super:
        print("  postgis not installed yet; this role can create it, proceeding.")
    else:
        # CREATE EXTENSION is superuser-only, and it is per DATABASE -- a cluster
        # is created with defaultdb, so enabling it there leaves this database
        # untouched. Say exactly that rather than failing on a geometry column.
        print("  postgis NOT INSTALLED in this database.", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Run this first, as the cluster's admin role (doadmin):", file=sys.stderr)
        print("      uv run python db/00_enable_postgis.py \\", file=sys.stderr)
        print("          --user doadmin --password <admin-password>", file=sys.stderr)
        print("", file=sys.stderr)
        print("  CREATE EXTENSION applies per database, so enabling PostGIS in", file=sys.stderr)
        print("  defaultdb does not make it available here.", file=sys.stderr)
        cur.close()
        conn.close()
        return 2

    if args.drop:
        print("\n-- dropping (destructive) --")
        for t in DROP_ORDER:
            cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            print(f"   dropped {t}")

    ok = run_phase(cur, "phase 1: tables", TABLES)
    ok &= run_phase(cur, "phase 2: migrations", MIGRATIONS)
    ok &= run_phase(cur, "phase 3: indexes", INDEXES)

    if args.clean_legacy:
        print("\n-- removing superseded objects --")
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'processed_data' AND column_name = 'location_geom'
        """)
        if cur.fetchone():
            cur.execute("ALTER TABLE processed_data DROP COLUMN location_geom CASCADE")
            print("   dropped processed_data.location_geom (and its index)")
        else:
            print("   processed_data.location_geom already absent")
        for t in ("participants", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            print(f"   dropped {t} (if present)")

    print("\n-- summary --")
    cur.execute("""
        SELECT table_name, (SELECT count(*) FROM information_schema.columns c
                            WHERE c.table_name = t.table_name) AS cols
        FROM information_schema.tables t
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    for name, cols in cur.fetchall():
        print(f"   {name:<26} {cols:>3} columns")

    cur.close()
    conn.close()
    if not ok:
        print("\ncompleted with errors.", file=sys.stderr)
        return 1
    print("\nschema ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
