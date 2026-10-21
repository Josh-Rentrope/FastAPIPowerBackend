#!/usr/bin/env python3
"""Enable the extensions the pipeline needs, and grant the app role access.

This has to run as a **superuser** -- on a DigitalOcean managed cluster that
means `doadmin`, not the application role. `CREATE EXTENSION` is superuser-only,
which is precisely why the application role should not be `doadmin` for
everything else.

## Why this is a separate step

`CREATE EXTENSION` is **per database**. A cluster is created with `defaultdb`,
and running this against that database leaves `clinical_trials` untouched. The
symptom is a later `function postgis_full_version() does not exist`, which reads
like a broken cluster and is really "you enabled it in the wrong database".

## Why the grants are here too

PostgreSQL 15 changed the `public` schema: it is now owned by
`pg_database_owner` and no longer grants `CREATE` to `PUBLIC`. On a managed
cluster the owner is `doadmin`, so an application role cannot create tables even
though it can connect. The fix is an explicit grant, and doing it here rather
than discovering it mid-schema means the next stage simply works.

Usage:
    uv run python db/00_enable_postgis.py \
        --user doadmin --password "$env:DB_ADMIN_PASSWORD" \
        --grant-to trials_app
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg2
from psycopg2 import sql

REQUIRED = ("postgis", "pgcrypto", "pg_trgm")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_ADMIN_USER", "doadmin"),
                   help="a superuser on the cluster; doadmin on DigitalOcean")
    p.add_argument("--password", default=os.environ.get("DB_ADMIN_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    p.add_argument("--grant-to", default=os.environ.get("DB_USER"),
                   help="application role to grant schema access to")
    args = p.parse_args()

    if not all((args.dbname, args.user, args.password)):
        raise SystemExit("need --dbname, --user and --password (or DB_* env vars)")

    dsn = dict(dbname=args.dbname, user=args.user, password=args.password,
               host=args.host, port=args.port, sslmode=args.sslmode)
    try:
        conn = psycopg2.connect(**dsn)
    except psycopg2.OperationalError as exc:
        print(f"could not connect as {args.user}: {exc}", file=sys.stderr)
        return 1
    conn.autocommit = True
    cur = conn.cursor()

    print(f"connected as {args.user} to {args.host}:{args.port}/{args.dbname}")

    # `rolsuper` is the WRONG test on a managed cluster, and getting it wrong
    # rejects the only role that works. DigitalOcean's doadmin is
    # *superuser-like*: it can create trusted extensions such as PostGIS, but
    # pg_roles.rolsuper is still false for it. So this reports the flag and then
    # attempts the work, rather than refusing on a flag that does not mean here
    # what it means on a self-hosted server.
    cur.execute("SELECT rolsuper, rolname FROM pg_roles WHERE rolname = current_user")
    is_super, who = cur.fetchone()

    for ext in REQUIRED:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = %s", (ext,))
        row = cur.fetchone()
        if row:
            print(f"  {ext:<10} already installed ({row[0]})")
            continue
        try:
            cur.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(
                sql.Identifier(ext)))
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = %s", (ext,))
            print(f"  {ext:<10} installed ({cur.fetchone()[0]})")
        except psycopg2.errors.InsufficientPrivilege as exc:
            print(f"  {ext:<10} permission denied for {who}: {exc}".rstrip(),
                  file=sys.stderr)
            if ext == "postgis":
                print("", file=sys.stderr)
                print("  DigitalOcean requires a temporary superuser grant for some", file=sys.stderr)
                print("  extensions. In the console: Databases -> this cluster ->", file=sys.stderr)
                print("  Users -> doadmin -> Temporary superuser, then re-run this.", file=sys.stderr)
                return 1
        except psycopg2.Error as exc:
            # pg_trgm before 13, or a cluster that genuinely lacks the package.
            print(f"  {ext:<10} FAILED: {exc}", file=sys.stderr)
            if ext == "postgis":
                return 1

    if args.grant_to:
        print(f"\ngranting schema access to {args.grant_to}")
        try:
            cur.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(
                sql.Identifier(args.grant_to)))
            cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {}").format(
                sql.Identifier(args.grant_to)))
            cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {}").format(
                sql.Identifier(args.grant_to)))
            # So objects created later by doadmin (or by this role) are usable
            # without another grant.
            cur.execute(sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {}").format(
                sql.Identifier(args.grant_to)))
            cur.execute(sql.SQL(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {}").format(
                sql.Identifier(args.grant_to)))
            print(f"  granted USAGE, CREATE on schema public and all existing objects")
        except psycopg2.Error as exc:
            print(f"  grant failed: {exc}", file=sys.stderr)
            return 1

    cur.execute("SELECT postgis_full_version()")
    version = cur.fetchone()[0]
    print(f"\nPostGIS ready: {version.split(' ')[0]} {version.split(' ')[1]}")
    print("This was applied to the database named above -- CREATE EXTENSION is "
          "per-database, so a cluster's defaultdb does not inherit it.")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
