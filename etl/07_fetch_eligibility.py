#!/usr/bin/env python3
"""Fetch eligibility data for every study and store the structured parts as-is.

## Why this is a separate fetch

The CSV export used for the main load has **no eligibility column** — the 30
columns stop at Study Documents. Eligibility only exists in the JSON API, under
`protocolSection.eligibilityModule`.

## The thing that makes this tractable

The API does not return eligibility as prose alone. It returns:

    sex                 ALL | FEMALE | MALE
    minimumAge          "18 Years"
    maximumAge          "75 Years"
    healthyVolunteers   true | false
    stdAges             ["ADULT", "OLDER_ADULT"]
    eligibilityCriteria free text

The first five are **already discrete values**, so the highest-yield screening
filters — age, sex, healthy volunteers — need no parsing and no model at all.
Only `eligibilityCriteria` has to be interpreted, and it is stored verbatim so
that interpretation can be redone without ever fetching again.

## Cost

`fields=NCTId,EligibilityModule` is ~210 KB per 100 studies. At pageSize 1000 that
is ~608 requests for the whole registry, so the run is bounded by the polite
rate limit rather than by bandwidth. The cursor is persisted, so an interrupted
run resumes instead of restarting.

Usage:
    uv run python etl/07_fetch_eligibility.py
    uv run python etl/07_fetch_eligibility.py --limit-pages 5      # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import psycopg2
from psycopg2.extras import execute_values

API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,EligibilityModule"
CURSOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "lookups", ".eligibility_cursor")
EXTRACTOR_VERSION = "eligibility-fetch/1"

_AGE = re.compile(r"(\d+)\s*(year|month|week|day)", re.I)


def parse_age(value: str | None) -> int | None:
    """'18 Years' -> 18. Interpreted in whole years, rounded down."""
    if not value:
        return None
    m = _AGE.search(value)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit.startswith("year"):
        return n
    if unit.startswith("month"):
        return n // 12
    if unit.startswith("week"):
        return n // 52
    return n // 365


def fetch_page(token: str | None, page_size: int, timeout: int = 60) -> dict:
    url = f"{API}?fields={FIELDS}&pageSize={page_size}"
    if token:
        url += f"&pageToken={urllib.parse.quote(token)}"
    req = urllib.request.Request(url, headers={"User-Agent": "clinical-trials-etl/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--delay", type=float, default=1.2,
                   help="seconds between requests; the API asks for <= 50/min")
    p.add_argument("--limit-pages", type=int, help="stop after N pages (smoke test)")
    p.add_argument("--restart", action="store_true",
                   help="ignore the saved cursor and start from the beginning")
    args = p.parse_args()

    token = None
    if not args.restart and os.path.exists(CURSOR_FILE):
        token = open(CURSOR_FILE, encoding="utf-8").read().strip() or None
        if token:
            print(f"resuming from saved cursor ({token[:20]}...)")

    conn = psycopg2.connect(
        dbname=args.dbname or os.environ.get("DB_NAME"),
        user=args.user or os.environ.get("DB_USER"),
        password=args.password or os.environ.get("DB_PASSWORD"),
        host=args.host, port=args.port, sslmode=args.sslmode,
    )
    conn.autocommit = False
    cur = conn.cursor()

    pages = 0
    written = 0
    t0 = time.time()

    while True:
        for attempt in range(5):
            try:
                payload = fetch_page(token, args.page_size)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                wait = 2 ** attempt
                print(f"  retry {attempt + 1} in {wait}s: {exc}", file=sys.stderr)
                time.sleep(wait)
        else:
            print("giving up after repeated failures; progress is saved", file=sys.stderr)
            break

        studies = payload.get("studies", [])
        rows = []
        for s in studies:
            ps = s.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            if not nct:
                continue
            em = ps.get("eligibilityModule", {}) or {}
            rows.append((
                nct,
                em.get("eligibilityCriteria"),
                (em.get("sex") or "").upper() or None,
                parse_age(em.get("minimumAge")),
                parse_age(em.get("maximumAge")),
                em.get("healthyVolunteers"),
                em.get("stdAges") or None,
                EXTRACTOR_VERSION,
            ))

        if rows:
            execute_values(cur, """
                INSERT INTO trial_eligibility
                    (nct_number, criteria_text, sex, min_age_years, max_age_years,
                     healthy_volunteers, std_ages, extractor_version)
                VALUES %s
                ON CONFLICT (nct_number) DO UPDATE SET
                    criteria_text      = EXCLUDED.criteria_text,
                    sex                = EXCLUDED.sex,
                    min_age_years      = EXCLUDED.min_age_years,
                    max_age_years      = EXCLUDED.max_age_years,
                    healthy_volunteers = EXCLUDED.healthy_volunteers,
                    std_ages           = EXCLUDED.std_ages,
                    fetched_at         = now(),
                    extractor_version  = EXCLUDED.extractor_version
            """, rows, page_size=500)
            conn.commit()
            written += len(rows)

        pages += 1
        token = payload.get("nextPageToken")
        os.makedirs(os.path.dirname(CURSOR_FILE), exist_ok=True)
        open(CURSOR_FILE, "w", encoding="utf-8").write(token or "")

        if pages % 10 == 0 or not token:
            rate = written / max(time.time() - t0, 1)
            print(f"  page {pages:>4}  written {written:>7,}  "
                  f"{rate:>5.0f}/s  elapsed {time.time()-t0:>5.0f}s")

        if not token:
            print("  reached the last page")
            break
        if args.limit_pages and pages >= args.limit_pages:
            print(f"  stopping after {pages} pages (--limit-pages)")
            break

        time.sleep(args.delay)

    print(f"\nfetched {written:,} studies across {pages} pages in {time.time()-t0:.0f}s")

    cur.execute("SELECT count(*), count(criteria_text), count(min_age_years), count(sex) "
                "FROM trial_eligibility")
    total, with_text, with_age, with_sex = cur.fetchone()
    print(f"  rows                {total:>9,}")
    print(f"  with criteria text  {with_text:>9,}  ({100*with_text/max(total,1):.1f}%)")
    print(f"  with a minimum age  {with_age:>9,}")
    print(f"  with a sex value    {with_sex:>9,}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
