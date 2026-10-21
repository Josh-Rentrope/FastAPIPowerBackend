#!/usr/bin/env python3
"""Parse the free-text `locations` column into structured, deduplicated rows.

This is the stage that never existed. Nothing in the original repository populated
`locations` or `study_locations`, and `locations.last_geocoded_at` shows a geocoder
did run -- it just was not committed.

## The input

`raw_data.locations` holds pipe-delimited free text. Across the full export that is
**3,546,381 entries**. A random sample of 8,000 breaks down as:

    2 parts   2.1%   City, Country
    3 parts  13.8%   Facility, City, Country       OR  City, State, Country
    4 parts  25.9%   Facility, City, Postal, Country OR  Facility, City, State, Country
    5 parts  51.8%   Facility, City, State, Postal, Country
    6+ parts  6.4%   as above, with a comma inside the facility name

The two ambiguities are real and cannot be resolved by shape alone. "Kettering,
Ohio, United States" is (city, state) while "Novartis Investigator Site,
Gummersbach, Germany" is (facility, city) -- both three fields. The parser resolves
them by asking whether the candidate field is a **real admin1 division for that
country**, which is why the gazetteers must be loaded first.

## Parsing from the right

Facility names contain commas ("University of California, Los Angeles, Los
Angeles, California, 90095, United States"), so the split is anchored at the
country end and works leftward. Country names themselves sometimes contain commas
("Korea, Republic of"); rather than pre-processing the text, the parser checks
whether the last TWO fields together name a known country and, if so, consumes both.

## Deduplication is at CITY level, not site level

`location_key` is a hash of (normalized city | admin1 | postal | country) and
deliberately excludes the facility. Keying on the full raw string would under-merge
-- the same city appears via thousands of different site names -- and would leave
~3.5M rows for what is really a few hundred thousand places. The consequence is
that ten sites in one city become one map point per study, which is the behaviour
the map wants. `facility` and `location_text` retain the first spelling seen.

Usage:
    uv run python etl/03_extract_locations.py --all
    uv run python etl/03_extract_locations.py --file-name subset5000.csv
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

import psycopg2
from psycopg2.extras import execute_values

# Must match lookups/01_load_gazetteers.normalize exactly, or the joins miss.
_PUNCT = str.maketrans({c: " " for c in ".,'`´’-‐‑‒–—()/\\"})

def normalize(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(_PUNCT).lower()
    return " ".join(s.split())


# ── postal-code detection ───────────────────────────────────────────────────
#
# The per-country patterns from GeoNames are necessary but not sufficient, and
# three failure modes showed up against real data:
#
#   * Some patterns REQUIRE a space -- Sweden's is `^(?:SE)?\d{3}\s\d{2}$` -- so
#     normalising "549 49" to "54949" before matching made it fail. Match the raw
#     value first, and only then try the stripped one.
#   * Some values carry a country prefix the pattern does not allow: Norway's
#     "N-3515" against `^(\d{4})$`, Hungary's "H-1027".
#   * Some values are truncated or use a local convention: Brazil's "74605"
#     against `^\d{5}-\d{3}$`, and US zips written as nine digits with no hyphen
#     ("432101228" for 43210-1228).
#
# So: try the country pattern, then the generic numeric shapes below. A field that
# is entirely digits is essentially never a city name, which is what makes the
# generic pass safe.
GENERIC_POSTAL = [
    re.compile(r"^\d{4,10}$"),                       # 74605, 432101228, 31062
    re.compile(r"^[A-Za-z]{0,2}[- ]?\d{3,6}$"),      # N-3515, H-1027, 1105
    re.compile(r"^\d{2,5}[- ]\d{2,4}$"),             # 40-040, 93-513, 01246-000
    re.compile(r"^\d{3,5}\s\d{2,4}$"),               # "549 49", "116 91"
    re.compile(r"^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$"),  # Canadian A1A 1A1
]


def is_postal(candidate: str, cc: str | None, postal_re: dict[str, re.Pattern]) -> bool:
    """Is this field a postal code rather than a city or a state?

    Order matters: the country's own pattern is authoritative when it matches, and
    the generic shapes only get a look when it does not.
    """
    c = candidate.strip()
    if not c:
        return False
    if any(ch.isdigit() for ch in c):
        if cc and cc in postal_re:
            for probe in (c, c.replace(" ", ""), c.replace("-", "").replace(" ", "")):
                try:
                    if postal_re[cc].fullmatch(probe):
                        return True
                except re.error:
                    break
        # A value with no letters at all, or a short letter-prefixed number, is a
        # postal code far more often than it is a place name.
        for rx in GENERIC_POSTAL:
            if rx.match(c):
                return True
    return False


# Region names arrive with a descriptive suffix far more often than the gazetteer
# spells them that way: "Porto District" vs GeoNames' "Porto", "Beijing
# Municipality" vs "Beijing", "Silesian Voivodeship" vs the same. Stripping the
# suffix from BOTH sides is what makes those meet.
_REGION_SUFFIX = re.compile(
    r"\b("
    r"district|province|voivodeship|region|regional|municipality|municipal|"
    r"canton|prefecture|oblast|krai|okrug|autonomous|republic|state|county|"
    r"governorate|governate|department|departmento|parish|arrondissement|"
    r"metropolitan|metro|division|territory|special|administrative"
    r")\b"
)


def core_region(name: str | None) -> str:
    """Fold a region name to its bare form so suffix differences stop mattering."""
    n = normalize(name)
    if not n:
        return ""
    out = _REGION_SUFFIX.sub(" ", n)
    out = " ".join(out.split())
    return out or n


# "Sharjah city", "Buenos Aires City". The gazetteer stores these without the
# suffix, and the export sometimes adds it, so it has to be stripped from the
# location's side too -- stripping only the gazetteer's side fixed nothing.
_CITY_SUFFIX = re.compile(r"\s+(city|ciudad|stadt|ville)$")


def core_city(name: str | None) -> str:
    n = normalize(name)
    if not n:
        return ""
    out = _CITY_SUFFIX.sub("", n).strip()
    return out if out and out != n else ""


# "Turkey (Türkiye)", "Bolivia (Plurinational State of)" -- the export sometimes
# carries both a common and an official name in one field.
PAREN = re.compile(r"\(([^)]*)\)")


class Parser:
    """Holds the gazetteer lookups needed to disambiguate a location string."""

    def __init__(self, conn, alias_conn=None) -> None:
        cur = conn.cursor()
        cur.execute("SELECT raw_country, iso2 FROM country_aliases")
        self.alias: dict[str, str] = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute("SELECT iso2, postal_regex FROM geo_country WHERE postal_regex IS NOT NULL")
        self.postal_re: dict[str, re.Pattern] = {}
        for iso2, rx in cur.fetchall():
            try:
                self.postal_re[iso2] = re.compile(rx)
            except re.error:
                pass

        # country -> set of normalized admin1 AND admin2 names. admin2 is included
        # because the export's middle field is sometimes a sub-region (UK "East
        # Midlands") that is not a first-level division; without it the parser
        # reads the region as the city and the geocode degrades a tier.
        # Maps rather than sets: a hit yields the gazetteer's CANONICAL spelling,
        # which is what gets stored in normalized_admin1. Storing the raw text
        # instead would leave tier 3 unable to find it whenever the parser matched
        # through the core form ("Porto District" vs the gazetteer's "Porto").
        cur.execute("SELECT country_code, normalized_name, normalized_name_local FROM geo_admin1")
        self.admin1: dict[str, dict[str, str]] = defaultdict(dict)
        self.admin1_core: dict[str, dict[str, str]] = defaultdict(dict)
        for cc, nn, nnl in cur.fetchall():
            canon = nn or nnl
            if not canon:
                continue
            for key in {nn, nnl}:
                if key:
                    self.admin1[cc].setdefault(key, canon)
                    core = core_region(key)
                    if core:
                        self.admin1_core[cc].setdefault(core, canon)
        cur.close()

        # Fuzzy misses are resolved on a SEPARATE autocommit connection. Doing it
        # on the main one would interleave statements with the open server-side
        # read cursor, and the resolutions are written back so they become
        # inspectable and correctable rather than living only in this process.
        self._fz = alias_conn or conn
        self._fz.autocommit = True
        self._miss: set[str] = set()
        self.resolved_new: dict[str, str] = {}

    def _candidates(self, s: str) -> list[str]:
        """Every reasonable reading of one country string, normalized."""
        outs = [s]
        m = PAREN.search(s)
        if m:
            outs.append(PAREN.sub(" ", s))   # "Turkey (Türkiye)" -> "Turkey"
            outs.append(m.group(1))          # ...or the parenthetical itself
        seen, res = set(), []
        for o in outs:
            n = normalize(o)
            if n and n not in seen:
                seen.add(n)
                res.append(n)
        return res

    def _fuzzy(self, cand: str) -> str | None:
        """Last resort: trigram similarity against the country name table.

        Handles variants nobody enumerated -- "Democratic Republic of the Congo",
        "Palestinian Territories" -- without growing a hand-maintained list.
        """
        if cand in self._miss:
            return None
        if len(cand) < 4:
            self._miss.add(cand)
            return None
        cur = self._fz.cursor()
        cur.execute("""
            SELECT iso2, similarity(normalized_name, %s) AS s
            FROM geo_country
            WHERE normalized_name %% %s
            ORDER BY s DESC
            LIMIT 1
        """, (cand, cand))
        row = cur.fetchone()
        cur.close()
        if row and row[1] is not None and row[1] >= 0.62:
            self.alias[cand] = row[0]
            self.resolved_new[cand] = row[0]
            return row[0]
        self._miss.add(cand)
        return None

    def _resolve(self, s: str) -> str | None:
        for cand in self._candidates(s):
            hit = self.alias.get(cand)
            if hit:
                return hit
            hit = self._fuzzy(cand)
            if hit:
                return hit
        return None

    def country(self, parts: list[str]) -> tuple[str | None, str | None, list[str]]:
        """Return (country_raw, iso2, remaining_parts)."""
        if not parts:
            return None, None, []
        last = parts[-1]
        hit = self._resolve(last)
        if hit is None and len(parts) >= 2:
            # "Korea, Republic of" / "Taiwan, Province of China" / "Congo, The
            # Democratic Republic of the" -- a country whose name contains commas,
            # so the split put half of it in the previous field.
            joined = f"{parts[-2]}, {last}"
            hit2 = self._resolve(joined)
            if hit2 is not None:
                return joined, hit2, parts[:-2]
        return last, hit, parts[:-1]

    def persist_new_aliases(self) -> int:
        if not self.resolved_new:
            return 0
        cur = self._fz.cursor()
        rows = list(self.resolved_new.items())
        execute_values(cur, """
            INSERT INTO country_aliases (raw_country, iso2, canonical_name)
            VALUES %s ON CONFLICT (raw_country) DO UPDATE SET iso2 = EXCLUDED.iso2
        """, [(k, v, k) for k, v in rows])
        cur.close()
        return len(rows)

    def parse(self, entry: str) -> dict:
        parts = [p.strip() for p in entry.split(",")]
        parts = [p for p in parts if p]           # drop empties from "a,,b"
        country_raw, cc, rest = self.country(parts)

        postal = None
        admin1 = None

        if rest and is_postal(rest[-1], cc, self.postal_re):
            postal = rest[-1].strip()
            rest = rest[:-1]

        # Only treat the next field as a state/province if it really is one for
        # this country. This is the disambiguation that makes 3- and 4-part
        # strings resolvable at all.
        normalized_admin1 = None
        if len(rest) >= 2 and cc:
            cand = normalize(rest[-1])
            core = core_region(rest[-1])
            canon = self.admin1.get(cc, {}).get(cand)
            if not canon and core:
                canon = self.admin1_core.get(cc, {}).get(core)
            if canon:
                admin1 = rest[-1].strip()
                normalized_admin1 = canon
                rest = rest[:-1]

        city = rest[-1].strip() if rest else None
        facility = ", ".join(rest[:-1]).strip() if len(rest) > 1 else None

        # "Moscow Russia, Russia" -- the city repeats the country. Left alone it
        # matches no populated place and the row falls to a country centroid.
        if city and country_raw:
            cn = normalize(city)
            rn = normalize(country_raw)
            if cn.endswith(" " + rn):
                city = city[: len(city) - len(country_raw)].strip(" ,-") or None
            elif cn == rn:
                city = None

        # A left-over bare number is a postal code the pattern missed, not a city.
        # Keeping it would send the lookup after a town called "462025250".
        if city and re.fullmatch(r"[\d\- ]{4,}", city):
            if not postal:
                postal = city.strip()
            city = None

        # 1-field strings ("Pfizer Investigational Site") carry no country and no
        # city. Keep them so the study still has a location row, but leave them to
        # the country tier -- there is nothing to geocode.
        key_src = "|".join([
            normalize(city), normalize(admin1), (postal or "").strip().lower(), cc or "",
        ])
        if not any((city, admin1, postal, cc)):
            key_src = "|raw|" + normalize(entry)
            if not normalize(entry):
                return {}

        return {
            "location_text": entry.strip()[:500],
            "facility": (facility or "")[:500] or None,
            "city": (city or "")[:200] or None,
            "normalized_city": normalize(city)[:200] or None,
            "normalized_city_core": core_city(city)[:200] or None,
            "admin1": (admin1 or "")[:200] or None,
            # The gazetteer's canonical spelling when the region was recognised,
            # otherwise the folded raw text, which simply will not match.
            "normalized_admin1": (normalized_admin1
                                  or normalize(admin1))[:200] or None,
            "postal_code": (postal or "")[:20] or None,
            "country_raw": (country_raw or "")[:120] or None,
            "country_code": cc,
            "location_key": hashlib.md5(key_src.encode("utf-8")).hexdigest(),
        }


STAGE_DDL = """
CREATE TEMP TABLE loc_stage (
    id              BIGSERIAL,
    study_nct_number TEXT,
    location_key    TEXT,
    location_text   TEXT,
    facility        TEXT,
    city            TEXT,
    normalized_city TEXT,
    normalized_city_core TEXT,
    admin1          TEXT,
    normalized_admin1 TEXT,
    postal_code     TEXT,
    country_raw     TEXT,
    country_code    TEXT
) ON COMMIT DROP;
"""

COLS = ["study_nct_number", "location_key", "location_text", "facility",
        "city", "normalized_city", "normalized_city_core", "admin1", "normalized_admin1",
        "postal_code", "country_raw", "country_code"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file-name", help="only this source_file_name")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, help="process at most N raw rows (debugging)")
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    args = p.parse_args()

    if not args.all and not args.file_name:
        raise SystemExit("pass --all or --file-name")

    conn = psycopg2.connect(
        dbname=args.dbname or os.environ.get("DB_NAME"),
        user=args.user or os.environ.get("DB_USER"),
        password=args.password or os.environ.get("DB_PASSWORD"),
        host=args.host, port=args.port, sslmode=args.sslmode,
    )
    cur = conn.cursor()

    t0 = time.time()
    print("-- loading gazetteer lookups into memory --")
    alias_conn = psycopg2.connect(
        dbname=args.dbname or os.environ.get("DB_NAME"),
        user=args.user or os.environ.get("DB_USER"),
        password=args.password or os.environ.get("DB_PASSWORD"),
        host=args.host, port=args.port, sslmode=args.sslmode,
    )
    parser = Parser(conn, alias_conn=alias_conn)
    print(f"   {len(parser.alias):,} country aliases, {len(parser.postal_re):,} postal patterns, "
          f"{len(parser.admin1):,} countries with admin divisions")

    cur.execute(STAGE_DDL)

    where = "" if args.all else "AND source_file_name = %(f)s"
    params = {} if args.all else {"f": args.file_name}
    limit = f"LIMIT {args.limit}" if args.limit else ""

    # A named (server-side) cursor: 607k rows of raw text is a lot to hold in
    # memory, and iterating server-side streams them instead.
    read_cur = conn.cursor(name="raw_locations")
    read_cur.itersize = 2000
    read_cur.execute(f"""
        SELECT nct_number, locations
        FROM raw_data
        WHERE locations IS NOT NULL AND locations <> '' {where} {limit}
    """, params)

    print("-- parsing --")
    batch: list[tuple] = []
    raw_rows = 0
    entries = 0
    unparsed = 0
    no_country = 0
    for nct, locs in read_cur:
        raw_rows += 1
        for entry in locs.split("|"):
            entry = entry.strip().rstrip(".").strip()
            if not entry:
                continue
            entries += 1
            parsed = parser.parse(entry)
            if not parsed:
                unparsed += 1
                continue
            if not parsed["country_code"]:
                no_country += 1
            batch.append((nct,) + tuple(parsed.get(c) for c in COLS[1:]))
        if len(batch) >= 20000:
            execute_values(cur, f"INSERT INTO loc_stage ({', '.join(COLS)}) VALUES %s",
                           batch, page_size=5000)
            batch.clear()
            if raw_rows % 100000 == 0:
                print(f"   {raw_rows:,} studies, {entries:,} entries, {time.time()-t0:.0f}s")
    if batch:
        execute_values(cur, f"INSERT INTO loc_stage ({', '.join(COLS)}) VALUES %s",
                       batch, page_size=5000)
    read_cur.close()

    print(f"   {raw_rows:,} studies -> {entries:,} entries "
          f"({unparsed:,} unusable, {no_country:,} with no country)")
    n_new = parser.persist_new_aliases()
    if n_new:
        print(f"   {n_new:,} country names resolved by similarity and cached in country_aliases")
    cur.execute("SELECT count(*), count(DISTINCT location_key) FROM loc_stage")
    staged, distinct = cur.fetchone()
    print(f"   staged {staged:,} rows, {distinct:,} distinct locations "
          f"({100*(1-distinct/max(staged,1)):.1f}% collapsed by city-level dedup)")

    print("-- inserting locations --")
    t1 = time.time()
    # DISTINCT ON with the staging id ordering keeps the first spelling seen.
    cur.execute("""
        INSERT INTO locations (location_key, location_text, facility, city, normalized_city,
                               normalized_city_core, admin1, normalized_admin1,
                               postal_code, country_raw, country_code)
        SELECT DISTINCT ON (location_key)
               location_key, location_text, facility, city, normalized_city,
               normalized_city_core, admin1, normalized_admin1,
               postal_code, country_raw, country_code
        FROM loc_stage
        ORDER BY location_key, id
        ON CONFLICT (location_key) DO NOTHING
    """)
    inserted = cur.rowcount
    print(f"   {inserted:,} new locations in {time.time()-t1:.1f}s")

    print("-- linking study_locations --")
    t1 = time.time()
    cur.execute("""
        INSERT INTO study_locations (study_nct_number, location_id)
        SELECT DISTINCT s.study_nct_number, l.id
        FROM loc_stage s
        JOIN locations l ON l.location_key = s.location_key
        JOIN processed_data pd ON pd.nct_number = s.study_nct_number
        ON CONFLICT DO NOTHING
    """)
    linked = cur.rowcount
    conn.commit()
    print(f"   {linked:,} study-location pairs in {time.time()-t1:.1f}s")

    # ANALYZE, not optional. Without fresh statistics the planner misestimates the
    # spatial semi-join and sequentially scans all 3.4M rows of study_locations
    # rather than probing the index -- measured at 310 ms against 150 ms for a
    # single bbox lookup. Autovacuum gets there eventually; doing it here means the
    # database is usable the moment the load finishes.
    print("-- ANALYZE --")
    t1 = time.time()
    conn.autocommit = True
    for table in ("locations", "study_locations", "processed_data"):
        cur.execute(f"ANALYZE {table}")
    print(f"   analyzed in {time.time()-t1:.1f}s")

    print("\n-- result --")
    cur.execute("SELECT count(*) FROM locations")
    print(f"   locations          {cur.fetchone()[0]:>10,}")
    cur.execute("SELECT count(*) FROM study_locations")
    print(f"   study_locations    {cur.fetchone()[0]:>10,}")
    cur.execute("SELECT count(DISTINCT study_nct_number) FROM study_locations")
    print(f"   studies with a site{cur.fetchone()[0]:>10,}")
    cur.execute("SELECT count(*) FROM locations WHERE country_code IS NULL")
    print(f"   unresolved country {cur.fetchone()[0]:>10,}")

    print("\n-- top countries --")
    cur.execute("""
        SELECT country_code, count(*) FROM locations
        GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT 15
    """)
    for cc, n in cur.fetchall():
        print(f"   {str(cc):<8} {n:>10,}")

    print("\n-- distinct country_raw strings that did not resolve (top 25) --")
    cur.execute("""
        SELECT country_raw, count(*) FROM locations
        WHERE country_code IS NULL AND country_raw IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT 25
    """)
    rows = cur.fetchall()
    if not rows:
        print("   (none)")
    for raw, n in rows:
        print(f"   {raw[:60]:<62} {n:>8,}")

    cur.close()
    conn.close()
    print(f"\ndone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
