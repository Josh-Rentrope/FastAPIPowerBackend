#!/usr/bin/env python3
"""Load the downloaded reference files into the geocoding tables.

Loads: GeoNames cities500 / admin1 / admin2 / countryInfo, and the Census ZCTA
gazetteer. Country name variants are seeded from pycountry plus a manual table,
because the source strings are not country names so much as whatever the sponsor
typed -- "Korea, Republic of", "Turkey (Türkiye)", "Viet Nam".

Normalisation is applied on the way in, not at match time: both `geo_cities` and
`locations` end up holding the same folded form, so matching is a plain equality
join and needs no function index.

GeoNames data is CC BY 4.0 and attribution is required if this is published.
See the README.

Usage:
    uv run python lookups/01_load_gazetteers.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata

import psycopg2
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))

# Fold a place name to a comparable key.
#
# NFKD then dropping combining marks turns "São Paulo" into "Sao Paulo" and
# "Besançon" into "Besancon" -- necessary because the export is inconsistently
# accented (both "São Paulo" and "Sao Paulo" appear). Apostrophes and periods are
# removed because "St. Louis" and "St Louis" are the same place.
_PUNCT = str.maketrans({c: " " for c in ".,'`´’-‐‑‒–—()/\\"})


def normalize(name: str | None) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(_PUNCT).lower()
    return " ".join(s.split())


# Country strings that appear in the export and are NOT covered by pycountry's
# name list. The comma-bearing ones matter most: splitting on "," without
# handling them first turns "Korea, Republic of" into country="Republic of".
MANUAL_COUNTRY_ALIASES: dict[str, str] = {
    # comma-bearing official names
    "korea, republic of": "KR",
    "korea, democratic people's republic of": "KP",
    "taiwan, province of china": "TW",
    "iran, islamic republic of": "IR",
    "bolivia, plurinational state of": "BO",
    "venezuela, bolivarian republic of": "VE",
    "tanzania, united republic of": "TZ",
    "moldova, republic of": "MD",
    "macedonia, the former yugoslav republic of": "MK",
    "congo, the democratic republic of the": "CD",
    "virgin islands, british": "VG",
    "virgin islands, u.s.": "VI",
    "palestine, state of": "PS",
    "syrian arab republic": "SY",
    "lao people's democratic republic": "LA",
    "brunei darussalam": "BN",
    "russian federation": "RU",
    "viet nam": "VN",
    "hong kong": "HK",
    "macao": "MO",
    "macau": "MO",
    "kosovo": "XK",
    "cote d'ivoire": "CI",
    "côte d'ivoire": "CI",
    "czech republic": "CZ",
    "united states of america": "US",
    "usa": "US",
    "u.s.a.": "US",
    "uk": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "south korea": "KR",
    "north korea": "KP",
    "turkey": "TR",
    "turkiye": "TR",
    "east timor": "TL",
    "cape verde": "CV",
    "swaziland": "SZ",
    "burma": "MM",
    "ivory coast": "CI",
    "vatican city": "VA",
    "reunion": "RE",
    "curacao": "CW",
    "sint maarten": "SX",
    "the gambia": "GM",
    "the bahamas": "BS",
    "gambia": "GM",
    "bahamas": "BS",
    # pycountry in this build exposes only `name` and `official_name` -- no
    # `common_name` -- so the everyday English names are not derivable from it and
    # have to be listed. These are the ones the export actually uses.
    "russia": "RU",
    "south korea": "KR",
    "korea south": "KR",
    "korea north": "KP",
    "republic of korea": "KR",
    "vietnam": "VN",
    "taiwan": "TW",
    "iran": "IR",
    "syria": "SY",
    "laos": "LA",
    "bolivia": "BO",
    "venezuela": "VE",
    "tanzania": "TZ",
    "moldova": "MD",
    "macedonia": "MK",
    "north macedonia": "MK",
    "bosnia and herzegovina": "BA",
    "bosnia": "BA",
    "brunei": "BN",
    "czechia": "CZ",
    "uae": "AE",
    "united arab emirates": "AE",
    "democratic republic of the congo": "CD",
    "democratic republic of congo": "CD",
    "drc": "CD",
    "republic of the congo": "CG",
    "republic of congo": "CG",
    "congo": "CG",
    "palestinian territories": "PS",
    "palestine": "PS",
    "west bank": "PS",
    "gaza": "PS",
    "the former yugoslav republic of macedonia": "MK",
    "ivory coast": "CI",
    "cote divoire": "CI",
    "cape verde": "CV",
    "east timor": "TL",
    "burma": "MM",
    "myanmar": "MM",
    "south sudan": "SS",
    "central african republic": "CF",
    "dominican republic": "DO",
    "costa rica": "CR",
    "el salvador": "SV",
    "sri lanka": "LK",
    "new zealand": "NZ",
    "papua new guinea": "PG",
    "trinidad and tobago": "TT",
    "sierra leone": "SL",
    "burkina faso": "BF",
    "guinea bissau": "GW",
    "equatorial guinea": "GQ",
    "solomon islands": "SB",
    "marshall islands": "MH",
    "micronesia": "FM",
    "saint lucia": "LC",
    "st lucia": "LC",
    "united states": "US",
    "saudi arabia": "SA",
    "south africa": "ZA",
    "united kingdom": "GB",
}


def build_country_lookup() -> dict[str, str]:
    """raw normalized country string -> ISO2."""
    m = dict(MANUAL_COUNTRY_ALIASES)
    try:
        import pycountry
    except ImportError:
        print("   pycountry unavailable; using the manual alias table only", file=sys.stderr)
        return m
    for c in pycountry.countries:
        for attr in ("name", "official_name", "common_name"):
            v = getattr(c, attr, None)
            if v:
                m.setdefault(normalize(v), c.alpha_2)
        m.setdefault(normalize(c.alpha_2), c.alpha_2)
        m.setdefault(normalize(getattr(c, "alpha_3", "")), c.alpha_2)
    return m


def load_countries(cur, countries_path: str) -> int:
    """countryInfo.txt: ISO/name/postal-regex, plus centroids from cities500 later."""
    iso_names: dict[str, str] = {}
    iso3s: dict[str, str] = {}
    iso_postal: dict[str, str] = {}
    with open(countries_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 16:
                continue
            # Field indices per the GeoNames countryInfo spec:
            #   4 = Country name, 13 = Postal Code Format, 14 = Postal Code Regex,
            #   15 = Languages. Reading 15 here silently yields every language code
            #   as a "regex", which then matches nothing.
            iso2, iso3, name, postal_regex = p[0], p[1], p[4], p[14].strip()
            iso_names[iso2] = name
            iso3s[iso2] = iso3
            # GeoNames writes "#####" style placeholders in some rows; only keep
            # things that look like an actual regex.
            if postal_regex.startswith("^") or postal_regex.endswith("$"):
                iso_postal[iso2] = postal_regex
            else:
                iso_postal[iso2] = None  # type: ignore[assignment]

    rows = [
        (iso2, iso3s.get(iso2), name, normalize(name), iso_postal.get(iso2))
        for iso2, name in iso_names.items()
    ]
    execute_values(cur, """
        INSERT INTO geo_country (iso2, iso3, name, normalized_name, postal_regex)
        VALUES %s
        ON CONFLICT (iso2) DO UPDATE SET
            iso3 = EXCLUDED.iso3, name = EXCLUDED.name,
            normalized_name = EXCLUDED.normalized_name,
            postal_regex = EXCLUDED.postal_regex
    """, rows)
    return len(rows)


CITIES_INSERT = """
    INSERT INTO geo_cities (geonameid, name, ascii_name, normalized_name,
                            normalized_name_local, country_code, admin1_code,
                            latitude, longitude, population, feature_code)
    VALUES %s ON CONFLICT (geonameid) DO NOTHING
"""


def load_cities(cur, cities_path: str, batch: int = 20000) -> int:
    """cities500.txt. Alternatenames (field 4) are deliberately not loaded: it is
    the largest field by far and would multiply the table size for a matching
    benefit we can add later if coverage turns out to be poor."""
    total = 0
    buf = []
    with open(cities_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 15:
                continue
            try:
                geonameid = int(p[0])
                lat, lon = float(p[4]), float(p[5])
            except ValueError:
                continue
            name, ascii_name = p[1], p[2]
            cc = p[8] or None
            admin1 = p[10] or None
            feature_code = p[7] or None
            try:
                pop = int(p[14]) if p[14] else None
            except ValueError:
                pop = None
            # Both spellings are kept: GeoNames' `name` is local (Roma, Munchen,
            # Koln) and `ascii_name` is the transliteration (Rome, Munich,
            # Cologne). The export uses whichever it likes, so matching on one
            # form alone loses every city whose English name differs.
            buf.append((geonameid, name, ascii_name, normalize(ascii_name or name),
                        normalize(name), cc, admin1, lat, lon, pop, feature_code))
            if len(buf) >= batch:
                execute_values(cur, CITIES_INSERT, buf)
                total += len(buf)
                buf.clear()
    if buf:
        execute_values(cur, CITIES_INSERT, buf)
        total += len(buf)
    return total


def load_admin1(cur, path: str) -> int:
    """admin1CodesASCII.txt: 'US.CA\\tCalifornia\\tCalifornia\\t5332921'."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or "." not in p[0]:
                continue
            cc, code = p[0].split(".", 1)
            name, ascii_name = p[1], p[2]
            # normalized_name is the ASCII/transliterated form, and
            # normalized_name_local the local one. Both are matched, so
            # "Mazowieckie" and "Masovian Voivodeship" both land here.
            rows.append((cc, code, name, normalize(ascii_name or name), normalize(name)))
    execute_values(cur, """
        INSERT INTO geo_admin1 (country_code, admin1_code, name, normalized_name, normalized_name_local)
        VALUES %s ON CONFLICT (country_code, admin1_code) DO NOTHING
    """, rows)
    return len(rows)


def load_admin2(cur, path: str) -> int:
    """admin2Codes.txt: 'AE.01.101\\tAbu Dhabi Municipality\\t...'. Feeds the same
    admin1 table under a synthetic code, because the export's middle field is
    sometimes a sub-region (UK 'East Midlands') that is not a first-level
    division. Without this the parser reads it as the city name."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            parts = p[0].split(".")
            if len(parts) < 3:
                continue
            cc, a1, a2 = parts[0], parts[1], parts[2]
            name, ascii_name = p[1], p[2]
            rows.append((cc, f"{a1}.{a2}#2", name, normalize(ascii_name or name), normalize(name)))
    execute_values(cur, """
        INSERT INTO geo_admin1 (country_code, admin1_code, name, normalized_name, normalized_name_local)
        VALUES %s ON CONFLICT (country_code, admin1_code) DO NOTHING
    """, rows)
    return len(rows)


def load_zcta(cur, path: str) -> int:
    """2023_Gaz_zcta_national.txt: tab-separated, GEOID + INTPTLAT + INTPTLONG.
    Fields are space-padded, so everything is stripped."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if "GEOID" not in header.upper():
            raise SystemExit(f"unexpected ZCTA header: {header[:80]!r}")
        for line in f:
            p = [x.strip() for x in line.split("\t")]
            if len(p) < 7 or not p[0]:
                continue
            try:
                rows.append((p[0], "US", float(p[5]), float(p[6])))
            except ValueError:
                continue
    execute_values(cur, """
        INSERT INTO geo_postal (postal_code, country_code, latitude, longitude)
        VALUES %s ON CONFLICT (country_code, postal_code) DO UPDATE
            SET latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude
    """, rows, page_size=5000)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    p.add_argument("--port", default=os.environ.get("DB_PORT", "5432"))
    p.add_argument("--dbname", default=os.environ.get("DB_NAME"))
    p.add_argument("--user", default=os.environ.get("DB_USER"))
    p.add_argument("--password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--sslmode", default=os.environ.get("DB_SSLMODE", "require"))
    args = p.parse_args()

    def f(name):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run lookups/00_fetch_reference_data.py first")
        return path

    conn = psycopg2.connect(
        dbname=args.dbname or os.environ.get("DB_NAME"),
        user=args.user or os.environ.get("DB_USER"),
        password=args.password or os.environ.get("DB_PASSWORD"),
        host=args.host, port=args.port, sslmode=args.sslmode,
    )
    cur = conn.cursor()

    t0 = time.time()
    print("-- countries --")
    n = load_countries(cur, f("countryInfo.txt"))
    print(f"   {n:,} countries")
    conn.commit()

    print("-- admin1 / admin2 --")
    n1 = load_admin1(cur, f("admin1CodesASCII.txt"))
    n2 = load_admin2(cur, f("admin2Codes.txt"))
    print(f"   {n1:,} admin1 + {n2:,} admin2 name rows")
    conn.commit()

    print("-- cities (this is the big one) --")
    t1 = time.time()
    nc = load_cities(cur, f("cities500.txt"))
    print(f"   {nc:,} cities in {time.time() - t1:.1f}s")
    conn.commit()

    print("-- ZCTA postal centroids --")
    nz = load_zcta(cur, f("2023_Gaz_zcta_national.txt"))
    print(f"   {nz:,} postal codes")
    conn.commit()

    # Country centroids as the mean of their cities. Done in SQL because the rows
    # are already there and this is a one-time aggregate.
    print("-- country centroids (mean of member cities) --")
    cur.execute("""
        UPDATE geo_country gc SET latitude = c.lat, longitude = c.lon
        FROM (
            SELECT country_code, avg(latitude) AS lat, avg(longitude) AS lon
            FROM geo_cities WHERE country_code IS NOT NULL
            GROUP BY country_code
        ) c
        WHERE gc.iso2 = c.country_code
    """)
    print(f"   {cur.rowcount:,} countries given a centroid")
    cur.execute("""
        UPDATE country_aliases ca SET centroid = ST_SetSRID(ST_MakePoint(gc.longitude, gc.latitude), 4326)
        FROM geo_country gc WHERE ca.iso2 = gc.iso2 AND gc.latitude IS NOT NULL
    """)
    print(f"   {cur.rowcount:,} alias rows given a centroid")

    # Admin1 centroids, same trick, for the tier-4 fallback.
    print("-- admin1 centroids --")
    cur.execute("""
        UPDATE geo_admin1 a SET latitude = c.lat, longitude = c.lon
        FROM (
            SELECT country_code, admin1_code, avg(latitude) AS lat, avg(longitude) AS lon
            FROM geo_cities WHERE admin1_code IS NOT NULL AND country_code IS NOT NULL
            GROUP BY country_code, admin1_code
        ) c
        WHERE a.country_code = c.country_code AND a.admin1_code = c.admin1_code
    """)
    print(f"   {cur.rowcount:,} admin1 rows given a centroid")

    # admin2 rows carry synthetic codes of the form "<a1>.<a2>#2", so they were
    # never in the geo_cities join above and have no centroid of their own. The
    # parser still recognises them -- UK "East Midlands" is an admin2, not an
    # admin1 -- so without this they reach tier 3 with a NULL coordinate and fall
    # all the way to a country centroid. Give them their parent's.
    cur.execute("""
        UPDATE geo_admin1 a
        SET latitude = p.latitude, longitude = p.longitude
        FROM geo_admin1 p
        WHERE a.admin1_code LIKE '%#2'
          AND a.latitude IS NULL
          AND p.country_code = a.country_code
          AND p.admin1_code = split_part(a.admin1_code, '.', 1)
          AND p.latitude IS NOT NULL
    """)
    print(f"   {cur.rowcount:,} admin2 rows inherited their parent's centroid")
    conn.commit()

    print("\n-- seeding country aliases --")
    lookup = build_country_lookup()
    rows = [(raw, iso2, raw) for raw, iso2 in lookup.items()]
    execute_values(cur, """
        INSERT INTO country_aliases (raw_country, iso2, canonical_name)
        VALUES %s ON CONFLICT (raw_country) DO UPDATE SET iso2 = EXCLUDED.iso2
    """, rows, page_size=5000)
    conn.commit()
    print(f"   {len(rows):,} alias strings")

    # Must run AFTER the aliases are seeded -- the first pass above matched zero
    # rows because the table it was updating was still empty.
    cur.execute("""
        UPDATE country_aliases ca
        SET centroid = ST_SetSRID(ST_MakePoint(gc.longitude, gc.latitude), 4326)
        FROM geo_country gc
        WHERE ca.iso2 = gc.iso2 AND gc.latitude IS NOT NULL
    """)
    print(f"   {cur.rowcount:,} alias rows given a centroid")
    conn.commit()

    print("\n-- summary --")
    for table in ("geo_country", "geo_admin1", "geo_cities", "geo_postal", "country_aliases"):
        cur.execute(f"SELECT count(*) FROM {table}")
        print(f"   {table:<18} {cur.fetchone()[0]:>10,}")

    cur.close()
    conn.close()
    print(f"\ndone in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
