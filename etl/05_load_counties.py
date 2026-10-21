#!/usr/bin/env python3
"""Load US county boundaries and County Health Rankings data, nationally.

Replaces CHRNRSetup.py, which could only ever load one state:

* It hardcoded `fips = f"37{county_fips_num:03d}"`, prefixing every FIPS with 37
  -- North Carolina's state code. Every county outside NC would have been written
  with a fabricated FIPS, silently, and the correlation map would have joined
  them to the wrong places.
* It expected a legacy ArcGIS shapefile JSON (`features[].attributes.FIPS`,
  `geometry.rings`), not GeoJSON.
* It expected CHR columns named `v001_raw_value`. The current CHR release has TWO
  header rows -- human labels on the first and machine names on the second -- and
  the machine names are `v001_rawvalue`, with no underscore.

## The measure codes have drifted

Three of the six measures the front end offers are no longer at the code it asks
for. Resolving by *label* rather than by code is the only reliable way to do this,
and that is what MEASURE_MAP below encodes:

    front-end value     label                    code asked   code now
    v001_raw_value      Premature Death Rate     v001         v001
    v009_raw_value      Adult Smoking Rate       v009         v009
    v011_raw_value      Adult Obesity Rate       v011         v011
    v021_raw_value      % 65 and over            v021         v053   <- moved
    v043_raw_value      % Uninsured              v043         v085   <- moved
    v057_raw_value      Children in Poverty      v057         v024   <- moved

The database columns keep the front end's existing names, because the API
interpolates whatever name the client sends straight into SQL. So the column is
`v021_raw_value` while the source column is `v053_rawvalue` -- odd to read, and
deliberate: it means no front-end change is needed for the dropdown to work.

Usage:
    uv run python etl/05_load_counties.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import psycopg2
import shapefile  # pyshp
from psycopg2.extras import execute_values

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKUPS = os.path.join(os.path.dirname(HERE), "lookups")

# destination column -> (source machine name, human label). The label is here so
# the mapping can be re-verified by eye against a future CHR release.
MEASURE_MAP: dict[str, tuple[str, str]] = {
    "v001_raw_value": ("v001_rawvalue", "Premature Death raw value"),
    "v009_raw_value": ("v009_rawvalue", "Adult Smoking raw value"),
    "v011_raw_value": ("v011_rawvalue", "Adult Obesity raw value"),
    "v021_raw_value": ("v053_rawvalue", "% 65 and Older raw value"),
    "v043_raw_value": ("v085_rawvalue", "Uninsured raw value"),
    "v057_raw_value": ("v024_rawvalue", "Children in Poverty raw value"),
}

# STATEFP -> name, so counties carry a readable state without another download.
STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas",
    "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
    "60": "American Samoa", "66": "Guam", "69": "Northern Mariana Islands",
    "72": "Puerto Rico", "78": "Virgin Islands",
}


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


def rings_to_multipolygon_wkt(parts) -> str | None:
    """Shapefile parts -> WKT MULTIPOLYGON.

    pyshp does not say which rings are outer boundaries and which are holes; it
    just hands over a flat list. Treating every ring as its own polygon therefore
    produces "Nested shells" errors wherever a county has an interior ring (a
    lake, or a county enclaved inside another) -- 30 of 3,235 rows, before this
    was fixed.

    Rather than reimplement ring nesting, the WKT is assembled naively and handed
    to ST_MakeValid on insert, which resolves both nesting and the handful of
    genuine self-intersections TIGER carries. ST_CollectionExtract then keeps only
    the polygonal parts of whatever it returns, because ST_MakeValid can hand back
    a GeometryCollection.
    """
    if not parts:
        return None
    polys = []
    for ring in parts:
        if len(ring) < 4:
            continue
        coords = ", ".join(f"{x:.6f} {y:.6f}" for x, y in ring)
        polys.append(f"(({coords}))")
    if not polys:
        return None
    return "MULTIPOLYGON(" + ", ".join(polys) + ")"


def load_boundaries(cur, shp_path: str) -> int:
    """TIGER/Line counties. GEOID is the full 5-digit FIPS -- no prefixing."""
    base = os.path.splitext(shp_path)[0]
    reader = shapefile.Reader(shp=base + ".shp", dbf=base + ".dbf", shx=base + ".shx")
    fields = [f[0] for f in reader.fields[1:]]
    idx = {name: i for i, name in enumerate(fields)}
    for required in ("GEOID", "NAME", "STATEFP"):
        if required not in idx:
            raise SystemExit(f"shapefile is missing {required}; has {fields}")

    # Shape and record are read together so a geometry can never be paired with
    # the wrong county.
    rows, skipped = [], 0
    for sr in reader.iterShapeRecords():
        rec = sr.record
        geoid = str(rec[idx["GEOID"]]).strip().zfill(5)
        name = str(rec[idx["NAME"]]).strip()
        state_fp = str(rec[idx["STATEFP"]]).strip().zfill(2)
        wkt = rings_to_multipolygon_wkt(sr.shape.parts and
                                        [sr.shape.points[a:b] for a, b in
                                         zip(sr.shape.parts,
                                             list(sr.shape.parts[1:]) + [len(sr.shape.points)])]
                                        or [])
        if not wkt or not geoid:
            skipped += 1
            continue
        rows.append((geoid, name, state_fp, STATE_NAMES.get(state_fp), wkt))

    cur.execute("TRUNCATE TABLE county_boundaries")
    execute_values(cur, """
        INSERT INTO county_boundaries (fipscode, county_name, state_fips, state_name, geom)
        VALUES %s
        ON CONFLICT (fipscode) DO UPDATE SET
            county_name = EXCLUDED.county_name,
            state_fips  = EXCLUDED.state_fips,
            state_name  = EXCLUDED.state_name,
            geom        = EXCLUDED.geom
    """, rows, template=(
        "(%s, %s, %s, %s, "
        "ST_Multi(ST_CollectionExtract("
        "ST_MakeValid(ST_SetSRID(ST_GeomFromText(%s), 4326)), 3)))"),
        page_size=200)
    print(f"   {len(rows):,} counties loaded, {skipped} skipped for empty geometry")
    return len(rows)


def load_chr(cur, csv_path: str) -> int:
    """County Health Rankings analytic data.

    Two header rows: line 1 is human labels, line 2 is machine names. Reading
    line 1 as the header -- which is what the previous version did -- yields
    column names containing spaces that no query will ever name.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        human = next(reader)
        machine = next(reader)
        if "fipscode" not in [c.strip() for c in machine]:
            raise SystemExit(
                "expected the second row to be machine column names containing "
                f"'fipscode'; got {machine[:6]}"
            )
        col = {c.strip(): i for i, c in enumerate(machine)}
        hcol = {c.strip(): i for i, c in enumerate(human)}

        # Verify each source column is the measure we think it is, by label.
        for dest, (src, label) in MEASURE_MAP.items():
            if src not in col:
                raise SystemExit(f"CHR file has no column {src!r} for {dest}")
            actual = human[col[src]].strip()
            if actual != label:
                print(f"   NOTE {src}: expected label {label!r}, file says {actual!r}",
                      file=sys.stderr)

        rows, skipped = [], 0
        for r in reader:
            if len(r) < len(machine):
                continue
            fips = r[col["fipscode"]].strip()
            # '00000' is the US total and 'XX000' rows are state totals. Neither
            # is a county; including them would put a whole-state value on a
            # polygon-less row and skew any correlation.
            if not fips or fips == "00000" or fips.endswith("000"):
                skipped += 1
                continue

            def num(src):
                try:
                    v = r[col[src]].strip()
                    return float(v) if v not in ("", ".", "*") else None
                except (KeyError, ValueError):
                    return None

            rows.append((
                fips.zfill(5),
                r[col["state"]].strip() or None,
                r[col["county"]].strip() or None,
                *(num(src) for src, _ in MEASURE_MAP.values()),
            ))

    print(f"   {len(rows):,} county rows, {skipped} state/total rows skipped")
    execute_values(cur, """
        INSERT INTO county_health_rankings
            (fipscode, state_name, county_name, v001_raw_value, v009_raw_value,
             v011_raw_value, v021_raw_value, v043_raw_value, v057_raw_value)
        VALUES %s
        ON CONFLICT (fipscode) DO UPDATE SET
            state_name     = EXCLUDED.state_name,
            county_name    = EXCLUDED.county_name,
            v001_raw_value = EXCLUDED.v001_raw_value,
            v009_raw_value = EXCLUDED.v009_raw_value,
            v011_raw_value = EXCLUDED.v011_raw_value,
            v021_raw_value = EXCLUDED.v021_raw_value,
            v043_raw_value = EXCLUDED.v043_raw_value,
            v057_raw_value = EXCLUDED.v057_raw_value
    """, rows, page_size=500)
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

    shp = os.path.join(LOOKUPS, "tl_2024_us_county.shp")
    chr_csv = os.path.join(LOOKUPS, "analytic_data2025_v2.csv")
    for path in (shp, chr_csv):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run lookups/00_fetch_reference_data.py")

    conn = connect(args)
    cur = conn.cursor()
    t0 = time.time()

    print("-- county boundaries (TIGER/Line) --")
    t1 = time.time()
    n = load_boundaries(cur, shp)
    conn.commit()
    print(f"   {time.time()-t1:.1f}s")

    print("\n-- County Health Rankings --")
    m = load_chr(cur, chr_csv)
    conn.commit()

    print("\n-- result --")
    cur.execute("SELECT count(*), count(DISTINCT state_fips) FROM county_boundaries")
    cb, states = cur.fetchone()
    print(f"   county_boundaries      {cb:>7,} rows across {states} states/territories")
    cur.execute("SELECT count(*) FROM county_health_rankings")
    print(f"   county_health_rankings {cur.fetchone()[0]:>7,} rows")

    # How many counties can actually be joined to a trial location? This is the
    # number the correlation map depends on.
    cur.execute("""
        SELECT count(DISTINCT cb.fipscode)
        FROM county_boundaries cb
        JOIN locations l ON ST_Contains(cb.geom, l.geom)
        WHERE l.geom IS NOT NULL AND l.geocode_precision IN ('postal', 'city')
    """)
    matched = cur.fetchone()[0]
    print(f"   counties containing at least one well-placed trial location: {matched:,}")

    cur.execute("""
        SELECT count(DISTINCT cb.fipscode)
        FROM county_boundaries cb
        JOIN county_health_rankings chr ON chr.fipscode = cb.fipscode
    """)
    print(f"   counties with both geometry and CHR data: {cur.fetchone()[0]:,}")

    cur.execute("""
        SELECT count(*) FROM county_boundaries
        WHERE NOT EXISTS (SELECT 1 FROM county_health_rankings c WHERE c.fipscode = county_boundaries.fipscode)
    """)
    print(f"   counties with geometry but no CHR row:  {cur.fetchone()[0]:,}")

    cur.close()
    conn.close()
    print(f"\ndone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
