#!/usr/bin/env python3
"""Download the offline reference data the geocoder needs.

Everything here is fetched once and cached in `lookups/`. Nothing is called at
query time and nothing costs money per lookup, which is the whole point: the
alternative is a geocoding API at ~3.5M requests.

Sources and terms:

* **GeoNames** (cities500, countryInfo, admin1CodesASCII) -- Creative Commons
  Attribution 4.0. The attribution requirement is real, so it belongs in the
  deployment's about page if this is ever shown publicly.
* **US Census ZCTA gazetteer** -- public domain. Gives a centroid per ZIP code
  tabulation area, which is the most precise rung available offline for US
  addresses.
* **US Census TIGER/Line counties** -- public domain. Used by the county stage.
* **County Health Rankings & Roadmaps** -- the analytic data file. Redistribution
  terms were NOT verified; see the note in the README before publishing this.

Usage:
    uv run python lookups/00_fetch_reference_data.py
    uv run python lookups/00_fetch_reference_data.py --only zcta
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, url, destination filename). A URL ending .zip is extracted.
SOURCES: list[tuple[str, str, str]] = [
    # ~200k populated places worldwide with population > 500. cities500 rather
    # than cities1000 because trial sites are frequently small towns.
    ("geonames_cities",
     "https://download.geonames.org/export/dump/cities500.zip", "cities500.txt"),

    # Country names and ISO codes.
    ("geonames_countries",
     "https://download.geonames.org/export/dump/countryInfo.txt", "countryInfo.txt"),

    # admin1 = state / province / region names, per country. This is what lets the
    # parser tell "Kettering, Ohio, United States" (city, state) apart from
    # "Novartis Investigator Site, Gummersbach, Germany" (facility, city).
    ("geonames_admin1",
     "https://download.geonames.org/export/dump/admin1CodesASCII.txt", "admin1CodesASCII.txt"),

    # admin2 as well: the export's third field is sometimes a sub-region that is
    # not a first-level division (UK "East Midlands", Chinese municipalities).
    ("geonames_admin2",
     "https://download.geonames.org/export/dump/admin2Codes.txt", "admin2Codes.txt"),

    # ZIP -> centroid.
    ("zcta",
     "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip",
     "2023_Gaz_zcta_national.txt"),

    ("tiger_counties",
     "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip",
     "tl_2024_us_county.shp"),

    ("county_health_rankings",
     "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2025_v2.csv",
     "analytic_data2025_v2.csv"),
]


# countyhealthrankings.org answers 403 to urllib's default User-Agent while
# serving the same URL fine to curl. Identifying the client fixes it, and it is
# the polite thing to do regardless.
USER_AGENT = "clinical-trials-etl/1.0 (+https://codingshinobi.org)"


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"   cached  {os.path.basename(dest)}")
        return
    tmp = dest + ".part"
    print(f"   fetch   {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        os.replace(tmp, dest)
        print(f" {total / 1048576:,.1f} MB")
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f" FAILED: {exc}", file=sys.stderr)
        raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", action="append",
                   help="fetch just this source (repeatable)")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    args = p.parse_args()

    wanted = SOURCES
    if args.only:
        wanted = [s for s in SOURCES if s[0] in set(args.only)]
        if not wanted:
            raise SystemExit(f"unknown source(s): {args.only}. "
                             f"known: {[s[0] for s in SOURCES]}")

    for name, url, destname in wanted:
        dest = os.path.join(HERE, destname)
        print(f"-- {name}")
        if args.force and os.path.exists(dest):
            os.remove(dest)
        if url.endswith(".zip"):
            # Land the archive first, then extract only the member we want, so a
            # 84 MB shapefile bundle does not also drop .dbf/.shx/.prj clutter
            # into the repo root.
            zpath = dest + ".zip"
            download(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                members = z.namelist()
                target = next((m for m in members if m.lower() == destname.lower()), None)
                if target is None:
                    stem = os.path.splitext(destname)[0].lower()
                    target = next((m for m in members if m.lower().startswith(stem)), None)
                if target is None:
                    print(f"   members: {members}", file=sys.stderr)
                    raise SystemExit(f"{destname} not found inside {zpath}")
                if not os.path.exists(dest):
                    z.extract(target, HERE)
                    print(f"   extract {target}")
                # shapefiles come as a set: keep every sibling .shp component
                if destname.endswith(".shp"):
                    for m in members:
                        out = os.path.join(HERE, os.path.basename(m))
                        if not os.path.exists(out):
                            z.extract(m, HERE)
        else:
            download(url, dest)

    print("\nlookups directory:")
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") or f.endswith(".zip"):
            continue
        print(f"   {f:<34} {os.path.getsize(os.path.join(HERE, f)) / 1048576:>9.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
