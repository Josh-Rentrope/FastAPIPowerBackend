# ingest_county_data.py
import os
import argparse
import psycopg2
import pandas as pd
import json
from io import StringIO

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    parser = argparse.ArgumentParser(description="Setup and ingest County Health Rankings and Boundary data.")
    parser.add_argument("--csv-path", required=True, help="Path to the County Health Rankings CSV file.")
    parser.add_argument("--geojson-path", required=True, help="Path to the County Boundaries GeoJSON file.")
    parser.add_argument("--db-host", default=os.environ.get("DB_HOST", "localhost"))
    parser.add_argument("--db-port", default=os.environ.get("DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.environ.get("DB_NAME"))
    parser.add_argument("--db-user", default=os.environ.get("DB_USER"))
    parser.add_argument("--db-password", default=os.environ.get("DB_PASSWORD"))
    args = parser.parse_args()

    if not all([args.db_name, args.db_user, args.db_password]):
        raise ValueError("Database name, user, and password must be provided.")

    conn = psycopg2.connect(
        dbname=args.db_name, user=args.db_user, password=args.db_password,
        host=args.db_host, port=args.db_port
    )
    return conn, args.csv_path, args.geojson_path

def setup_tables(cur):
    """Creates the necessary tables for county health and boundary data."""
    print("Creating or updating tables...")
    # Health Rankings Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS county_health_rankings (
        id SERIAL PRIMARY KEY, fipscode TEXT UNIQUE NOT NULL, state_name TEXT, county_name TEXT,
        v001_raw_value FLOAT, v011_raw_value FLOAT, v009_raw_value FLOAT,
        v057_raw_value FLOAT, v043_raw_value FLOAT, v021_raw_value FLOAT,
        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    # Table for County Boundaries
    cur.execute("""
    CREATE TABLE IF NOT EXISTS county_boundaries (
        fipscode TEXT PRIMARY KEY,
        county_name TEXT,
        geom GEOMETRY(MultiPolygon, 4326) NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_county_boundaries_geom ON county_boundaries USING GIST (geom);
    """)
    print("-> Tables are ready.")

def ingest_geojson_boundaries(conn, geojson_path):
    """Ingests county boundary polygons from an ESRI-formatted JSON file."""
    print(f"Ingesting county boundaries from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE county_boundaries RESTART IDENTITY;")
        print("-> Cleared existing data from county_boundaries.")
        
        ingested_count = 0
        for feature in geojson_data.get('features', []):
            properties = feature.get('attributes')
            esri_geometry = feature.get('geometry')

            if properties and esri_geometry and 'FIPS' in properties and 'NAME' in properties and 'rings' in esri_geometry:
                try:
                    county_fips_num = int(properties['FIPS'])
                    fips = f"37{county_fips_num:03d}"
                    name = properties['NAME']
                    
                    # --- FIX: Add an extra level of nesting to the coordinates array ---
                    # A GeoJSON MultiPolygon's coordinates are an array of polygon coordinate arrays.
                    # Each polygon coordinate array is an array of linear ring coordinate arrays.
                    # The ESRI 'rings' format is equivalent to a single polygon's linear rings.
                    # We must wrap it in another array to make it a valid MultiPolygon.
                    geojson_geometry = {
                        "type": "MultiPolygon",
                        "coordinates": [esri_geometry['rings']] # Wrap 'rings' in an array
                    }
                    geom_json_string = json.dumps(geojson_geometry)
                    
                    cur.execute(
                        """
                        INSERT INTO county_boundaries (fipscode, county_name, geom) 
                        VALUES (%s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)) 
                        ON CONFLICT (fipscode) 
                        DO UPDATE SET county_name = EXCLUDED.county_name, geom = EXCLUDED.geom;
                        """,
                        (fips, name, geom_json_string)
                    )
                    ingested_count += 1
                except (ValueError, TypeError, KeyError) as e:
                    print(f"Skipping feature due to data conversion error: {e} in {properties}")
            else:
                print(f"Skipping malformed feature (missing keys): {feature}")

        conn.commit()
    print(f"-> Successfully ingested {ingested_count} county boundaries.")


def ingest_chrr_data(conn, csv_file_path):
    """Ingests data from the CHR&R CSV file into the database."""
    print(f"Starting ingestion of data from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, header=0, skiprows=[1], low_memory=False)

        column_map_to_db = {
            'fipscode': '5-digit FIPS Code', 'state_name': 'State Abbreviation', 'county_name': 'Name',
            'v001_raw_value': 'Premature Death raw value', 'v011_raw_value': 'Adult obesity raw value',
            'v009_raw_value': 'Adult smoking raw value', 'v057_raw_value': 'Children in poverty raw value',
            'v043_raw_value': '% Uninsured raw value', 'v021_raw_value': '% 65 and over raw value',
        }

        df.columns = [col.strip().lower().replace('%', 'pct').replace(' ', '_').replace('-', '_').replace('/', '_') for col in df.columns]

        rename_map = {}
        for db_col, original_csv_col in column_map_to_db.items():
            cleaned_csv_col = original_csv_col.strip().lower().replace('%', 'pct').replace(' ', '_').replace('-', '_').replace('/', '_')
            if cleaned_csv_col in df.columns:
                rename_map[cleaned_csv_col] = db_col
        
        if 'fipscode' not in rename_map.values():
            raise KeyError("The essential 'fipscode' column could not be found.")

        df_to_ingest = df[list(rename_map.keys())].rename(columns=rename_map)
        
        for col in df_to_ingest.columns:
            if col.endswith('_raw_value'):
                df_to_ingest[col] = pd.to_numeric(df_to_ingest[col], errors='coerce')

        df_to_ingest['fipscode'] = pd.to_numeric(df_to_ingest['fipscode'], errors='coerce').astype('Int64').astype(str).str.zfill(5)

        buffer = StringIO()
        df_to_ingest.to_csv(buffer, index=False, header=False, na_rep='')
        buffer.seek(0)

        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE county_health_rankings RESTART IDENTITY;")
            db_columns = ','.join(df_to_ingest.columns)
            copy_sql = f"COPY county_health_rankings({db_columns}) FROM stdin WITH CSV"
            cur.copy_expert(sql=copy_sql, file=buffer)
            conn.commit()
        
        print(f"-> Successfully ingested and committed {len(df_to_ingest)} rows from CSV.")

    except Exception as e:
        print(f"An error occurred during CHR&R ingestion: {e}")
        conn.rollback()


def main():
    """Main function to set up tables and ingest all data."""
    conn = None
    try:
        conn, csv_path, geojson_path = get_db_connection()
        with conn.cursor() as cur:
            setup_tables(cur)
            conn.commit()
        
        ingest_geojson_boundaries(conn, geojson_path)
        ingest_chrr_data(conn, csv_path)

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error in main execution: {error}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
