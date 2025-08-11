# ingest_county_data.py
import os
import argparse
import psycopg2
import pandas as pd
from io import StringIO

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    parser = argparse.ArgumentParser(description="Setup and ingest County Health Rankings data.")
    # Use add_argument for all parameters to avoid conflicts with the main script's args
    parser.add_argument("--csv-path", required=True, help="Path to the County Health Rankings CSV file.")
    parser.add_argument("--db-host", default=os.environ.get("DB_HOST", "localhost"))
    parser.add_argument("--db-port", default=os.environ.get("DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.environ.get("DB_NAME"))
    parser.add_argument("--db-user", default=os.environ.get("DB_USER"))
    parser.add_argument("--db-password", default=os.environ.get("DB_PASSWORD"))
    args = parser.parse_args()

    if not all([args.db_name, args.db_user, args.db_password]):
        raise ValueError("Database name, user, and password must be provided.")

    conn = psycopg2.connect(
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port
    )
    return conn, args.csv_path


def setup_county_health_table(cur):
    """Creates the county_health_rankings table."""
    print("Creating or updating 'county_health_rankings' table...")
    # This schema is expanded to better match the likely columns in the 2024 CHR&R data
    cur.execute("""
    CREATE TABLE IF NOT EXISTS county_health_rankings (
        id SERIAL PRIMARY KEY,
        fipscode TEXT UNIQUE NOT NULL,
        state_name TEXT,
        county_name TEXT,
        v001_raw_value FLOAT, -- Premature death raw value
        v011_raw_value FLOAT, -- Adult obesity raw value
        v009_raw_value FLOAT, -- Adult smoking raw value
        v057_raw_value FLOAT, -- Children in poverty raw value
        v043_raw_value FLOAT, -- % Uninsured raw value
        v021_raw_value FLOAT, -- % 65 and over raw value
        ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    print("-> 'county_health_rankings' table schema is up-to-date.")


def ingest_chrr_data(conn, csv_file_path):
    """Ingests data from the CHR&R CSV file into the database."""
    print(f"Starting ingestion of data from {csv_file_path}...")
    try:
        # Use header=0 to read the first row as headers and skiprows=[1] to skip the second row
        df = pd.read_csv(csv_file_path, header=0, skiprows=[1], low_memory=False)

        # --- More robust column mapping and renaming logic ---

        # 1. Define the mapping from the desired DB column name to the expected original CSV column name.
        column_map_to_db = {
            'fipscode': '5-digit FIPS Code',
            'state_name': 'State Abbreviation',
            'county_name': 'Name',
            'v001_raw_value': 'Premature Death raw value',
            'v011_raw_value': 'Adult obesity raw value',
            'v009_raw_value': 'Adult smoking raw value',
            'v057_raw_value': 'Children in poverty raw value',
            'v043_raw_value': '% Uninsured raw value',
            'v021_raw_value': '% 65 and over raw value',
        }

        # 2. Clean the actual column names from the DataFrame
        df.columns = [col.strip().lower().replace('%', 'pct').replace(' ', '_').replace('-', '_').replace('/', '_') for col in df.columns]

        # 3. Build the final rename mapping based on what columns actually exist
        rename_map = {}
        for db_col, original_csv_col in column_map_to_db.items():
            cleaned_csv_col = original_csv_col.strip().lower().replace('%', 'pct').replace(' ', '_').replace('-', '_').replace('/', '_')
            if cleaned_csv_col in df.columns:
                rename_map[cleaned_csv_col] = db_col
                print(f"Found column: '{original_csv_col}' as '{cleaned_csv_col}', will map to '{db_col}'")
            else:
                print(f"Warning: Column '{original_csv_col}' not found in the CSV file.")
        
        if 'fipscode' not in rename_map.values():
            raise KeyError("The essential 'fipscode' column could not be found. Please check the CSV file headers.")

        # 4. Select and rename the columns that were found
        df_to_ingest = df[list(rename_map.keys())].rename(columns=rename_map)

        # --- FIX: Perform data type conversions and formatting ---
        # Convert all value columns to numeric, coercing errors to NaN.
        # This ensures that any non-numeric values (like placeholders) become proper NULLs.
        for col in df_to_ingest.columns:
            if col.endswith('_raw_value'):
                df_to_ingest[col] = pd.to_numeric(df_to_ingest[col], errors='coerce')

        # Format fipscode correctly
        df_to_ingest['fipscode'] = pd.to_numeric(df_to_ingest['fipscode'], errors='coerce').astype('Int64').astype(str).str.zfill(5)

        # Use an in-memory buffer for the COPY operation
        buffer = StringIO()
        # --- FIX: Use an empty string for NULL, which is the default for COPY ... WITH CSV ---
        df_to_ingest.to_csv(buffer, index=False, header=False, na_rep='')
        buffer.seek(0)

        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE county_health_rankings RESTART IDENTITY;")
            print("-> Cleared existing data from county_health_rankings table.")

            db_columns = ','.join(df_to_ingest.columns)
            copy_sql = f"COPY county_health_rankings({db_columns}) FROM stdin WITH CSV"
            
            cur.copy_expert(sql=copy_sql, file=buffer)
            conn.commit()
        
        print(f"-> Successfully ingested and committed {len(df_to_ingest)} rows.")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        conn.rollback()


def main():
    """Main function to set up the table and ingest data."""
    conn = None
    try:
        conn, csv_path = get_db_connection()
        with conn.cursor() as cur:
            setup_county_health_table(cur)
            conn.commit() # Commit table creation
        
        ingest_chrr_data(conn, csv_path)

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error in main execution: {error}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()
