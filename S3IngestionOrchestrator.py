import os
import argparse
import boto3
import psycopg2
from botocore.exceptions import NoCredentialsError, ClientError
import tempfile

# Import the ingestion task from your existing script
# Assumes CSVDataIngestion.py is in the same directory
try:
    from CSVDataIngestion import ingest_csv_task
except ImportError:
    print("Error: Could not import 'ingest_csv_task'.")
    print("Please ensure 'CSVDataIngestion.py' is in the same directory as this script.")
    exit(1)

def get_db_connection(args):
    """Establishes and returns a database connection."""
    print("Connecting to the PostgreSQL database...")
    try:
        conn = psycopg2.connect(
            dbname=args.dbname,
            user=args.user,
            password=args.password,
            host=args.host,
            port=args.port
        )
        print("-> Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"-> Database connection failed: {e}")
        return None

def get_processed_files(conn):
    """
    Queries the database to get a set of all source_file_name
    that have already been ingested.
    """
    print("Fetching list of already processed files from the database...")
    processed_files = set()
    try:
        with conn.cursor() as cur:
            # Ensure the table exists before querying
            cur.execute("SELECT to_regclass('public.raw_data');")
            if cur.fetchone()[0]:
                 cur.execute("SELECT DISTINCT source_file_name FROM raw_data;")
                 rows = cur.fetchall()
                 for row in rows:
                     processed_files.add(row[0])
            else:
                print("-> 'raw_data' table does not exist. Assuming no files processed.")
        
        print(f"-> Found {len(processed_files)} previously ingested files.")
    except psycopg2.Error as e:
        print(f"-> Error fetching processed files: {e}")
        conn.rollback() # Rollback the failed transaction
        raise
    return processed_files

def list_s3_keys(bucket_name, s3_prefix):
    """
    Lists the full keys of all CSV files in a given S3 bucket and prefix.
    """
    print(f"Listing CSV files in S3 bucket '{bucket_name}' with prefix '{s3_prefix}'...")
    s3_keys = []
    try:
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
        for page in pages:
            if "Contents" in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Ensure it's a CSV file and not a directory placeholder
                    if key.lower().endswith('.csv'):
                        s3_keys.append(key)
        print(f"-> Found {len(s3_keys)} CSV files in S3.")
    except NoCredentialsError:
        print("-> AWS credentials not found. Please configure them.")
        return None
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"-> Error: The bucket '{bucket_name}' does not exist.")
        else:
            print(f"-> An S3 client error occurred: {e}")
        return None
    return s3_keys


def main():
    """Main function to orchestrate the S3 CSV ingestion process."""
    parser = argparse.ArgumentParser(description="Orchestrate ingestion of CSV files from an S3 bucket.")
    # S3 arguments
    parser.add_argument("--s3-bucket", required=True, help="The S3 bucket name.")
    parser.add_argument("--s3-prefix", default="", help="The prefix (directory) in the S3 bucket where CSVs are located.")
    # Database arguments (with environment variable fallbacks)
    parser.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"), help="Database host")
    parser.add_argument("--port", default=os.environ.get("DB_PORT", "5432"), help="Database port")
    parser.add_argument("--dbname", default=os.environ.get("DB_NAME"), help="Database name")
    parser.add_argument("--user", default=os.environ.get("DB_USER"), help="Database user")
    parser.add_argument("--password", default=os.environ.get("DB_PASSWORD"), help="Database password")
    args = parser.parse_args()

    if not all([args.dbname, args.user, args.password]):
        raise ValueError("Database name, user, and password must be provided.")

    conn = None
    try:
        # 1. Connect to the database
        conn = get_db_connection(args)
        if not conn:
            return

        # 2. Get the list of already processed files
        processed_files = get_processed_files(conn)

        # 3. Get the list of file keys present in the S3 directory
        s3_keys = list_s3_keys(args.s3_bucket, args.s3_prefix)
        if s3_keys is None:
            return

        # 4. Determine which files are new by comparing basenames
        keys_to_process = [
            key for key in s3_keys if os.path.basename(key) not in processed_files
        ]

        if not keys_to_process:
            print("\nNo new files to process. Database is up-to-date.")
            return

        print(f"\nFound {len(keys_to_process)} new files to ingest.")

        # 5. Process each new file
        s3_resource = boto3.resource('s3')
        for s3_key in keys_to_process:
            file_name = os.path.basename(s3_key)
            temp_file_path = None
            try:
                # Create a temporary file to download the S3 object to
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmp:
                    temp_file_path = tmp.name
                
                print(f"\nDownloading '{s3_key}' to '{temp_file_path}'...")
                s3_resource.Bucket(args.s3_bucket).download_file(s3_key, temp_file_path)
                print("-> Download complete.")

                # Call the ingestion task from the other script
                print(f"Starting ingestion for '{file_name}'...")
                ingest_csv_task(csv_file_path=temp_file_path, conn=conn)

            except ClientError as e:
                print(f"-> Failed to download or process {file_name}: {e}")
            except Exception as e:
                print(f"-> An unexpected error occurred during ingestion of {file_name}: {e}")
                # If the ingestion fails, we should rollback the transaction for that file
                if conn:
                    conn.rollback()
            finally:
                # Clean up the temporary file if it was created
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"-> Cleaned up temporary file: {temp_file_path}")

        print("\n--- S3 ingestion process finished ---")

    except Exception as e:
        print(f"An error occurred during the orchestration process: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
