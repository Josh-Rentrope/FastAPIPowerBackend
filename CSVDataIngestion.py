import os
import argparse
from datetime import datetime
import time

# --- Configuration ---
# Set to True to run as an Airflow DAG, False to run as a standalone script.
AIRFLOW_ENABLED = False

# When running in Airflow, connection details should be stored in an Airflow Connection.
POSTGRES_CONN_ID = "postgres_clinical_trials"
DEFAULT_CSV_PATH = '/opt/airflow/dags/data/ctg-studies-short.csv' # Example path in Airflow environment

# --- Conditional Imports and Stubs for Airflow ---
if AIRFLOW_ENABLED:
    from airflow.decorators import dag, task
    from airflow.providers.postgres.hooks.postgres import PostgresHook
else:
    # If Airflow is not enabled, create stubs for the decorators
    print("Airflow is not enabled. Using decorator stubs.")
    def decorator_stub(*args, **kwargs):
        def decorator(f):
            def wrapper(*a, **kw):
                return f(*a, **kw)
            return wrapper
        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator
    task = decorator_stub
    dag = decorator_stub

def get_db_connection_for_airflow():
    """
    Gets a raw psycopg2 connection using the Airflow Hook.
    This function is only used when running within an Airflow DAG.
    """
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    return hook.get_conn()

# --- Task Definition ---
@task(task_id="ingest_csv_to_raw_data")
def ingest_csv_task(csv_file_path: str, table_name: str = 'raw_data', conn=None):
    """
    Connects to PostgreSQL and uses the COPY command to bulk-load a CSV file.
    Accepts an optional connection object for local execution.
    """
    is_local_run = conn is not None
    print(f"Starting ingestion of {csv_file_path} into {table_name}.")
    try:
        if not is_local_run:
            # If no connection is passed, we are in an Airflow run.
            conn = get_db_connection_for_airflow()
        
        print("Database connection successful.")
        cur = conn.cursor()

        with open(csv_file_path, 'r', encoding='utf-8') as f:
            # Skip the header row
            next(f)
            
            file_name = os.path.basename(csv_file_path)
            temp_table_name = f"temp_ingest_{int(time.time())}"
            
            # Create a temporary table that matches the CSV structure
            cur.execute(f"""
                CREATE TEMP TABLE {temp_table_name} (
                    nct_number TEXT, study_title TEXT, study_url TEXT, acronym TEXT, study_status TEXT,
                    brief_summary TEXT, study_results TEXT, conditions TEXT, interventions TEXT,
                    sponsor TEXT, collaborators TEXT, sex TEXT, age TEXT, phases TEXT, enrollment TEXT,
                    study_type TEXT, other_ids TEXT, start_date TEXT, primary_completion_date TEXT,
                    completion_date TEXT, first_posted TEXT, last_update_posted TEXT, locations TEXT
                );
            """)
            
            # Use copy_expert to efficiently load data into the temporary table
            cur.copy_expert(f"COPY {temp_table_name} FROM STDIN WITH CSV", f)
            
            # Insert from the temp table into the final table, adding the source filename
            cur.execute(f"""
                INSERT INTO {table_name} (
                    nct_number, study_title, study_url, acronym, study_status,
                    brief_summary, study_results, conditions, interventions,
                    sponsor, collaborators, sex, age, phases, enrollment,
                    study_type, other_ids, start_date, primary_completion_date,
                    completion_date, first_posted, last_update_posted, locations,
                    source_file_name
                )
                SELECT *, '{file_name}' FROM {temp_table_name};
            """)
            
            # Drop the temporary table
            cur.execute(f"DROP TABLE {temp_table_name};")

        conn.commit()
        cur.close()
        print(f"Successfully loaded data from {csv_file_path} into {table_name}.")

    except Exception as error:
        print(f"Error during data ingestion: {error}")
        if conn:
            conn.rollback()
        # Re-raise the exception to make the Airflow task fail
        raise
    finally:
        # Only close the connection if it was created by this task (i.e., in an Airflow run)
        if conn and not is_local_run:
            conn.close()
            print("Database connection closed.")

# --- DAG Definition ---
@dag(
    dag_id="csv_ingestion_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['ingestion', 'csv'],
    doc_md="""
    ### CSV Ingestion Pipeline
    This DAG ingests a specified CSV file into the `raw_data` table.
    It can be triggered manually and provided with a path to the CSV file via the `params` config.
    Example config: `{"csv_path": "/path/to/my/data.csv"}`
    """
)
def ingestion_dag():
    """Defines the CSV ingestion DAG."""
    csv_path = "{{ params.get('csv_path', '" + DEFAULT_CSV_PATH + "') }}"
    ingest_csv_task(csv_file_path=csv_path)

# --- Execution Logic ---
if AIRFLOW_ENABLED:
    csv_ingestion_dag = ingestion_dag()
else:
    def run_local():
        # Setup argument parser for local execution
        parser = argparse.ArgumentParser(description="Run the CSV ingestion script locally.")
        parser.add_argument("csv_path", help="The full path to the CSV file to ingest.")
        parser.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"), help="Database host")
        parser.add_argument("--port", default=os.environ.get("DB_PORT", "5432"), help="Database port")
        parser.add_argument("--dbname", default=os.environ.get("DB_NAME"), help="Database name")
        parser.add_argument("--user", default=os.environ.get("DB_USER"), help="Database user")
        parser.add_argument("--password", default=os.environ.get("DB_PASSWORD"), help="Database password")
        args = parser.parse_args()

        if not all([args.dbname, args.user, args.password]):
            raise ValueError("Database name, user, and password must be provided via arguments or environment variables.")

        conn = None
        print("\n--- Running CSV ingestion script locally ---")
        try:
            # Import psycopg2 only for local runs
            import psycopg2
            conn = psycopg2.connect(
                dbname=args.dbname,
                user=args.user,
                password=args.password,
                host=args.host,
                port=args.port
            )
            # Pass the established connection to the task function
            ingest_csv_task(csv_file_path=args.csv_path, conn=conn)
        except Exception as e:
            print(f"Local run failed: {e}")
        finally:
            if conn:
                conn.close()
            print("--- Local execution finished ---\n")

    if __name__ == "__main__":
        run_local()
