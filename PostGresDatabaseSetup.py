import os
import argparse
import psycopg2
from psycopg2 import sql

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database using credentials
    from environment variables, with command-line arguments as fallbacks.
    """
    parser = argparse.ArgumentParser(description="Setup PostgreSQL database for the ETL pipeline and Chat features.")
    parser.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"), help="Database host")
    parser.add_argument("--port", default=os.environ.get("DB_PORT", "5432"), help="Database port")
    parser.add_argument("--dbname", default=os.environ.get("DB_NAME"), help="Database name")
    parser.add_argument("--user", default=os.environ.get("DB_USER"), help="Database user")
    parser.add_argument("--password", default=os.environ.get("DB_PASSWORD"), help="Database password")
    args = parser.parse_args()

    if not all([args.dbname, args.user, args.password]):
        raise ValueError("Database name, user, and password must be provided via arguments or environment variables.")

    conn = psycopg2.connect(
        dbname=args.dbname,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port
    )
    return conn

def setup_database():
    """
    Connects to the database and executes all necessary SQL commands
    to set up the extensions and tables for the pipeline and chat features.
    """
    conn = None
    print("Starting database setup...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. Enable Required Extensions
        print("Enabling PostGIS & pgcrypto extensions...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;") # Needed for gen_random_uuid()
        print("-> Extensions enabled successfully.")

        # 2. Create 'raw_data' Table
        print("Creating 'raw_data' table...")
        raw_data_sql = """
        CREATE TABLE IF NOT EXISTS raw_data (
            id BIGSERIAL PRIMARY KEY,
            nct_number TEXT,
            study_title TEXT,
            study_url TEXT,
            acronym TEXT,
            study_status TEXT,
            brief_summary TEXT,
            study_results TEXT,
            conditions TEXT,
            interventions TEXT,
            sponsor TEXT,
            collaborators TEXT,
            sex TEXT,
            age TEXT,
            phases TEXT,
            enrollment TEXT,
            study_type TEXT,
            other_ids TEXT,
            start_date TEXT,
            primary_completion_date TEXT,
            completion_date TEXT,
            first_posted TEXT,
            last_update_posted TEXT,
            locations TEXT,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source_file_name TEXT NOT NULL
        );
        """
        cur.execute(raw_data_sql)
        print("-> 'raw_data' table created successfully.")

        # 3. Create 'processed_data' Table and Index
        print("Creating 'processed_data' table...")
        processed_data_sql = """
        CREATE TABLE IF NOT EXISTS processed_data (
            nct_number TEXT PRIMARY KEY,
            study_title TEXT,
            study_url TEXT,
            status VARCHAR(50),
            conditions TEXT,
            interventions TEXT,
            sponsor TEXT,
            sex VARCHAR(20),
            age_group VARCHAR(50),
            phase VARCHAR(50),
            enrollment_count INTEGER,
            study_type VARCHAR(100),
            start_date DATE,
            completion_date DATE,
            first_posted_date DATE,
            last_updated_date DATE,
            location_geom GEOMETRY(Point, 4326)
        );
        """
        cur.execute(processed_data_sql)
        print("-> 'processed_data' table created successfully.")

        print("Creating GiST index on 'processed_data'...")
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_processed_data_location
        ON processed_data
        USING GIST (location_geom);
        """
        cur.execute(index_sql)
        print("-> GiST index created successfully.")

        # --- NEW TABLES FOR CHAT FUNCTIONALITY ---

        # 4. Create 'users' table for Clerk synchronization
        print("Creating 'users' table for chat feature...")
        users_sql = """
        CREATE TABLE IF NOT EXISTS users (
            clerk_user_id VARCHAR(255) PRIMARY KEY NOT NULL,
            role VARCHAR(50) NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        cur.execute(users_sql)
        print("-> 'users' table created successfully.")

        # 5. Create 'conversations' table
        print("Creating 'conversations' table for chat feature...")
        conversations_sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        cur.execute(conversations_sql)
        print("-> 'conversations' table created successfully.")

        # 6. Create 'participants' table
        print("Creating 'participants' table for chat feature...")
        participants_sql = """
        CREATE TABLE IF NOT EXISTS participants (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id VARCHAR(255) NOT NULL REFERENCES users(clerk_user_id) ON DELETE CASCADE,
            joined_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (conversation_id, user_id)
        );
        """
        cur.execute(participants_sql)
        print("-> 'participants' table created successfully.")
        
        # 7. Create 'messages' table
        print("Creating 'messages' table for chat feature...")
        messages_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            sender_id VARCHAR(255) NOT NULL REFERENCES users(clerk_user_id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            sent_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        cur.execute(messages_sql)
        print("-> 'messages' table created successfully.")
        
        # 8. Create Indexes for Chat Performance
        print("Creating indexes for chat tables...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id, sent_at DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_participants_user_id ON participants (user_id);")
        print("-> Chat indexes created successfully.")
        
        conn.commit()
        cur.close()
        print("\nDatabase setup completed successfully!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"\nError during database setup: {error}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    setup_database()