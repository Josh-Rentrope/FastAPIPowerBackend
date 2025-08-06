#!/usr/bin/env python3

import os
import argparse
import psycopg2
from psycopg2 import sql

def get_db_config():
    """
    Parses command-line arguments and environment variables to get database config.
    Command-line arguments override environment variables.
    """
    parser = argparse.ArgumentParser(
        description="Connect to a PostgreSQL database to list tables and columns.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Argument Definitions ---
    parser.add_argument(
        '-H', '--host',
        default=os.environ.get('PGHOST'),
        help="Database host or IP address.\n(env: PGHOST)"
    )
    parser.add_argument(
        '-d', '--dbname',
        default=os.environ.get('PGDATABASE'),
        help="Database name to connect to.\n(env: PGDATABASE)"
    )
    parser.add_argument(
        '-u', '--user',
        default=os.environ.get('PGUSER'),
        help="Database user name.\n(env: PGUSER)"
    )
    parser.add_argument(
        '-p', '--password',
        default=os.environ.get('PGPASSWORD'),
        help="Password for the database user.\n(env: PGPASSWORD)"
    )
    parser.add_argument(
        '-P', '--port',
        default=os.environ.get('PGPORT', '5432'),
        help="Database server port.\n(env: PGPORT, default: 5432)"
    )
    parser.add_argument(
        '-s', '--schema',
        default='public',
        help="Database schema to inspect.\n(default: public)"
    )
    parser.add_argument(
        '-f', '--format',
        choices=['table', 'tree'],
        default='table',
        help="Output format for the schema.\n(default: table)"
    )

    args = parser.parse_args()

    # --- Validation ---
    # Check if essential connection details are provided
    if not all([args.host, args.dbname, args.user, args.password]):
        parser.error(
            "Missing required connection info. "
            "Please provide host, dbname, user, and password "
            "via arguments or environment variables."
        )

    return args

def get_schema_details(config):
    """
    Connects to the database and fetches schema details.

    Returns:
        A dictionary where keys are table names and values are lists of
        (column_name, data_type) tuples. Returns None on connection failure.
    """
    schema_data = {}
    conn = None
    try:
        # Establish the database connection
        conn = psycopg2.connect(
            dbname=config.dbname,
            user=config.user,
            password=config.password,
            host=config.host,
            port=config.port
        )
        print(f"Successfully connected to '{config.dbname}' on '{config.host}'.\n")
        cur = conn.cursor()

        # --- Query to get all table names in the specified schema ---
        table_query = sql.SQL("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            ORDER BY table_name;
        """)
        cur.execute(table_query, (config.schema,))
        tables = [row[0] for row in cur.fetchall()]

        if not tables:
            print(f"No tables found in schema '{config.schema}'.")
            return None

        # --- Query to get columns for each table ---
        column_query = sql.SQL("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = %s
            ORDER BY ordinal_position;
        """)
        for table_name in tables:
            cur.execute(column_query, (table_name, config.schema))
            columns = cur.fetchall()
            schema_data[table_name] = columns

        cur.close()
        return schema_data

    except psycopg2.OperationalError as e:
        print(f"Connection Error: Could not connect to the database.")
        print(f"Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        if conn is not None:
            conn.close()

def print_as_table(schema_data):
    """Formats the schema information as a series of tables."""
    print("--- Schema Output (Table Format) ---")
    for table, columns in schema_data.items():
        print("\n" + "=" * 50)
        print(f" Table: {table}")
        print("-" * 50)
        if columns:
            # Find the longest column name for alignment
            max_len = max(len(col[0]) for col in columns)
            print(f" {'COLUMN'.ljust(max_len)} | DATA_TYPE")
            print(f" {'-' * max_len}-+-----------")
            for col_name, col_type in columns:
                print(f" {col_name.ljust(max_len)} | {col_type}")
        else:
            print(" (No columns found)")
        print("=" * 50)

def print_as_tree(schema_data):
    """Formats the schema information as a tree structure."""
    print("--- Schema Output (Tree Format) ---")
    for i, (table, columns) in enumerate(schema_data.items()):
        is_last_table = (i == len(schema_data) - 1)
        print(f" {'└' if is_last_table else '├'}── {table}")
        if columns:
            for j, (col_name, col_type) in enumerate(columns):
                is_last_column = (j == len(columns) - 1)
                tree_prefix = ' ' * 4
                col_prefix = '└' if is_last_column else '├'
                print(f" {tree_prefix}{col_prefix}── {col_name} ({col_type})")

def main():
    """Main function to drive the script."""
    config = get_db_config()
    schema_data = get_schema_details(config)

    if schema_data:
        if config.format == 'table':
            print_as_table(schema_data)
        elif config.format == 'tree':
            print_as_tree(schema_data)

if __name__ == "__main__":
    main()
