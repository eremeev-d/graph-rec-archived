import argparse
import sqlite3

import pandas as pd


def prepare_items_db(items_path, db_path):
    items = pd.read_csv(items_path)
    with sqlite3.connect(db_path) as conn:
        items.to_sql("items", conn, if_exists="replace", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare items database from a CSV file.")
    parser.add_argument("--items_path", required=True, type=str, help="Path to the CSV file containing items.")
    parser.add_argument("--db_path", required=True, type=str, help="Path to the SQLite database file.")
    
    args = parser.parse_args()
    
    prepare_items_db(args.items_path, args.db_path)