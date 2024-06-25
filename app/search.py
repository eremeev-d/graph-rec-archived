import sqlite3

import pandas as pd


class SearchSystem:
    def __init__(self, items_path):
        self._items_path = items_path

    def search_items(self, query, n_items=10):
        with sqlite3.connect(self._items_path) as conn:
            c = conn.cursor()
            c.execute(f"select item_id from items where title like '%{query}%'")
            rows = c.fetchall()[:n_items]
            return [row[0] for row in rows]