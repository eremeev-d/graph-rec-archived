import pandas as pd


class SearchSystem:
    def __init__(self, items_path):
        self._items = pd.read_csv(items_path)

    def search_items(self, query, n_items=10):
        results = self._items[self._items["title"].str.contains(query, case=False)].index
        return results[:n_items]