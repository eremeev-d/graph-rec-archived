from data import items_data


def search_items(search_term, n_items=10):
    results = items_data[items_data["title"].str.contains(search_term, case=False)].index
    return results[:n_items]