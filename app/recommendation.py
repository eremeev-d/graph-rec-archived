import random

from data import items_data


def recommend_items(rec_query, n_items=10):
    random.seed(rec_query)
    results = [random.randint(0, len(items_data)) for _ in range(n_items)]
    return results