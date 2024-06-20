import random

import numpy as np

from data import items_data, get_embeddings, get_faiss_index


def recommend_items(query, n_items=10):
    # TODO: check that we dont spend too much time at this
    embeddings = get_embeddings()
    query_embedding = embeddings[query, :]
    
    index = get_faiss_index()
    _, results = index.search(query_embedding[None, :], k=n_items)
    return results[0]