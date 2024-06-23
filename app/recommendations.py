import itertools
import os

import numpy as np
import faiss


class RecommenderSystem:
    def __init__(self, faiss_index_path, embeddings_path):
        self._index = faiss.read_index(faiss_index_path)
        self._embeddings = np.load(embeddings_path)

    def recommend_items(self, query, n_items=10):
        query_embedding = self._embeddings[query, :]

        _, results = self._index.search(query_embedding[None, :], k=n_items+1)
        results = filter(lambda item: item != query, results[0])
        return itertools.islice(results, n_items)
