import numpy as np
import pandas as pd
import dgl
import torch
import faiss


def normalize_embeddings(embeddings):
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    nonzero_embeddings = embeddings_norm > 0.0 
    embeddings[nonzero_embeddings] /= embeddings_norm[nonzero_embeddings, None]
    return embeddings


def build_index(embeddings, save_path, n_neighbors=32):
    index = faiss.IndexHNSWFlat(embeddings.shape[-1], 32)
    index.add(embeddings)
    faiss.write_index(index, save_path)


def prepare_graphs(items_path, ratings_path):
    items = pd.read_csv(items_path)
    ratings = pd.read_csv(ratings_path)

    n_users = np.max(ratings["user_id"].unique()) + 1
    item_ids = torch.tensor(sorted(items["item_id"].unique()))

    edges = torch.tensor(ratings["user_id"]), torch.tensor(ratings["item_id"])
    reverse_edges = (edges[1], edges[0])

    bipartite_graph = dgl.heterograph(
        data_dict={
            ("User", "Edge", "Item"): edges,
            ("Item", "Edge", "User"): reverse_edges
        },
        num_nodes_dict={
            "User": n_users,
            "Item": len(item_ids)
        }
    )
    graph = dgl.to_homogeneous(bipartite_graph)
    graph = dgl.add_self_loop(graph)
    return bipartite_graph, graph


def extract_item_embeddings(node_embeddings, bipartite_graph, graph):
    item_ntype = bipartite_graph.ntypes.index("Item")
    item_mask = graph.ndata[dgl.NTYPE] == item_ntype
    item_embeddings = node_embeddings[item_mask]
    original_ids = graph.ndata[dgl.NID][item_mask]
    item_embeddings = item_embeddings[torch.argsort(original_ids)]
    return item_embeddings.numpy()