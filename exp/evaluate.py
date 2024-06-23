import argparse
import json

import pandas as pd
import numpy as np

from app.recommendations import RecommenderSystem


def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    return len(recommended_at_k & relevant_set) / k


def evaluate_recsys(
    metrics_savepath,
    val_ratings_path, 
    faiss_index_path, 
    embeddings_path,
    n_recommend_items=10,
):
    recsys = RecommenderSystem(
        faiss_index_path=faiss_index_path, 
        embeddings_path=embeddings_path)

    val_ratings = pd.read_csv(val_ratings_path)
    grouped_items = val_ratings.groupby("user_id")["item_id"].apply(list).reset_index()
    grouped_items = grouped_items["item_id"].tolist()


    metric_arrays = {
        "precision@1": [],
        "precision@3": [],
        "precision@10": []
    }

    for item_group in grouped_items:
        if len(item_group) == 1:
            continue

        for item in item_group:
            recommend_items = list(recsys.recommend_items(item, n_recommend_items))
            relevant_items = set(item_group) - {item}

            # TODO: first aggregate by user?
            metric_arrays["precision@1"].append(
                precision_at_k(recommend_items, relevant_items, k=1))
            metric_arrays["precision@3"].append(
                precision_at_k(recommend_items, relevant_items, k=3))
            metric_arrays["precision@10"].append(
                precision_at_k(recommend_items, relevant_items, k=10))

    metrics = dict()
    for metric, array in metric_arrays.items():
        metrics[metric] = np.mean(array)

    with open(metrics_savepath, "w") as f:
        json.dump(metrics, f)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommender system.")
    
    parser.add_argument("--metrics_savepath", type=str, required=True, help="Path to save evaluation metrics")
    parser.add_argument("--val_ratings_path", type=str, required=True, help="Path to the validation ratings file.")
    parser.add_argument("--faiss_index_path", type=str, required=True, help="Path to the FAISS index file.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the embeddings file.")
    parser.add_argument("--n_recommend_items", type=int, default=10, help="Number of items to recommend (default: 10).")
    
    args = parser.parse_args()

    metrics = evaluate_recsys(
        metrics_savepath=args.metrics_savepath,
        val_ratings_path=args.val_ratings_path,
        faiss_index_path=args.faiss_index_path,
        embeddings_path=args.embeddings_path,
        n_recommend_items=args.n_recommend_items
    )