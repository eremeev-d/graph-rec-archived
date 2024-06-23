import argparse
import os

import numpy as np
import pandas as pd
import faiss
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col


def prepare_als_embeddings(items_path, ratings_path, save_directory, rank, maxIter, regParam):
    ### Compute ALS embeddings using Spark
    spark = SparkSession.builder.appName("ALS").getOrCreate()
    
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    ratings = ratings.select(
        col("user_id").cast("int"),
        col("item_id").cast("int"),
        col("rating").cast("float"))

    als = ALS(
        rank=rank,
        maxIter=maxIter, 
        regParam=regParam, 
        userCol="user_id", 
        itemCol="item_id", 
        ratingCol="rating", 
        coldStartStrategy="drop"
    )

    model = als.fit(ratings)
    embeddings = model.itemFactors.toPandas()
    embeddings = embeddings.rename(columns={"id": "item_id"})
    spark.stop()

    ### Set missing embeddings to zeros
    items = pd.read_csv(items_path)["item_id"]
    embeddings = pd.merge(items, embeddings, on="item_id", how="left")
    print(f"Number of missing embeddings: {embeddings['features'].isna().sum()} / {items.shape[0]}")
    zeros_list = [0.0 for _ in range(rank)]
    embeddings["features"] = embeddings["features"].apply(lambda x: zeros_list if isinstance(x, float) and np.isnan(x) else x)

    ### Convert to numpy
    embeddings = embeddings.sort_values("item_id")["features"].tolist()
    embeddings = np.array(embeddings)
    assert embeddings.shape[0] == pd.read_csv(items_path).shape[0]
    
    ### Normalize emeddings
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    nonzero_embeddings = embeddings_norm > 0.0 
    embeddings[nonzero_embeddings] /= embeddings_norm[nonzero_embeddings, None]
    np.save(os.path.join(save_directory, "embeddings.npy"), embeddings)
    
    ### Build index
    index = faiss.IndexHNSWFlat(embeddings.shape[-1], 32)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(save_directory, "index.faiss"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ALS embeddings.")
    parser.add_argument("--items_path", required=True, type=str, help="Path to the file with items data.")
    parser.add_argument("--ratings_path", required=True, type=str, help="Path to the ratings file.")
    parser.add_argument("--save_directory", required=True, type=str, help="Directory where embeddings will be saved.")
    parser.add_argument("--rank", type=int, default=32, help="Rank parameter for ALS. Defaults to 32.")
    parser.add_argument("--maxIter", type=int, default=10, help="Maximum number of iterations. Defaults to 10.")
    parser.add_argument("--regParam", type=float, default=0.1, help="Regularization parameter. Defaults to 0.1.")
    args = parser.parse_args()

    print("Preparing ALS embeddings...")
    prepare_als_embeddings(args.items_path, args.ratings_path, args.save_directory, args.rank, args.maxIter, args.regParam)
    print("ALS embeddings have been successfully prepared.")