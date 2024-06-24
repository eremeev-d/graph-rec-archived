import argparse
import os

import numpy as np
import pandas as pd
import faiss
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

from utils import normalize_embeddings, build_index


def prepare_als_embeddings(items_path, ratings_path, save_directory, rank, max_iter, reg_param):
    ### Compute ALS embeddings using Spark
    spark = SparkSession.builder.appName("ALS").getOrCreate()
    
    ratings = spark.read.csv(ratings_path, header=True, inferSchema=True)
    ratings = ratings.select(
        col("user_id").cast("int"),
        col("item_id").cast("int"),
        col("rating").cast("float"))

    als = ALS(
        rank=rank,
        maxIter=max_iter, 
        regParam=reg_param, 
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
    
    ### Normalize & save embeddings
    embeddings = normalize_embeddings(embeddings)
    np.save(os.path.join(save_directory, "embeddings.npy"), embeddings)
    
    ### Build index
    build_index(embeddings, os.path.join(save_directory, "index.faiss"))