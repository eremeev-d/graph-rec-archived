import argparse
import os

import pandas as pd
import numpy as np


def process_raw_data(input_directory, save_directory, positive_rating_threshold = 4.0):
    os.makedirs(save_directory, exist_ok=True)

    items = pd.read_csv(os.path.join(input_directory, "movies.csv"))
    items["item_id"] = items.index
    items["initial_id"] = items["movieId"]
    items.drop("movieId", axis=1, inplace=True)
    items.to_csv(os.path.join(save_directory, "items.csv"), index=False)

    from_initial_id = dict(zip(items["initial_id"], items["item_id"]))
    ratings = pd.read_csv(os.path.join(input_directory, "ratings.csv"))
    ratings["user_id"] = ratings["userId"] - 1
    ratings["item_id"] = ratings["movieId"].map(from_initial_id)
    ratings.drop(["userId", "movieId"], axis=1, inplace=True)
    assert (ratings["user_id"].unique() == range(0, max(ratings["user_id"])+1)).all()
    ratings = ratings[ratings["rating"] >= positive_rating_threshold]
    ratings.to_csv(os.path.join(save_directory, "ratings.csv"), index=False)


def create_train_val_split(ratings_path, train_savepath, val_savepath, seed=42):
    np.random.seed(seed)
    
    ratings = pd.read_csv(ratings_path)
    shuffled_indices = np.random.permutation(ratings.index)
    train_size = int(len(ratings) * 0.8)
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_data = ratings.loc[train_indices]
    val_data = ratings.loc[val_indices]

    train_data.to_csv(train_savepath, index=False)
    val_data.to_csv(val_savepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data.")
    parser.add_argument("--input_directory", required=True, type=str, help="Directory containing the raw data.")
    parser.add_argument("--save_directory", required=True, type=str, help="Directory where processed data will be saved.")
    parser.add_argument("--create_train_val_split", action="store_true", help="Flag to indicate whether to create a train-validation split.")
    args = parser.parse_args()

    print("Processing raw data...")
    process_raw_data(args.input_directory, args.save_directory)
    if args.create_train_val_split:
        create_train_val_split(
            os.path.join(args.save_directory, "ratings.csv"),
            os.path.join(args.save_directory, "train_ratings.csv"),
            os.path.join(args.save_directory, "val_ratings.csv")
        )
    print("The raw data has been successfully processed.")