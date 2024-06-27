import argparse
import os

from als import prepare_als_embeddings
from deepwalk import prepare_deepwalk_embeddings
from sbert import prepare_sbert_embeddings
from evaluate import evaluate_recsys


def run_experiment(args):
    items_path = os.path.join(args.data_directory, "items.csv")
    train_ratings_path = os.path.join(args.data_directory, "train_ratings.csv")
    val_ratings_path = os.path.join(args.data_directory, "val_ratings.csv")

    ### Prepare embeddngs
    print("Preparing embeddings...")
    os.makedirs(args.save_directory, exist_ok=True)
    if args.method == "ALS":
        prepare_als_embeddings(
            items_path=items_path,
            ratings_path=train_ratings_path,
            save_directory=args.save_directory,
            rank=args.rank,
            max_iter=args.max_iter,
            reg_param=args.reg_param
        )
    elif args.method == "DeepWalk":
        prepare_deepwalk_embeddings(
            items_path=items_path,
            ratings_path=train_ratings_path,
            save_directory=args.save_directory,
            emb_dim=args.emb_dim,
            window_size=args.window_size,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            device=args.device,
            wandb_name=args.wandb_name,
            use_wandb=args.use_wandb,
        )
    elif args.method == "SBERT":
        prepare_sbert_embeddings(
            items_path=items_path,
            save_directory=args.save_directory
        )
    else:
        raise ValueError("Invalid method of creating embeddings.")
    print("Embeddings have been successfully prepared.")

    ### Evaluate RecSys
    evaluate_recsys(
        metrics_savepath=os.path.join(args.save_directory, "metrics.json"),
        val_ratings_path=val_ratings_path,
        faiss_index_path=os.path.join(args.save_directory, "index.faiss"),
        embeddings_path=os.path.join(args.save_directory, "embeddings.npy"),
        n_recommend_items=args.n_recommend_items
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare embeddings and evaluate the quality of the recommender system that uses these embeddings.")

    ### General args
    parser.add_argument("--method", required=True, type=str, choices=["ALS", "DeepWalk", "SBERT"], help="Method of creating embeddings.")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to the directory with items and train/val ratings files.")
    parser.add_argument("--save_directory", required=True, type=str, help="Directory where embeddings, FAISS index and metrics will be saved.")
    parser.add_argument("--n_recommend_items", type=int, default=10, help="Number of items to recommend (default: 10).")

    ### ALS args
    parser.add_argument("--rank", type=int, default=32, help="Rank parameter for ALS. Defaults to 32.")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of iterations of ALS. Defaults to 10.")
    parser.add_argument("--reg_param", type=float, default=0.1, help="Regularization parameter of ALS. Defaults to 0.1.")

    ### DeepWalk args
    parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension (default: 32)")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for DeepWalk model (default: 4)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training DeepWalk (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for DeepWalk (default: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train DeepWalk (default: 4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run DeepWalk on (default: cpu)")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for WandB DeepWalk run (optional)")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable WandB logging for DeepWalk")

    run_experiment(args=parser.parse_args())