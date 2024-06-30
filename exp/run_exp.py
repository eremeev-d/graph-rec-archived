import argparse
import os

from als import prepare_als_embeddings
from deepwalk import prepare_deepwalk_embeddings
from sbert import prepare_sbert_embeddings
from gnn import prepare_gnn_embeddings
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
    elif args.method == "GNN":
        prepare_gnn_embeddings(
            # Paths
            items_path=items_path,
            ratings_path=train_ratings_path,
            text_embeddings_path=args.text_embeddings_path,
            deepwalk_embeddings_path=args.deepwalk_embeddings_path,
            save_directory=args.save_directory,
            # Learning hyperparameters
            temperature=args.temperature, 
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            # Model hyperparameters
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            aggregator_type=args.aggregator_type,
            skip_connection=args.skip_connection,
            bidirectional=args.bidirectional,
            num_traversals=args.num_traversals, 
            termination_prob=args.termination_prob, 
            num_random_walks=args.num_random_walks, 
            num_neighbor=args.num_neighbor,
            # Misc
            device=args.device,
            wandb_name=args.wandb_name,
            use_wandb=args.use_wandb,
    )
    else:
        raise ValueError("Invalid method of creating embeddings.")
    print("Embeddings have been successfully prepared.")

    ### Evaluate RecSys
    evaluate_recsys(
        metrics_savepath=os.path.join(args.save_directory, "val_metrics.json"),
        val_ratings_path=val_ratings_path,
        faiss_index_path=os.path.join(args.save_directory, "index.faiss"),
        embeddings_path=os.path.join(args.save_directory, "embeddings.npy"),
        n_recommend_items=args.n_recommend_items
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare embeddings and evaluate the quality of the recommender system that uses these embeddings.")

    # TODO: some args have different default values for different methods (e.g. batch_size for DeepWalk and GNN), do something with that.
    ### General args
    parser.add_argument("--method", required=True, type=str, choices=["ALS", "DeepWalk", "SBERT", "GNN"], help="Method of creating embeddings.")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to the directory with items and train/val ratings files.")
    parser.add_argument("--save_directory", required=True, type=str, help="Directory where embeddings, FAISS index and metrics will be saved.")
    parser.add_argument("--n_recommend_items", type=int, default=10, help="Number of items to recommend (default: 10).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (default: cpu)")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for WandB run (optional)")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb", help="Disable WandB logging")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate (default: 0.01)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train (default: 4)")

    ### ALS args
    parser.add_argument("--rank", type=int, default=32, help="Rank parameter for ALS. Defaults to 32.")
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of iterations of ALS. Defaults to 10.")
    parser.add_argument("--reg_param", type=float, default=0.1, help="Regularization parameter of ALS. Defaults to 0.1.")

    ### DeepWalk args
    parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension (default: 32)")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for DeepWalk model (default: 4)")

    # TODO: add args for SBERT
    
    ### GNN
    parser.add_argument("--text_embeddings_path", type=str, required=True, help="Path to the file with text embeddings in .npy format.")
    parser.add_argument("--deepwalk_embeddings_path", type=str, required=True, help="Path to the file with deepwalk embeddings in .npy format.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature in NT-Xent Loss")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in GNN")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Hidden dimension in GNN")
    parser.add_argument("--aggregator_type", type=str, default="gcn", choices=["mean", "gcn", "pool", "lstm"], help="Aggregation type in SAGE Convolution")
    parser.add_argument("--no_skip_connection", action="store_false", dest="skip_connection", help="Disable skip connections")
    parser.add_argument("--bidirectional", action="store_true", dest="bidirectional", help="Use direct and reversed edges in convolutions.")
    parser.add_argument("--num_traversals", type=int, default=4)  # TODO: add description 
    parser.add_argument("--termination_prob", type=float, default=0.5)  # TODO: add description 
    parser.add_argument("--num_random_walks", type=int, default=200)  # TODO: add description 
    parser.add_argument("--num_neighbor", type=int, default=3)  # TODO: add description 

    run_experiment(args=parser.parse_args())