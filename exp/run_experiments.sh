input_directory=$1
save_directory=$2

python exp/process_raw_data.py \
    --input_directory "$input_directory" \
    --save_directory "$save_directory" \
    --create_train_val_split

python exp/als.py \
    --items_path "$save_directory/items.csv" \
    --ratings_path "$save_directory/train_ratings.csv" \
    --save_directory "$save_directory"

PYTHONPATH=. python exp/evaluate.py \
    --metrics_savepath "$save_directory/als_metrics.json" \
    --val_ratings_path "$save_directory/val_ratings.csv" \
    --faiss_index_path "$save_directory/index.faiss" \
    --embeddings_path "$save_directory/embeddings.npy"