import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

from utils import normalize_embeddings, build_index


def prepare_sbert_embeddings(
        items_path, 
        save_directory, 
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32
):
    items = pd.read_csv(items_path).sort_values("item_id")
    sentences = items["description"].values
    model = SentenceTransformer(model_name)
    embeddings = []
    for start_index in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[start_index:start_index+batch_size]
            embeddings.extend(model.encode(batch))
    embeddings = normalize_embeddings(np.array(embeddings))
    np.save(os.path.join(save_directory, "embeddings.npy"), embeddings)
    build_index(embeddings, os.path.join(save_directory, "index.faiss"))