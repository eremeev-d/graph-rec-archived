import argparse
import os

import numpy as np
import pandas as pd
import dgl
import torch
import wandb
from tqdm.auto import tqdm

from utils import (
    prepare_graphs, extract_item_embeddings, 
    normalize_embeddings, build_index
)


def prepare_deepwalk_embeddings(
        items_path,
        ratings_path, 
        save_directory,
        emb_dim, 
        window_size, 
        batch_size, 
        lr, 
        num_epochs, 
        device, 
        wandb_name, 
        use_wandb
):
    ### Prepare graph
    bipartite_graph, graph = prepare_graphs(items_path, ratings_path)

    ### Run deepwalk
    if use_wandb:
        wandb.init(project="graph-recs-deepwalk", name=wandb_name)

    model = dgl.nn.DeepWalk(graph, emb_dim=emb_dim, window_size=window_size)
    model = model.to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.arange(graph.num_nodes()), 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=model.sample)
    
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for batch_walk in tqdm(dataloader):
            loss = model(batch_walk)
            if use_wandb:
                wandb.log({"loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
    if use_wandb:
        wandb.finish()
    
    node_embeddings = model.node_embed.weight.detach().to(device)

    ### Extract & save item embeddings
    item_embeddings = extract_item_embeddings(node_embeddings, bipartite_graph, graph)
    item_embeddings = normalize_embeddings(item_embeddings)
    build_index(item_embeddings, os.path.join(save_directory, "index.faiss"))
    np.save(os.path.join(save_directory, "embeddings.npy"), item_embeddings)