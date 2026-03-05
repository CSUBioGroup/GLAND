import faiss
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from torch import nn
from torch_sparse import SparseTensor
from tqdm import tqdm
from .preprocess import (fix_seed, permutation, preprocess, norm_add_eye, eye_norm_sparse)

def build_graph(adata, n_neighbors=2):
    """Constructs a spatial adjacency matrix using a full distance matrix."""
    position = adata.obsm['spatial']

    # Calculate pairwise Euclidean distances
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    indices = np.argsort(distance_matrix, axis=1)[:, 1:n_neighbors + 1]
    interaction = np.zeros([n_spot, n_spot])
    rows = np.repeat(np.arange(n_spot), n_neighbors)
    interaction[rows, indices.flatten()] = 1
    adata.obsm['graph_neigh'] = interaction

    # Symmetrize the adjacency matrix
    return np.maximum(interaction, interaction.T)

def update_adj_with_feature_neighbors(gland_obj, emb):
    """
    Dynamically refines the adjacency matrix by calculating k-nearest neighbors
    in the feature embedding space and intersecting them with the spatial space.
    """
    # Convert latent embeddings to NumPy for FAISS-based neighbor search
    emb_np = emb.cpu().detach().numpy()

    # Build a FAISS index to find the top k nearest neighbors in the feature space
    index = faiss.IndexFlatL2(emb_np.shape[1])
    index.add(emb_np)
    distances, indices = index.search(emb_np, 15)
    adj_new = torch.zeros_like(gland_obj.knn_refine, dtype=torch.bool)
    row_indices = torch.arange(indices.shape[0]).unsqueeze(1).repeat(1, indices.shape[1]).flatten()
    col_indices = torch.tensor(indices).flatten()

    # Populate the adjacency matrix and ensure it is symmetric
    adj_new[row_indices, col_indices] = True
    adj_new[col_indices, row_indices] = True

    # Perform element-wise AND to keep edges that exist in both feature and spatial space
    adj_bool = gland_obj.knn_refine.bool()
    adj_new = (adj_new & adj_bool).to(gland_obj.device)
    adj_new = adj_new.float()

    return adj_new

def run_training(gland_obj):
    # Prepare sparse adjacency matrices for efficient graph convolutions
    edge_index = torch.nonzero(gland_obj.adj.to_dense(), as_tuple=False).t().contiguous()
    num_nodes = gland_obj.adj.size(0)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    adj_t = adj.t()

    from .model import Encoder

    # Initialize model architecture, loss function, and optimizer
    gland_obj.model = Encoder(gland_obj.in_dim, gland_obj.out_dim, gland_obj.graph_neigh).to(gland_obj.device)
    gland_obj.loss_CSL = nn.BCEWithLogitsLoss()
    gland_obj.optimizer = torch.optim.Adam(gland_obj.model.parameters(), gland_obj.learning_rate,
                                           weight_decay=gland_obj.weight_decay)

    # Construct and preprocess the spatial k-nearest neighbor graph for refinement
    gland_obj.knn_refine = build_graph(gland_obj.adata, 5)
    gland_obj.knn_refine = norm_add_eye(gland_obj.knn_refine)
    gland_obj.knn_refine = torch.FloatTensor(gland_obj.knn_refine).to(gland_obj.device)

    # Precompute smoothed neighbor features for reconstruction
    degree = gland_obj.adj.to_dense().sum(dim=1, keepdim=True)
    neighbor_features = torch.spmm(gland_obj.adj.to_dense(), gland_obj.features) / degree
    adj_new = gland_obj.graph_neigh

    print('Start training')
    for epoch in tqdm(range(gland_obj.epochs)):
        gland_obj.model.train()
        # Data augmentation
        gland_obj.features_a = permutation(gland_obj.features)

        embb, emb, ret, ret_a = gland_obj.model(gland_obj.features, gland_obj.features_a, adj_t, adj_new)

        # Calculate total loss: Feature reconstruction (SmoothL1) + Contrastive loss (BCE)
        loss_feat = F.smooth_l1_loss(neighbor_features, emb)
        gland_obj.loss_sl_1 = gland_obj.loss_CSL(ret, gland_obj.label_CSL)
        gland_obj.loss_sl_2 = gland_obj.loss_CSL(ret_a, gland_obj.label_CSL)
        loss = 5 * loss_feat + (gland_obj.loss_sl_1 + gland_obj.loss_sl_2)

        gland_obj.optimizer.zero_grad()
        loss.backward()
        gland_obj.optimizer.step()

        # Dynamically refine the graph structure
        if epoch == gland_obj.ADJ_UPDATE_EPOCH:
            print(f"Updating adjacency matrix at epoch {epoch}...")
            adj_new = update_adj_with_feature_neighbors(gland_obj, emb)

    print("Training finished")

    # Extract final latent embeddings and save to the AnnData object
    with torch.no_grad():
        gland_obj.model.eval()
        gland_obj.emb_rec = gland_obj.model(gland_obj.features, gland_obj.features_a, adj_t, adj_new)[1]
        gland_obj.emb_rec = gland_obj.emb_rec.detach().cpu().numpy()
        gland_obj.adata.obsm['emb'] = gland_obj.emb_rec
        return gland_obj.adata