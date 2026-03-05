import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import issparse

def permutation(feature):
    """
    Randomly shuffles the rows of the feature matrix.
    """
    np.random.seed(41)
    indices = np.random.permutation(feature.shape[0])
    return feature[indices]

def build_neighbor_graph(adata, n_neighbors=3):
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
    adata.obsm['adj'] = np.maximum(interaction, interaction.T)


def build_neighbor_graph_KNN(adata, n_neighbors=3):
    """Constructs a spatial adjacency matrix using Scikit-learn's KNN for efficiency."""
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    # Use KD-tree or Ball-tree to find neighbors efficiently
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    rows = np.repeat(np.arange(n_spot), n_neighbors)
    cols = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[rows, cols] = 1
    adata.obsm['graph_neigh'] = interaction

    # Symmetrize the adjacency matrix
    adata.obsm['adj'] = np.maximum(interaction, interaction.T)

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    if isinstance(adata.X, sp.spmatrix):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X
    pca = PCA(n_components=20, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    adata.obsm['X_pca'] = X_pca

def extract_feature(adata):
    """Subsets highly variable genes and prepares dense feature matrices."""
    hvg_adata = adata[:, adata.var['highly_variable']]
    feat = hvg_adata.X.toarray() if issparse(hvg_adata.X) else hvg_adata.X
    feat_a = permutation(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a

def prepare_csl_targets(adata):
    """Extracts features from the expression matrix and generates an augmented version."""
    n_spot = adata.n_obs
    label_CSL = np.zeros((n_spot, 2))
    label_CSL[:, 0] = 1
    adata.obsm['label_CSL'] = label_CSL

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def _symmetric_normalize(adj):
    """
    Perform symmetric normalization
    """
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def norm_adj(adj):
    adj = sp.coo_matrix(adj)
    return _symmetric_normalize(adj).toarray()

def norm_add_eye(adj):
    return norm_adj(adj) + np.eye(adj.shape[0])

def eye_norm_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    adj_normalized = _symmetric_normalize(adj_).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)

