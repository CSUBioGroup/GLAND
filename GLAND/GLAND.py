import numpy as np
import torch
from torch import nn

from .preprocess import (prepare_csl_targets, build_neighbor_graph,
                         build_neighbor_graph_KNN, fix_seed, extract_feature,
                         preprocess, norm_add_eye, eye_norm_sparse)
from .spLOF import filter_points_with_lof
from .trainer import run_training

class GLAND():
    ADJ_UPDATE_EPOCH = 200
    def __init__(self,
                 adata,
                 device=torch.device('cuda:0'),
                 learning_rate=0.001,
                 weight_decay=0.00,
                 epochs=600,
                 in_dim=3000,
                 out_dim=256,
                 random_seed=41,
                 datatype='10X',
                 dataset='',
                 n_clusters=7,
                 lofk=13,
                 spLOF_threshold=1.8
                 ):

        """
                Initializes the GLAND model.

                Args:
                    adata (AnnData): The annotated data matrix containing gene expression and spatial info.
                    device (torch.device): The device to run the model on ('cuda:0' or 'cpu').
                    learning_rate (float): The step size for the optimizer during training.
                    weight_decay (float): L2 penalty for regularization.
                    epochs (int): Total number of complete passes through the training dataset.
                    in_dim (int): Dimension of the input features(hvg).
                    out_dim (int): Dimension of the output embedding.
                    random_seed (int): Seed used for random number generators to ensure reproducibility.
                    datatype (str): The format or source of the data ('10X', 'ST').
                    dataset (str): Name of the dataset.
                    n_clusters (int): The target number of clusters for downstream clustering analysis.
                    lofk (int): The number of nearest neighbors (k) used for spLOF calculation.
                    spLOF_threshold (float): The threshold value for filtering outliers using spatial LOF.
                """
        fix_seed(random_seed)
        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.datatype = datatype
        self.dataset = dataset
        self.n_clusters = n_clusters
        fix_seed(self.random_seed)
        self.lofk = lofk
        self.spLOF_threshold = spLOF_threshold

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                build_neighbor_graph_KNN(self.adata)
            else:
                build_neighbor_graph(self.adata)

        if 'label_CSL' not in adata.obsm.keys():
            prepare_csl_targets(self.adata)

        if 'feat' not in adata.obsm.keys():
            extract_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(
            self.device)
        self.in_dim = self.features.shape[1]
        self.out_dim = out_dim
        self.linear_proj_concat = nn.Linear(out_dim * 2, out_dim)

        filter_method = 'threshold' if 0 < self.spLOF_threshold < 2 else 'percent'
        self.adj = filter_points_with_lof(
            adata=self.adata,
            features=self.adata.obsm['X_pca'],
            spatial_coords=self.adata.obsm['spatial'],
            adj=self.adj,
            ground_truth_labels=None,
            k=self.lofk,
            filter_method=filter_method,
            spLOF_threshold=self.spLOF_threshold
        )

        if self.datatype in ['Stereo', 'Slide']:
            # using sparse
            self.adj = eye_norm_sparse(self.adj).to(self.device)
        else:
            # standard version
            self.adj = norm_add_eye(self.adj)
            self.adj = torch.FloatTensor(self.adj).to(self.device)

    def train(self):

        return run_training(self)