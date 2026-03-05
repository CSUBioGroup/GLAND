import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GATv2Conv
from .preprocess import fix_seed


class Discriminator(nn.Module):
    """Bilinear discriminator used for contrastive learning."""

    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, context, positive_samples, negative_samples, s_bias1=None, s_bias2=None):
        context_expanded = context.expand_as(positive_samples)

        sc_1 = self.f_k(positive_samples, context_expanded)
        sc_2 = self.f_k(negative_samples, context_expanded)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    """Compute weighted average of local embeddings to get a global representation."""

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, embeddings, mask=None):
        summed_vectors = torch.mm(mask, embeddings)
        neighbor_counts = torch.sum(mask, 1)
        neighbor_counts_expanded = neighbor_counts.expand((summed_vectors.shape[1], neighbor_counts.shape[0])).T
        global_embedding = summed_vectors / neighbor_counts_expanded

        return F.normalize(global_embedding, p=2, dim=1)

class Encoder(nn.Module):
    """Graph encoder combining GATv2 and Transformer encoders."""

    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.leaky_relu, num_heads=8):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        # GATv2Conv layer
        self.gat1 = GATv2Conv(in_channels=in_features, out_channels=out_features // num_heads, heads=num_heads,
                              dropout=dropout)
        # Linear projection before Transformer
        self.linear_proj = nn.Linear(in_features, out_features)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=out_features, nhead=num_heads, dropout=dropout,
                                                            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        # Output layers and contrastive components
        self.alpha_param = nn.Parameter(torch.tensor([0.8]))
        self.beta_param = nn.Parameter(torch.tensor([0.8]))
        self.alpha_param1 = nn.Parameter(torch.tensor([0.8]))
        self.beta_param1 = nn.Parameter(torch.tensor([0.8]))

        self.linear_out = nn.Linear(out_features, in_features)
        self.discc = Discriminator(self.out_features)
        self.disc = Discriminator(self.out_features)

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

    def forward(self, feat, feat_a, edge_index, adj_new):
        # Graph Attention and Transformer encoding
        gat_features = F.dropout(feat, self.dropout, self.training)
        gat_features = self.gat1(gat_features, edge_index)

        projected_input_features = self.linear_proj(feat)
        transformer_features = self.transformer_encoder(projected_input_features.unsqueeze(0)).squeeze(0)

        alpha_weight = torch.sigmoid(self.alpha_param)
        combined_features = alpha_weight * gat_features + (1 - alpha_weight) * transformer_features

        embedding = self.act(combined_features)
        decoded_features = self.linear_out(combined_features)

        # Augmented branch
        gat_features_aug = F.dropout(feat_a, self.dropout, self.training)
        gat_features_aug = self.gat1(gat_features_aug, edge_index)

        projected_input_features_aug = self.linear_proj(feat_a)
        transformer_features_aug = self.transformer_encoder(projected_input_features_aug.unsqueeze(0)).squeeze(0)

        beta_weight = torch.sigmoid(self.alpha_param)
        combined_features_aug = beta_weight * gat_features_aug + (1 - beta_weight) * transformer_features_aug

        embedding_aug = self.act(combined_features_aug)

        global_representation = self.read(embedding, adj_new)
        global_representation = self.sigm(global_representation)

        global_representation_aug = self.read(embedding_aug, adj_new)
        global_representation_aug = self.sigm(global_representation_aug)

        logits = self.disc(global_representation, embedding, embedding_aug)
        logits_aug = self.disc(global_representation_aug, embedding_aug, embedding)

        return embedding, decoded_features, logits, logits_aug

