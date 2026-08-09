"""
models/layers.py

Core building blocks for NeuroMesh:
  - GraphConvLayer : a symmetric-normalized Graph Convolutional layer
                      (Kipf & Welling style: D^-1/2 A D^-1/2 X W).
  - NeuroMeshLayer : the topology controller (MLP -> GRU -> GCN -> Mask-MLP)
                      that predicts a dynamic, differentiable edge-activation
                      mask used to "rewire" a feature vector on the fly.

PROVENANCE NOTE
---------------
This is a from-scratch re-implementation based on the architecture *description*
in the NeuroMesh draft manuscript (Sec. 3.2.1) and its supplementary material
(Appendix A). The draft's own pseudocode has real bugs -- e.g. it feeds
concat(features, h_prev, failure_signal) into an nn.GRU whose declared
input_size only matches the feature vector, and its GCN example never defines
what "node features" means for a plain MLP layer (there are no natural graph
edges between scalar activations without such a construction). This
implementation fixes those specific issues (see comments below) but is
otherwise a straightforward, honest realization of the stated design.

No performance numbers, robustness guarantees, or benchmark results from the
draft are asserted anywhere in this file. Train and evaluate this yourself
before believing anything about how well it works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Symmetric-normalized GCN layer: H' = ReLU( D^-1/2 A D^-1/2 X W )

    Accepts a batch of node-feature matrices sharing one (or a per-sample)
    adjacency matrix.
    """

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.eps = eps

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, in_dim]
            adjacency:     [N, N]  or  [B, N, N]  (nonnegative; include self-loops)
        Returns:
            [B, N, out_dim]
        """
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)  # -> [1, N, N], broadcasts over batch

        degrees = adjacency.sum(dim=-1)                         # [B or 1, N]
        d_inv_sqrt = torch.pow(degrees.clamp(min=self.eps), -0.5)
        d_inv_sqrt = torch.diag_embed(d_inv_sqrt)               # [B or 1, N, N]

        a_norm = d_inv_sqrt @ adjacency @ d_inv_sqrt            # [B or 1, N, N]
        messages = torch.matmul(a_norm, node_features)          # broadcasts to [B, N, in_dim]
        return F.relu(self.linear(messages))


class NeuroMeshLayer(nn.Module):
    """
    Topology controller: predicts a [feature_dim x feature_dim] edge-activation
    mask from the current feature vector, a temporal (GRU) hidden state, and an
    optional binary failure signal, then applies that mask to a learnable
    adjacency to produce a "rewired" output vector.

    Design choices / fixes relative to the draft's pseudocode:
      * Uses nn.GRUCell (proper single-step recurrent update) instead of an
        nn.GRU fed a concatenated tensor of mismatched width.
      * Treats each of the `feature_dim` entries of x as a graph node with a
        scalar (1-dim) feature, so the GCN has a well-defined input.
      * Uses a straight-through Bernoulli estimator for stochastic masks during
        training (the draft names "stochastic hard sigmoid" but never specifies
        how gradients flow through a Bernoulli sample -- straight-through is the
        standard, differentiable choice).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128, stochastic: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.stochastic = stochastic

        self.feat_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.failure_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )

        # Temporal context across layers/timesteps.
        self.gru_cell = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)

        # in_dim=1 because each "node" here is a single scalar activation.
        self.gcn = GraphConvLayer(in_dim=1, out_dim=hidden_dim)

        self.mask_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, feature_dim * feature_dim),
        )

        # Learnable adjacency, initialized to identity (as in the draft) so the
        # network starts close to "no rewiring" and learns connections from there.
        self.A_learnable = nn.Parameter(torch.eye(feature_dim))

    def init_hidden(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None, failure_signal: torch.Tensor = None):
        """
        Args:
            x:              [B, feature_dim]
            h_prev:         [B, hidden_dim] or None (zero-initialized if None)
            failure_signal: [B, feature_dim] binary indicator, or None (all-zero if None)
        Returns:
            y:      [B, feature_dim]                       rewired output
            h_new:  [B, hidden_dim]                         updated GRU hidden state
            mask:   [B, feature_dim, feature_dim]           predicted edge mask (for losses/logging)
        """
        B = x.size(0)
        device = x.device
        if h_prev is None:
            h_prev = self.init_hidden(B, device)
        if failure_signal is None:
            failure_signal = torch.zeros_like(x)

        feat = self.feat_mlp(x)
        fail_emb = self.failure_encoder(failure_signal)
        gru_input = torch.cat([feat, fail_emb], dim=-1)
        h_new = self.gru_cell(gru_input, h_prev)

        node_features = x.unsqueeze(-1)                          # [B, feature_dim, 1]
        node_emb = self.gcn(node_features, self.A_learnable)     # [B, feature_dim, hidden_dim]
        node_agg = node_emb.mean(dim=1)                          # [B, hidden_dim]

        mask_input = torch.cat([node_agg, h_new], dim=-1)
        mask_logits = self.mask_mlp(mask_input).view(B, self.feature_dim, self.feature_dim)

        if self.training and self.stochastic:
            soft_mask = torch.sigmoid(mask_logits)
            hard_mask = torch.bernoulli(soft_mask)
            mask = soft_mask + (hard_mask - soft_mask).detach()  # straight-through estimator
        else:
            mask = torch.clamp(torch.sigmoid(mask_logits), 0.0, 1.0)

        effective_adj = mask * self.A_learnable.unsqueeze(0)     # [B, feature_dim, feature_dim]
        y = torch.bmm(effective_adj, x.unsqueeze(-1)).squeeze(-1)

        return y, h_new, mask
