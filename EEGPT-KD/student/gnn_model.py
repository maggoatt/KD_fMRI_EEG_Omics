"""ConnectivityGCN student model for knowledge distillation.

Adapted from teammate's connectivityGCN (model/gnn_pipeline.ipynb).
3-layer EdgeConv GNN with BatchNorm and global mean pooling.
Modified to return both logits and intermediate features for KD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import EdgeConv, global_mean_pool


class ConnectivityGCN(nn.Module):
    """EdgeConv-based GNN for brain graph classification.

    Args:
        input_dim: node feature dimension (1=connectivity, 295=connectivity+genes)
        hidden_channels: hidden layer width (default 64)
        num_classes: output classes (default 2: alert/drowsy)
    """

    def __init__(self, input_dim=1, hidden_channels=64, num_classes=2):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.mlp1 = Sequential(
            Linear(2 * input_dim, hidden_channels), ReLU()
        )
        self.mlp2 = Sequential(
            Linear(2 * hidden_channels, hidden_channels), ReLU()
        )
        self.mlp3 = Sequential(
            Linear(2 * hidden_channels, hidden_channels), ReLU()
        )

        self.conv1 = EdgeConv(self.mlp1, aggr='max')
        self.conv2 = EdgeConv(self.mlp2, aggr='max')
        self.conv3 = EdgeConv(self.mlp3, aggr='max')

        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, data):
        """Forward pass returning logits and features.

        Args:
            data: PyG Data or Batch object with x, edge_index, batch

        Returns:
            dict with:
                logits: (B, num_classes) raw classification scores
                features: (B, hidden_channels) pooled graph features
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        # Pool across nodes -> graph-level features
        features = global_mean_pool(x, batch)

        # Classification head (no softmax - raw logits for KD)
        logits = self.classifier(features)

        return {"logits": logits, "features": features}
