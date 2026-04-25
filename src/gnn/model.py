"""model.py — Deep GAT + Baseline for M5 28-day demand forecasting.

DTNetGNN: 3-layer GAT with residual connections, A100-optimized.
IsolatedBaseline: 4-layer MLP (no graph structure).
Both output (N, 28) per-item-store 28-day sales forecasts.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

np.random.seed(42)
torch.manual_seed(42)

IN_CHANNELS: int = 25
HIDDEN: int = 256
EDGE_DIM: int = 3
HEADS_1: int = 8
HEADS_2: int = 4
HEADS_3: int = 4
DROPOUT: float = 0.15
HORIZON: int = 28


class DTNetGNN(nn.Module):
    """3-layer GAT with residual connections for demand forecasting.

    Architecture:
        GATConv(25 -> 256, heads=8)  -> 2048-dim + residual projection
        GATConv(2048 -> 256, heads=4) -> 1024-dim + residual
        GATConv(1024 -> 256, heads=4) -> 1024-dim + residual
        Linear(1024 -> 512) -> ReLU -> Dropout
        Linear(512 -> 256) -> ReLU
        Linear(256 -> 28) -> Softplus (non-negative sales)
    """
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN, edge_dim=EDGE_DIM,
                 heads_1=HEADS_1, heads_2=HEADS_2, heads_3=HEADS_3,
                 dropout=DROPOUT, forecast_horizon=HORIZON):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, hidden, heads=heads_1, edge_dim=edge_dim, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden * heads_1)
        self.res1 = nn.Linear(in_channels, hidden * heads_1)

        dim1 = hidden * heads_1
        self.conv2 = GATConv(dim1, hidden, heads=heads_2, edge_dim=edge_dim, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden * heads_2)
        self.res2 = nn.Linear(dim1, hidden * heads_2)

        dim2 = hidden * heads_2
        self.conv3 = GATConv(dim2, hidden, heads=heads_3, edge_dim=edge_dim, concat=True)
        self.bn3 = nn.BatchNorm1d(hidden * heads_3)
        self.res3 = nn.Linear(dim2, hidden * heads_3)

        dim3 = hidden * heads_3
        self.fc1 = nn.Linear(dim3, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.fc3 = nn.Linear(hidden, forecast_horizon)
        self.softplus = nn.Softplus()

    def forward(self, x, edge_index, edge_attr=None):
        # Layer 1
        res = self.res1(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x + res)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        res = self.res2(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x + res)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        res = self.res3(x)
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.elu(x + res)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Forecast head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softplus(x)  # (N, 28) non-negative


class IsolatedBaseline(nn.Module):
    """4-layer MLP baseline (no graph structure)."""
    def __init__(self, in_channels=IN_CHANNELS, hidden=HIDDEN, forecast_horizon=HORIZON):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, forecast_horizon),
            nn.Softplus(),
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.net(x)  # (N, 28)
