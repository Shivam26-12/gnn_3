"""data_generator.py — Generate training data from M5 sales via sliding windows.

This replaces the simulation-based data generator. Instead of simulating
disruptions, we use the real M5 sales data directly.

Public API: ``generate_training_data(sales_df, G, ...)``
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

np.random.seed(42)
torch.manual_seed(42)

FORECAST_HORIZON: int = 28
FEATURE_WINDOW: int = 28
STRIDE: int = 7
LAYER_ORDER = ["state", "store", "department", "category", "item"]
OUTPUT_PATH_DEFAULT = Path("data/processed/m5_training_data.pkl")


def generate_training_data(
    sales_df: pd.DataFrame,
    G: nx.DiGraph,
    output_path: Optional[Path] = None,
    forecast_horizon: int = FORECAST_HORIZON,
    feature_window: int = FEATURE_WINDOW,
    stride: int = STRIDE,
) -> List[Dict[str, Any]]:
    """Generate sliding-window training data from M5 sales.

    Each sample contains node features from a historical window and
    28-day sales targets for evaluation.

    Returns list of dicts, each with:
      - node_features: (N, F) feature array
      - targets: (N, 28) sales targets
      - node_order: list of node IDs
      - window_start/end: temporal indices
    """
    if output_path is None:
        output_path = OUTPUT_PATH_DEFAULT
    output_path = Path(output_path)

    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_matrix = sales_df[day_cols].values
    n_days = len(day_cols)
    node_order = list(G.nodes())

    # Map items to row indices
    item_to_rows: Dict[str, List[int]] = {}
    for idx, item_id in enumerate(sales_df["item_id"].values):
        item_to_rows.setdefault(str(item_id), []).append(idx)

    print(f"[data_gen] Generating windows: {n_days} days, "
          f"window={feature_window}, horizon={forecast_horizon}, stride={stride}")

    dataset: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    pos = feature_window
    while pos + forecast_horizon <= n_days:
        feat_start = pos - feature_window
        feat_end = pos
        tgt_start = pos
        tgt_end = pos + forecast_horizon

        # Simple features per node
        n_nodes = len(node_order)
        features = np.zeros((n_nodes, len(LAYER_ORDER) + 7), dtype=np.float32)

        for i, nid in enumerate(node_order):
            layer = G.nodes[nid].get("layer", "item")
            li = LAYER_ORDER.index(layer) if layer in LAYER_ORDER else 4
            features[i, 7 + li] = 1.0  # one-hot

            if layer == "item":
                rows = item_to_rows.get(nid, [])
                if rows:
                    item_sales = sales_matrix[rows, feat_start:feat_end].sum(axis=0).astype(float)
                    features[i, 0] = item_sales.mean()
                    features[i, 1] = item_sales.std() if len(item_sales) > 1 else 0.0
                    features[i, 2] = features[i, 1] / (features[i, 0] + 1e-8)
                    features[i, 3] = (item_sales == 0).mean()
                    features[i, 4] = item_sales.sum()
                    # trend
                    if len(item_sales) >= 2:
                        x = np.arange(len(item_sales), dtype=float)
                        xm = x.mean()
                        denom = np.sum((x - xm) ** 2)
                        features[i, 5] = np.sum((x - xm) * (item_sales - item_sales.mean())) / (denom + 1e-8) if denom > 0 else 0
                    features[i, 6] = len(rows)  # n_stores

        # Targets
        targets = np.zeros((n_nodes, forecast_horizon), dtype=np.float32)
        for i, nid in enumerate(node_order):
            layer = G.nodes[nid].get("layer", "item")
            if layer == "item":
                rows = item_to_rows.get(nid, [])
                if rows:
                    targets[i] = sales_matrix[rows, tgt_start:tgt_end].sum(axis=0).astype(float)

        dataset.append({
            "node_features": features,
            "targets": targets,
            "node_order": node_order,
            "window_start": feat_start,
            "window_end": feat_end,
        })

        pos += stride

    elapsed = time.perf_counter() - t_start
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as fh:
        pickle.dump(dataset, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[data_gen] Generated {len(dataset)} windows in {elapsed:.1f}s")
    print(f"[data_gen] Saved to {output_path}")

    return dataset
