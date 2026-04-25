"""_attention_timeline_viz.py — Attention analysis for M5 retail GNN."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from src.viz.comparison_viz import BG, LAYER_COLORS, LAYER_ORDER, ONE_HOT_START, _savefig, _setup_ax


def plot_attention_heatmap(attention, runs, save_dir):
    """Top-K edge attention bar chart."""
    if not attention or 'top_k_edges' not in attention:
        return None
    top_k = attention['top_k_edges']
    if not top_k:
        return None

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    labels = [f"{e.get('src_name', e['src'])} -> {e.get('dst_name', e['dst'])}" for e in top_k]
    vals = [e['attention'] for e in top_k]
    ax.barh(range(len(labels)), vals, height=0.7, color='#f39c12', edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    _setup_ax(ax, 'Top Attention Edges', xlabel='Attention Weight')
    return _savefig(fig, save_dir, 'fig4_attention')


def plot_propagation_timeline(history, runs, save_dir):
    """Placeholder for propagation timeline."""
    return None
