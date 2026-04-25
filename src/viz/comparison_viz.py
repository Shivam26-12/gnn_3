"""comparison_viz.py — Thesis figures for M5 DTNet GNN vs Baseline comparison."""

from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

BG = '#0a0e17'
LAYER_COLORS = {
    'state': '#e74c3c', 'store': '#4c7aff', 'department': '#9b59b6',
    'category': '#2ecc71', 'item': '#f39c12',
}
LAYER_ORDER = ['state', 'store', 'department', 'category', 'item']
ONE_HOT_START = 7
DATE = date.today().strftime('%Y%m%d')
SAVE_DIR_DEFAULT = Path('results/thesis_figures')

matplotlib.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': '#3a3f4e',
    'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
    'text.color': 'white', 'grid.color': '#1e2330', 'grid.alpha': 0.4,
    'legend.facecolor': '#0d1117', 'legend.edgecolor': '#3a3f4e',
})


def _savefig(fig, save_dir, stem):
    p = Path(save_dir) / f'dtnet_{stem}_{DATE}.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  [viz] {p.name}')
    return p


def _setup_ax(ax, title, xlabel='', ylabel='', grid=True):
    ax.set_facecolor(BG)
    ax.set_title(title, color='white', fontsize=11, pad=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    if grid: ax.grid(True, alpha=0.35)


def plot_networked_vs_isolated(eval_results, save_dir):
    """Bar chart: MSE, MAE, WRMSSE comparison."""
    gm, bm = eval_results['gnn_test'], eval_results['baseline_test']
    specs = [('MSE', gm['mse'], bm['mse']),
             ('MAE', gm['mae'], bm['mae']),
             ('WRMSSE', gm['wrmsse'], bm['wrmsse'])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
    fig.suptitle('M5 DTNet GNN vs Isolated Baseline', color='white', fontsize=14, y=1.02)

    for ax, (lbl, gv, bv) in zip(axes, specs):
        bars = ax.bar(['DTNetGNN', 'Baseline'], [gv, bv],
                      color=['#4c7aff', '#9b59b6'], width=0.5, zorder=3)
        for bar, v in zip(bars, [gv, bv]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.03,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10, color='white')
        _setup_ax(ax, lbl, ylabel=lbl, grid=False)

    plt.tight_layout()
    return _savefig(fig, save_dir, 'fig1_wrmsse_comparison')


def generate_all_figures(eval_results, save_dir=SAVE_DIR_DEFAULT, **kwargs):
    """Generate thesis figures."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f'[viz] Generating figures -> {Path(save_dir).resolve()}')
    out = {
        'wrmsse_comparison': plot_networked_vs_isolated(eval_results, save_dir),
    }
    print(f'[viz] Done - {len(out)} figures saved')
    return out
