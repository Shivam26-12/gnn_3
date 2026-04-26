"""evaluate.py — Evaluation of trained models on M5 test set.

Calculates MSE, MAE, and WRMSSE for DTNetGNN vs Baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Any

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

from src.gnn.model import DTNetGNN, IsolatedBaseline
from src.gnn.train import GNN_SAVE_PATH, BASELINE_SAVE_PATH, compute_wrmsse

@torch.no_grad()
def _collect_and_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        ea = getattr(batch, "edge_attr", None)
        preds.append(model(batch.x_seq, batch.x_static, batch.edge_index, ea).cpu())
        targets.append(batch.y.cpu())
        
    y_pred = torch.cat(preds)
    y_true = torch.cat(targets)
    
    mse = F.mse_loss(y_pred, y_true).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    wrmsse = compute_wrmsse(y_pred, y_true)
    
    return {"mse": mse, "mae": mae, "wrmsse": wrmsse}

def run_evaluation(test_loader: DataLoader, G=None, device: torch.device = None) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    sample = next(iter(test_loader))
    seq_channels = sample.x_seq.shape[2]
    static_channels = sample.x_static.shape[1]
    horizon = sample.y.shape[1]
    
    gnn = DTNetGNN(seq_channels=seq_channels, static_channels=static_channels, forecast_horizon=horizon).to(device)
    gnn.load_state_dict(torch.load(GNN_SAVE_PATH, map_location=device, weights_only=True))
    
    baseline = IsolatedBaseline(seq_channels=seq_channels, static_channels=static_channels, forecast_horizon=horizon).to(device)
    baseline.load_state_dict(torch.load(BASELINE_SAVE_PATH, map_location=device, weights_only=True))
    
    gnn_test = _collect_and_eval(gnn, test_loader, device)
    base_test = _collect_and_eval(baseline, test_loader, device)
    
    print(f"\n[evaluate] === Final Test Set Results ===")
    print(f"  {'Model':<20} | {'MSE':>8} | {'MAE':>8} | {'WRMSSE':>8}")
    print("-" * 55)
    print(f"  {'DTNetGNN':<20} | {gnn_test['mse']:8.4f} | {gnn_test['mae']:8.4f} | {gnn_test['wrmsse']:8.4f}")
    print(f"  {'IsolatedBaseline':<20} | {base_test['mse']:8.4f} | {base_test['mae']:8.4f} | {base_test['wrmsse']:8.4f}")
    
    return {
        "gnn_test": gnn_test,
        "baseline_test": base_test
    }
