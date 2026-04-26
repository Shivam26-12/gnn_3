"""
run_pipeline.py — Complete DTNet M5 pipeline: data -> graph -> train -> evaluate.

Usage:
    python run_pipeline.py              # Full pipeline (preprocess + train + evaluate)
    python run_pipeline.py --step 1     # Step 1 only: Load data + build 30,490-node graph
    python run_pipeline.py --step 2     # Step 2 only: Build dataset (sliding windows)
    python run_pipeline.py --step 3     # Step 3 only: Train GNN + baseline
    python run_pipeline.py --step 4     # Step 4 only: Evaluate on test set
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

def step1_load_and_build_graph():
    """Step 1: Load M5 data, construct the 30k Item-Store graph."""
    print("\n" + "=" * 60)
    print("  STEP 1: Load Data & Build Graph")
    print("=" * 60)

    from src.data.loader import load_m5
    from src.graph.topology import infer_topology
    from src.graph.builder import build_graph
    from src.graph.metrics import network_stats
    from src.graph.metrics_vulnerability import vulnerability_analysis

    dfs = load_m5()
    nodes, edges = infer_topology(dfs["sales"])
    G = build_graph(nodes, edges)

    stats = network_stats(G)
    print(f"\n[step1] Network statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    top_vuln = vulnerability_analysis(G, top_k=5)
    print(f"\n[step1] Top-5 vulnerable item-store nodes:")
    for nid, score, layer in top_vuln:
        print(f"  {nid:<25} score={score:.4f}  layer={layer}")

    return dfs, G

def step2_build_dataset(dfs, G=None):
    """Step 2: Build sliding-window DataLoaders from M5 sales data."""
    print("\n" + "=" * 60)
    print("  STEP 2: Build Dataset (Sliding Windows)")
    print("=" * 60)

    from src.gnn.dataset import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        sales_df=dfs["sales"],
        calendar_df=dfs.get("calendar"),
        prices_df=dfs.get("prices"),
        G=G,
        batch_size=4,
        forecast_horizon=28,
        feature_window=28,
        stride=7, # stride=7 generates ~270 sliding windows
    )

    sample = next(iter(train_loader))
    print(f"\n[step2] Sample batch:")
    print(f"  x_seq shape  : {sample.x_seq.shape}")
    print(f"  x_static     : {sample.x_static.shape}")
    print(f"  y shape      : {sample.y.shape}")
    print(f"  y_hist shape : {sample.y_hist.shape}")
    print(f"  edge_index   : {sample.edge_index.shape}")
    return train_loader, val_loader, test_loader

def step3_train(train_loader, val_loader):
    """Step 3: Train Deep GAT and Baseline."""
    print("\n" + "=" * 60)
    print("  STEP 3: Train Models")
    print("=" * 60)

    from src.gnn.train import run_training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[step3] Device: {device}")
    
    gnn_hist, base_hist = run_training(train_loader, val_loader, device)
    
    try:
        from src.gnn.plot import plot_training_history
        plot_training_history(gnn_hist, base_hist)
    except Exception as e:
        print(f"[step3] Plotting failed (maybe missing matplotlib?): {e}")
        
    return gnn_hist, base_hist

def step4_evaluate(test_loader, G=None):
    """Step 4: Evaluate on the test set."""
    print("\n" + "=" * 60)
    print("  STEP 4: Evaluate on Test Set")
    print("=" * 60)

    from src.gnn.evaluate import run_evaluation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_evaluation(test_loader, G=G, device=device)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    t_start = time.perf_counter()
    dfs, G, train_loader, val_loader, test_loader = None, None, None, None, None

    if args.step == 0 or args.step == 1:
        dfs, G = step1_load_and_build_graph()
    
    if args.step == 0 or args.step == 2:
        if dfs is None:
            from src.data.loader import load_m5
            dfs = load_m5()
        train_loader, val_loader, test_loader = step2_build_dataset(dfs, G)
        
    if args.step == 0 or args.step == 3:
        if train_loader is None:
            from src.data.loader import load_m5
            from src.gnn.dataset import build_dataloaders
            dfs = load_m5()
            train_loader, val_loader, test_loader = build_dataloaders(
                dfs["sales"], G, dfs["calendar"], dfs["prices"], batch_size=4, stride=7)
        step3_train(train_loader, val_loader)
        
    if args.step == 0 or args.step == 4:
        if test_loader is None:
            from src.data.loader import load_m5
            from src.gnn.dataset import build_dataloaders
            dfs = load_m5()
            _, _, test_loader = build_dataloaders(
                dfs["sales"], G, dfs["calendar"], dfs["prices"], batch_size=4, stride=7)
        step4_evaluate(test_loader, G)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE - {time.perf_counter() - t_start:.1f}s")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
