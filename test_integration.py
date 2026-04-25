"""Integration test: full pipeline from M5 data to GNN forward pass."""

import sys
import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print("  DTNet M5 INTEGRATION TEST")
print("=" * 60)

# Phase 1: Load data
print("\n[1] Loading M5 data...")
from src.data.loader import load_m5
dfs = load_m5()

# Phase 1b: Entity mappings
print("\n[2] Building entity mappings...")
from src.data.entity_mapping import build_entity_mappings
em = build_entity_mappings(dfs["sales"])
print(f"    States={len(em.state_ids)}, Stores={len(em.store_ids)}, "
      f"Depts={len(em.dept_ids)}, Cats={len(em.cat_ids)}, Items={len(em.item_ids)}")

# Phase 2: Graph construction
print("\n[3] Building graph...")
from src.graph.topology import infer_topology
from src.graph.builder import build_graph
nodes, edges = infer_topology(em)
G = build_graph(nodes, edges)
print(f"    SUCCESS: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Phase 3: Vulnerability analysis
print("\n[4] Running vulnerability analysis...")
from src.graph.metrics import vulnerability_analysis, network_stats
stats = network_stats(G)
print(f"    Network stats: {stats}")
top_vuln = vulnerability_analysis(G, top_k=5)
print(f"    Top-5 vulnerable nodes:")
for nid, score, layer in top_vuln:
    print(f"      {nid:<20} score={score:.4f} layer={layer}")

# Phase 4: Build dataset (small test)
print("\n[5] Building dataset (sliding windows)...")
from src.gnn.dataset import build_dataloaders
train_loader, val_loader, test_loader = build_dataloaders(
    dfs["sales"], G, batch_size=4, stride=28,  # larger stride for faster test
)

# Phase 5: GNN forward pass
print("\n[6] Testing GNN forward pass...")
from src.gnn.model import DTNetGNN, IsolatedBaseline

sample = next(iter(train_loader))
in_channels = sample.x.shape[1]
forecast_h = sample.y.shape[1]
print(f"    Input features: {in_channels}")
print(f"    Target shape: {sample.y.shape}")
print(f"    Edge index shape: {sample.edge_index.shape}")
if sample.edge_attr is not None:
    print(f"    Edge attr shape: {sample.edge_attr.shape}")

device = torch.device("cpu")

gnn = DTNetGNN(in_channels=in_channels, forecast_horizon=forecast_h).to(device)
baseline = IsolatedBaseline(in_channels=in_channels, forecast_horizon=forecast_h).to(device)

sample = sample.to(device)
ea = getattr(sample, "edge_attr", None)

gnn_out = gnn(sample.x, sample.edge_index, ea)
base_out = baseline(sample.x, sample.edge_index, ea)

print(f"    GNN output shape: {gnn_out.shape}")
print(f"    Baseline output shape: {base_out.shape}")
print(f"    GNN output range: [{gnn_out.min().item():.4f}, {gnn_out.max().item():.4f}]")
print(f"    Target range: [{sample.y.min().item():.4f}, {sample.y.max().item():.4f}]")

# Phase 6: WRMSSE computation
print("\n[7] Testing WRMSSE computation...")
from src.gnn.train import compute_wrmsse
wrmsse = compute_wrmsse(gnn_out, sample.y)
print(f"    WRMSSE (untrained): {wrmsse:.4f}")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED!")
print("=" * 60)
