"""scenarios.py — Disruption scenarios for M5 item-store graph."""

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict

np.random.seed(42)

def single_item_stockout(G: nx.DiGraph, node_id: str, severity: float = 0.9) -> Dict[str, float]:
    """Simulate a stockout for a specific item at a specific store."""
    if node_id not in G.nodes:
        raise KeyError(f"Node '{node_id}' not found.")
    return {node_id: float(np.clip(severity, 0.0, 1.0))}

def store_wide_disruption(G: nx.DiGraph, store_id: str, severity: float = 0.8) -> Dict[str, float]:
    """Simulate complete store closure (e.g. natural disaster affecting all items in CA_1)."""
    disruptions = {}
    for nid, data in G.nodes(data=True):
        if data.get("store_id") == store_id or nid.endswith(f"_{store_id}"):
            disruptions[nid] = float(np.clip(severity, 0.0, 1.0))
    return disruptions

def category_recall(G: nx.DiGraph, cat_id: str, state_id: str = None, severity: float = 0.9) -> Dict[str, float]:
    """Simulate a category-wide supply shortage or recall."""
    disruptions = {}
    for nid, data in G.nodes(data=True):
        is_cat = data.get("cat_id") == cat_id or cat_id in nid
        is_state = state_id is None or data.get("state_id") == state_id or nid.endswith(f"_{state_id}")
        if is_cat and is_state:
            disruptions[nid] = float(np.clip(severity, 0.0, 1.0))
    return disruptions

def random_stockouts(G: nx.DiGraph, fraction: float = 0.01, severity: float = 0.8) -> Dict[str, float]:
    """Random isolated stockouts for training diversity."""
    rng = np.random.default_rng(42)
    nodes = list(G.nodes)
    n = max(1, int(len(nodes) * fraction))
    selected = rng.choice(nodes, size=n, replace=False)
    return {nid: float(np.clip(severity, 0.0, 1.0)) for nid in selected}
