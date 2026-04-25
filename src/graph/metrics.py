"""metrics.py — Network stats for M5 Item-Store graph."""

from __future__ import annotations
from typing import Any, Dict
import networkx as nx

from src.graph.metrics_vulnerability import vulnerability_analysis

def network_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """Compute basic network statistics rapidly for large graphs."""
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_weakly_connected": nx.is_weakly_connected(G),
        "n_weakly_components": nx.number_weakly_connected_components(G),
        # Skip clustering on 30k nodes as it is O(V^3)
        "avg_clustering": "skipped for performance on 30k nodes",
    }
