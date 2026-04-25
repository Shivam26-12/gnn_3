"""builder.py — Assemble the final NetworkX DiGraph for M5 retail data.

Public API: ``build_graph(nodes, edges) -> nx.DiGraph``
"""

from __future__ import annotations
from typing import List

import networkx as nx
import numpy as np

from src.graph.topology_specs import NodeSpec, EdgeSpec
from src.graph.builder_agents import create_agent

np.random.seed(42)


def build_graph(
    nodes: List[NodeSpec],
    edges: List[EdgeSpec],
) -> nx.DiGraph:
    """Build the M5 retail supply-chain graph from topology specs.

    Each node gets:
      - 'layer': str — hierarchy layer
      - 'twin': DigitalTwinAgent — the digital twin instance

    Each edge gets:
      - 'edge_type': str
      - 'criticality_weight': float
      - 'flow_capacity': float
      - 'shared_items_count': int

    Parameters
    ----------
    nodes : list[NodeSpec]
        All node specifications.
    edges : list[EdgeSpec]
        All edge specifications.

    Returns
    -------
    nx.DiGraph
        Fully constructed graph with agents attached.
    """
    G = nx.DiGraph()

    # ── Add nodes ──────────────────────────────────────────────────────────
    for spec in nodes:
        agent = create_agent(spec.node_id, spec.layer, spec.attrs)
        G.add_node(
            spec.node_id,
            layer=spec.layer,
            twin=agent,
        )

    # ── Add edges ──────────────────────────────────────────────────────────
    for spec in edges:
        if spec.src not in G.nodes or spec.dst not in G.nodes:
            continue  # skip dangling edges

        G.add_edge(
            spec.src,
            spec.dst,
            edge_type=spec.edge_type,
            criticality_weight=spec.attrs.get("criticality_weight", 0.5),
            flow_capacity=spec.attrs.get("flow_capacity", 1.0),
            shared_items_count=spec.attrs.get("shared_items_count", 0),
        )

    print(f"[builder] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[builder] Layers: { {l: sum(1 for _, d in G.nodes(data=True) if d['layer'] == l) for l in set(nx.get_node_attributes(G, 'layer').values())} }")

    return G
