"""topology.py — Orchestrate M5 30,490 Item-Store Graph Topology."""

from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.graph.topology_specs import NodeSpec, EdgeSpec

def infer_topology(sales_df: pd.DataFrame) -> Tuple[List[NodeSpec], List[EdgeSpec]]:
    """Infer the 30,490-node Item-Store graph from M5 sales data."""
    n = len(sales_df)
    item_ids = sales_df["item_id"].values
    store_ids = sales_df["store_id"].values
    dept_ids = sales_df["dept_id"].values
    state_ids = sales_df["state_id"].values
    cat_ids = sales_df["cat_id"].values

    nodes: List[NodeSpec] = []
    item_to_rows: Dict[str, List[int]] = {}
    store_dept_to_rows: Dict[str, List[int]] = {}

    print(f"[topology] Building {n} nodes...")
    for i in range(n):
        item_id = str(item_ids[i])
        store_id = str(store_ids[i])
        dept_id = str(dept_ids[i])
        state_id = str(state_ids[i])
        node_id = f"{item_id}_{store_id}"
        
        nodes.append(NodeSpec(
            node_id=node_id,
            layer="item_store",
            attrs={
                "item_id": item_id,
                "store_id": store_id,
                "dept_id": dept_id,
                "cat_id": str(cat_ids[i]),
                "state_id": state_id,
            }
        ))
        
        item_to_rows.setdefault(item_id, []).append(i)
        store_dept_to_rows.setdefault(f"{store_id}_{dept_id}", []).append(i)

    edges: List[EdgeSpec] = []
    print("[topology] Building cross-store & inter-state edges...")
    for item_id, rows in item_to_rows.items():
        if len(rows) <= 1: continue
        states = [str(state_ids[r]) for r in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                src = f"{item_ids[rows[i]]}_{store_ids[rows[i]]}"
                dst = f"{item_ids[rows[j]]}_{store_ids[rows[j]]}"
                if states[i] == states[j]:
                    # Same state: Geographic substitution (cross_store)
                    edges.append(EdgeSpec(src=src, dst=dst, edge_type="cross_store", attrs={"criticality_weight": 0.7}))
                    edges.append(EdgeSpec(src=dst, dst=src, edge_type="cross_store", attrs={"criticality_weight": 0.7}))
                else:
                    # Different state: Logistics balancing (inter_state)
                    edges.append(EdgeSpec(src=src, dst=dst, edge_type="inter_state", attrs={"criticality_weight": 0.2}))
                    edges.append(EdgeSpec(src=dst, dst=src, edge_type="inter_state", attrs={"criticality_weight": 0.2}))

    print("[topology] Building intra-dept edges...")
    rng = np.random.default_rng(42)
    K = 5
    for key, rows in store_dept_to_rows.items():
        if len(rows) <= 1: continue
        for r in rows:
            src = f"{item_ids[r]}_{store_ids[r]}"
            neighbors = rng.choice([x for x in rows if x != r], size=min(K, len(rows)-1), replace=False)
            for nb in neighbors:
                dst = f"{item_ids[nb]}_{store_ids[nb]}"
                edges.append(EdgeSpec(src=src, dst=dst, edge_type="intra_dept", attrs={"criticality_weight": 0.4}))

    print(f"[topology] Finished: {len(nodes)} nodes, {len(edges)} edges.")
    return nodes, edges
