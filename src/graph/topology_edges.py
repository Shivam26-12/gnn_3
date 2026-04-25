"""topology_edges.py — Build EdgeSpec objects for the M5 retail graph.

Edge types
----------
- distribution_flow : state -> store (material distribution)
- operational       : store -> department (store contains these depts)
- dept_group        : department -> category (dept belongs to category)
- category_item     : category -> item (category contains items)
                      OR department -> item (dept contains items)
- cross_store       : store <-> store (same-state or high-correlation)
"""

from __future__ import annotations
from typing import Dict, List, Set
import numpy as np
from src.data.entity_mapping import M5EntityMappings
from src.graph.topology_specs import EdgeSpec

np.random.seed(42)


def build_state_to_store_edges(em: M5EntityMappings) -> List[EdgeSpec]:
    """State -> Store edges (distribution flow from state-level to stores)."""
    edges: List[EdgeSpec] = []
    for state_id, store_ids in em.state_to_stores.items():
        for store_id in store_ids:
            edges.append(EdgeSpec(
                src=state_id,
                dst=store_id,
                edge_type="distribution_flow",
                attrs={
                    "criticality_weight": 0.9,
                    "flow_capacity": 1.0,
                    "shared_items_count": 0,
                },
            ))
    return edges


def build_store_to_dept_edges(em: M5EntityMappings) -> List[EdgeSpec]:
    """Store -> Department edges (operational: store hosts these departments)."""
    edges: List[EdgeSpec] = []
    for store_id, dept_ids in em.store_to_depts.items():
        n_depts = len(dept_ids)
        for dept_id in dept_ids:
            n_items = len(em.dept_to_items.get(dept_id, []))
            # Criticality proportional to item count (larger depts = more critical)
            criticality = min(1.0, n_items / 800.0)
            edges.append(EdgeSpec(
                src=store_id,
                dst=dept_id,
                edge_type="operational",
                attrs={
                    "criticality_weight": round(criticality, 3),
                    "flow_capacity": round(1.0 / n_depts, 3),
                    "shared_items_count": n_items,
                },
            ))
    return edges


def build_dept_to_item_edges(em: M5EntityMappings) -> List[EdgeSpec]:
    """Department -> Item edges (department contains these items)."""
    edges: List[EdgeSpec] = []
    for dept_id, item_ids in em.dept_to_items.items():
        n_items = len(item_ids)
        for item_id in item_ids:
            edges.append(EdgeSpec(
                src=dept_id,
                dst=item_id,
                edge_type="category_item",
                attrs={
                    "criticality_weight": round(1.0 / max(n_items, 1), 4),
                    "flow_capacity": 1.0,
                    "shared_items_count": 0,
                },
            ))
    return edges


def build_dept_to_category_edges(em: M5EntityMappings) -> List[EdgeSpec]:
    """Department -> Category edges (aggregation)."""
    edges: List[EdgeSpec] = []
    for dept_id, cat_id in em.dept_to_cat.items():
        edges.append(EdgeSpec(
            src=dept_id,
            dst=cat_id,
            edge_type="dept_group",
            attrs={
                "criticality_weight": 0.7,
                "flow_capacity": 1.0,
                "shared_items_count": len(em.dept_to_items.get(dept_id, [])),
            },
        ))
    return edges


def build_cross_store_edges(em: M5EntityMappings) -> List[EdgeSpec]:
    """Cross-store edges: connect stores in the same state (bidirectional).

    Since all 3,049 items appear in all 10 stores, we use same-state
    proximity as the edge criterion rather than connecting all pairs
    (which would create 45 edges vs 12 same-state edges).
    """
    edges: List[EdgeSpec] = []
    for state_id, store_ids in em.state_to_stores.items():
        n_stores = len(store_ids)
        total_items = len(em.item_ids)
        for i in range(n_stores):
            for j in range(i + 1, n_stores):
                # Bidirectional
                for src, dst in [(store_ids[i], store_ids[j]),
                                 (store_ids[j], store_ids[i])]:
                    edges.append(EdgeSpec(
                        src=src,
                        dst=dst,
                        edge_type="cross_store",
                        attrs={
                            "criticality_weight": 0.6,
                            "flow_capacity": 0.8,
                            "shared_items_count": total_items,
                        },
                    ))
    return edges
