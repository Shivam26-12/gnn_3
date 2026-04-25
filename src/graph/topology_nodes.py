"""topology_nodes.py — Build NodeSpec objects for each M5 hierarchy layer."""

from __future__ import annotations
from typing import List
from src.data.entity_mapping import M5EntityMappings
from src.graph.topology_specs import NodeSpec


def build_state_nodes(em: M5EntityMappings) -> List[NodeSpec]:
    """Create one NodeSpec per state (CA, TX, WI)."""
    return [
        NodeSpec(
            node_id=state_id,
            layer="state",
            attrs={"n_stores": len(em.state_to_stores.get(state_id, []))},
        )
        for state_id in em.state_ids
    ]


def build_store_nodes(em: M5EntityMappings) -> List[NodeSpec]:
    """Create one NodeSpec per store (CA_1, ..., WI_3)."""
    return [
        NodeSpec(
            node_id=store_id,
            layer="store",
            attrs={
                "state_id": em.store_to_state.get(store_id, ""),
                "n_depts": len(em.store_to_depts.get(store_id, [])),
            },
        )
        for store_id in em.store_ids
    ]


def build_department_nodes(em: M5EntityMappings) -> List[NodeSpec]:
    """Create one NodeSpec per department (FOODS_1, ..., HOUSEHOLD_2)."""
    return [
        NodeSpec(
            node_id=dept_id,
            layer="department",
            attrs={
                "cat_id": em.dept_to_cat.get(dept_id, ""),
                "n_items": len(em.dept_to_items.get(dept_id, [])),
            },
        )
        for dept_id in em.dept_ids
    ]


def build_category_nodes(em: M5EntityMappings) -> List[NodeSpec]:
    """Create one NodeSpec per category (FOODS, HOBBIES, HOUSEHOLD)."""
    return [
        NodeSpec(
            node_id=cat_id,
            layer="category",
            attrs={"n_depts": len(em.cat_to_depts.get(cat_id, []))},
        )
        for cat_id in em.cat_ids
    ]


def build_item_nodes(em: M5EntityMappings) -> List[NodeSpec]:
    """Create one NodeSpec per unique item_id."""
    return [
        NodeSpec(
            node_id=item_id,
            layer="item",
            attrs={"n_stores": len(em.item_to_stores.get(item_id, []))},
        )
        for item_id in em.item_ids
    ]
