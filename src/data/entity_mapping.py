"""
entity_mapping.py — Extract M5 retail hierarchy relationships.

Responsibility: build all lookup dictionaries that graph construction
needs to wire up nodes and edges.

Mappings produced
-----------------
- state_to_stores    : state_id  -> sorted list of store_ids
- store_to_depts     : store_id  -> sorted list of dept_ids
- dept_to_cat        : dept_id   -> cat_id
- cat_to_depts       : cat_id    -> sorted list of dept_ids
- store_to_items     : store_id  -> sorted list of item_ids
- dept_to_items      : dept_id   -> sorted list of item_ids
- item_to_stores     : item_id   -> sorted list of store_ids
- item_store_pairs   : list of (item_id, store_id) tuples

All mappings are bundled in the ``M5EntityMappings`` dataclass.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

np.random.seed(42)

# Column name constants
COL_STATE: str = "state_id"
COL_STORE: str = "store_id"
COL_DEPT: str = "dept_id"
COL_CAT: str = "cat_id"
COL_ITEM: str = "item_id"


@dataclass
class M5EntityMappings:
    """All entity relationship mappings for the M5 dataset.

    Attributes
    ----------
    state_ids : list[str]
        All unique state IDs, sorted. E.g. ['CA', 'TX', 'WI'].
    store_ids : list[str]
        All unique store IDs, sorted. E.g. ['CA_1', ..., 'WI_3'].
    dept_ids : list[str]
        All unique department IDs, sorted.
    cat_ids : list[str]
        All unique category IDs, sorted.
    item_ids : list[str]
        All unique item IDs, sorted.
    state_to_stores : dict[str, list[str]]
        Maps each state_id to sorted list of store_ids.
    store_to_depts : dict[str, list[str]]
        Maps each store_id to sorted list of dept_ids available in that store.
    dept_to_cat : dict[str, str]
        Maps each dept_id to its parent cat_id.
    cat_to_depts : dict[str, list[str]]
        Maps each cat_id to sorted list of dept_ids.
    dept_to_items : dict[str, list[str]]
        Maps each dept_id to sorted list of item_ids.
    item_to_stores : dict[str, list[str]]
        Maps each item_id to sorted list of store_ids it appears in.
    item_store_pairs : list[tuple[str, str]]
        All (item_id, store_id) combinations, sorted.
    store_to_state : dict[str, str]
        Maps each store_id to its parent state_id.
    """

    state_ids: List[str] = field(default_factory=list)
    store_ids: List[str] = field(default_factory=list)
    dept_ids: List[str] = field(default_factory=list)
    cat_ids: List[str] = field(default_factory=list)
    item_ids: List[str] = field(default_factory=list)

    state_to_stores: Dict[str, List[str]] = field(default_factory=dict)
    store_to_depts: Dict[str, List[str]] = field(default_factory=dict)
    dept_to_cat: Dict[str, str] = field(default_factory=dict)
    cat_to_depts: Dict[str, List[str]] = field(default_factory=dict)
    dept_to_items: Dict[str, List[str]] = field(default_factory=dict)
    item_to_stores: Dict[str, List[str]] = field(default_factory=dict)
    item_store_pairs: List[Tuple[str, str]] = field(default_factory=list)
    store_to_state: Dict[str, str] = field(default_factory=dict)


def _group_sorted(df: pd.DataFrame, by: str, collect: str) -> Dict[str, List[str]]:
    """Group df by ``by`` and collect unique sorted values of ``collect``."""
    return (
        df.groupby(by)[collect]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )


def build_entity_mappings(sales_df: pd.DataFrame) -> M5EntityMappings:
    """Extract all M5 hierarchy relationships from the sales DataFrame.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales DataFrame containing columns: id, item_id, dept_id, cat_id,
        store_id, state_id, plus day columns.

    Returns
    -------
    M5EntityMappings
        Fully populated dataclass with all mapping dictionaries.
    """
    required = [COL_STATE, COL_STORE, COL_DEPT, COL_CAT, COL_ITEM]
    missing = [c for c in required if c not in sales_df.columns]
    if missing:
        raise ValueError(f"[entity_mapping] Required columns not found: {missing}")

    cols = sales_df[required].copy().astype(str)

    # Unique entity lists
    state_ids = sorted(cols[COL_STATE].unique().tolist())
    store_ids = sorted(cols[COL_STORE].unique().tolist())
    dept_ids = sorted(cols[COL_DEPT].unique().tolist())
    cat_ids = sorted(cols[COL_CAT].unique().tolist())
    item_ids = sorted(cols[COL_ITEM].unique().tolist())

    # Directional mappings
    state_to_stores = _group_sorted(cols, COL_STATE, COL_STORE)
    store_to_depts = _group_sorted(cols, COL_STORE, COL_DEPT)
    cat_to_depts = _group_sorted(cols, COL_CAT, COL_DEPT)
    dept_to_items = _group_sorted(cols, COL_DEPT, COL_ITEM)
    item_to_stores = _group_sorted(cols, COL_ITEM, COL_STORE)

    # Dept -> Cat (many-to-one)
    dept_to_cat = (
        cols.drop_duplicates(subset=[COL_DEPT])
        .set_index(COL_DEPT)[COL_CAT]
        .to_dict()
    )

    # Store -> State (many-to-one)
    store_to_state = (
        cols.drop_duplicates(subset=[COL_STORE])
        .set_index(COL_STORE)[COL_STATE]
        .to_dict()
    )

    # All item-store pairs
    item_store_pairs = sorted(
        zip(cols[COL_ITEM].tolist(), cols[COL_STORE].tolist())
    )
    # Deduplicate
    item_store_pairs = sorted(set(item_store_pairs))

    return M5EntityMappings(
        state_ids=state_ids,
        store_ids=store_ids,
        dept_ids=dept_ids,
        cat_ids=cat_ids,
        item_ids=item_ids,
        state_to_stores=state_to_stores,
        store_to_depts=store_to_depts,
        dept_to_cat=dept_to_cat,
        cat_to_depts=cat_to_depts,
        dept_to_items=dept_to_items,
        item_to_stores=item_to_stores,
        item_store_pairs=item_store_pairs,
        store_to_state=store_to_state,
    )


def print_entity_summary(em: M5EntityMappings) -> None:
    """Print a structured summary of the M5 entity mappings.

    Parameters
    ----------
    em : M5EntityMappings
        Populated entity mappings as returned by ``build_entity_mappings``.
    """
    sep = "=" * 55

    print(f"\n{sep}")
    print("  M5 ENTITY STRUCTURE SUMMARY")
    print(sep)
    print(f"  {'Unique states':<35}: {len(em.state_ids):>6,}")
    print(f"  {'Unique stores':<35}: {len(em.store_ids):>6,}")
    print(f"  {'Unique departments':<35}: {len(em.dept_ids):>6,}")
    print(f"  {'Unique categories':<35}: {len(em.cat_ids):>6,}")
    print(f"  {'Unique items':<35}: {len(em.item_ids):>6,}")
    print(f"  {'Total item-store pairs':<35}: {len(em.item_store_pairs):>6,}")

    print(f"\n  Stores per state:")
    for state, stores in em.state_to_stores.items():
        print(f"    {state:<10}: {len(stores)} stores -> {stores}")

    print(f"\n  Departments per category:")
    for cat, depts in em.cat_to_depts.items():
        print(f"    {cat:<15}: {len(depts)} depts -> {depts}")

    print(f"\n  Items per department:")
    for dept, items in em.dept_to_items.items():
        print(f"    {dept:<15}: {len(items):>5,} items")

    print(sep + "\n")


def main() -> None:
    """Run entity mapping standalone from the CLI."""
    from src.data.loader import load_m5

    dfs = load_m5()
    em = build_entity_mappings(dfs["sales"])
    print_entity_summary(em)


if __name__ == "__main__":
    main()
