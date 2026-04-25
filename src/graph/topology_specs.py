"""topology_specs.py — Data containers for M5 retail graph topology.

NodeSpec and EdgeSpec are pure-data structures, no heavy objects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

# Canonical layer order for one-hot encoding
LAYER_ORDER = ["state", "store", "department", "category", "item"]


@dataclass
class NodeSpec:
    """Lightweight specification for a single graph node.

    Attributes
    ----------
    node_id : str
        Unique identifier (e.g. 'CA_1', 'FOODS_3', 'FOODS_3_001').
    layer : str
        One of LAYER_ORDER values.
    attrs : dict
        Arbitrary attributes (demand features, agent init kwargs).
    """
    node_id: str
    layer: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Lightweight specification for a single directed edge.

    Attributes
    ----------
    src : str
        Source node ID.
    dst : str
        Destination node ID.
    edge_type : str
        One of: 'distribution_flow', 'operational', 'category_group',
        'dept_group', 'cross_store_shared'.
    attrs : dict
        Attributes: criticality_weight, flow_capacity, shared_items_count, etc.
    """
    src: str
    dst: str
    edge_type: str
    attrs: Dict[str, Any] = field(default_factory=dict)
