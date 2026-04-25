"""builder_agents.py — Factory for ItemStoreAgent."""

from __future__ import annotations
from typing import Any, Dict

from src.agents.base_agent import DigitalTwinAgent
from src.agents.item_store_agent import ItemStoreAgent

def create_agent(node_id: str, layer: str, attrs: Dict[str, Any]) -> DigitalTwinAgent:
    """Instantiate an ItemStoreAgent for a given node."""
    agent = ItemStoreAgent(node_id=node_id, **{k: v for k, v in attrs.items() if k in ("mean_sales", "zero_rate", "price", "capacity", "throughput", "failure_prob")})
    agent.save_initial_state()
    return agent
