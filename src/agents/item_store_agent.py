"""item_store_agent.py — M5 Item-Store Agent for granular simulation."""

from __future__ import annotations
from src.agents.base_agent import DigitalTwinAgent

class ItemStoreAgent(DigitalTwinAgent):
    """Represents a specific item in a specific store (e.g., HOBBIES_1_001_CA_1)."""
    def __init__(
        self,
        node_id: str,
        mean_sales: float = 0.0,
        zero_rate: float = 0.0,
        price: float = 0.0,
        capacity: float = 100.0,
        throughput: float = 1.0,
        failure_prob: float = 0.01,
    ) -> None:
        super().__init__(node_id, capacity, throughput, failure_prob)
        self.mean_sales = mean_sales
        self.zero_rate = zero_rate
        self.price = price

    def compute_health_score(self) -> float:
        """Health decreases if stockout probability (zero_rate) is high."""
        # Base health derived from capacity and throughput
        base_health = (self.capacity * 0.5) + (self.throughput * 0.5)
        stockout_penalty = min(self.zero_rate, 0.5)
        return max(base_health - stockout_penalty, 0.0)

    def _apply_specific_disruption(self, severity: float, current_step: int) -> None:
        """Specific disruption impact for an item-store (reduces capacity/throughput)."""
        self.capacity *= (1.0 - (severity * 0.5))
        self.throughput *= (1.0 - (severity * 0.8))
