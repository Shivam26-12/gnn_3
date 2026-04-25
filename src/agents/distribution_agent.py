"""category_agent.py — Category-level digital twin (FOODS, HOBBIES, HOUSEHOLD)."""

from __future__ import annotations
import numpy as np
from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)


class CategoryAgent(DigitalTwinAgent):
    """Category-level twin tracking top-level retail category health.

    Attributes
    ----------
    total_demand : float
        Normalized aggregate demand across all items in this category [0, 1].
    dept_count : float
        Normalized count of departments under this category [0, 1].
    avg_price : float
        Normalized average sell price [0, 1].
    """

    def __init__(
        self,
        node_id: str,
        total_demand: float = 0.5,
        dept_count: float = 0.5,
        avg_price: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(node_id=node_id, layer="category", **kwargs)
        self.total_demand: float = total_demand
        self.dept_count: float = dept_count
        self.avg_price: float = avg_price

    def compute_health_score(self) -> float:
        return float(np.clip(
            0.40 * self.capacity
            + 0.30 * self.throughput
            + 0.20 * (1.0 - self.failure_prob)
            + 0.10 * self.total_demand,
            0.0, 1.0,
        ))

    def _apply_specific_disruption(self, severity: float, step: int) -> None:
        self.total_demand = max(0.0, self.total_demand * (1.0 - severity * 0.4))
        self.avg_price = min(1.0, self.avg_price * (1.0 + severity * 0.2))
