"""department_agent.py — Department-level digital twin (FOODS_1, HOBBIES_2, etc.)."""

from __future__ import annotations
import numpy as np
from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)


class DepartmentAgent(DigitalTwinAgent):
    """Department-level twin tracking category-level demand dynamics.

    Attributes
    ----------
    mean_sales : float
        Normalized mean daily sales for this department [0, 1].
    volatility : float
        Coefficient of variation of daily department sales [0, 1].
    zero_rate : float
        Fraction of item-days with zero sales [0, 1].
    item_count : float
        Normalized count of items in this department [0, 1].
    """

    def __init__(
        self,
        node_id: str,
        mean_sales: float = 0.5,
        volatility: float = 0.2,
        zero_rate: float = 0.5,
        item_count: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(node_id=node_id, layer="department", **kwargs)
        self.mean_sales: float = mean_sales
        self.volatility: float = volatility
        self.zero_rate: float = zero_rate
        self.item_count: float = item_count

    def compute_health_score(self) -> float:
        return float(np.clip(
            0.35 * self.capacity
            + 0.25 * (1.0 - self.zero_rate)
            + 0.25 * self.throughput
            + 0.15 * (1.0 - self.failure_prob),
            0.0, 1.0,
        ))

    def _apply_specific_disruption(self, severity: float, step: int) -> None:
        self.mean_sales = max(0.0, self.mean_sales * (1.0 - severity * 0.4))
        self.zero_rate = min(1.0, self.zero_rate + severity * 0.25)
        self.volatility = min(1.0, self.volatility + severity * 0.2)
