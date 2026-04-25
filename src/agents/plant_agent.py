"""store_agent.py — Store-level digital twin (CA_1, TX_2, WI_3, etc.)."""

from __future__ import annotations
import numpy as np
from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)


class StoreAgent(DigitalTwinAgent):
    """Store-level twin tracking demand, stockout risk, and fulfillment.

    Attributes
    ----------
    mean_daily_sales : float
        Normalized average daily sales [0, 1].
    sales_volatility : float
        Coefficient of variation of daily sales [0, 1].
    zero_rate : float
        Fraction of item-days with zero sales [0, 1].
    snap_sensitivity : float
        Normalized SNAP-day sales lift [0, 1].
    n_active_items : float
        Normalized count of items with recent sales [0, 1].
    weekend_ratio : float
        Normalized weekend-to-weekday sales ratio [0, 1].
    """

    def __init__(
        self,
        node_id: str,
        mean_daily_sales: float = 0.5,
        sales_volatility: float = 0.2,
        zero_rate: float = 0.5,
        snap_sensitivity: float = 0.1,
        n_active_items: float = 0.8,
        weekend_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(node_id=node_id, layer="store", **kwargs)
        self.mean_daily_sales: float = mean_daily_sales
        self.sales_volatility: float = sales_volatility
        self.zero_rate: float = zero_rate
        self.snap_sensitivity: float = snap_sensitivity
        self.n_active_items: float = n_active_items
        self.weekend_ratio: float = weekend_ratio

    def compute_health_score(self) -> float:
        return float(np.clip(
            0.30 * self.capacity
            + 0.25 * (1.0 - self.zero_rate)
            + 0.20 * self.throughput
            + 0.15 * self.n_active_items
            + 0.10 * (1.0 - self.failure_prob),
            0.0, 1.0,
        ))

    def _apply_specific_disruption(self, severity: float, step: int) -> None:
        self.mean_daily_sales = max(0.0, self.mean_daily_sales * (1.0 - severity * 0.5))
        self.zero_rate = min(1.0, self.zero_rate + severity * 0.3)
        self.n_active_items = max(0.0, self.n_active_items * (1.0 - severity * 0.4))
