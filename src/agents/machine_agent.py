"""item_agent.py — Item-level digital twin (leaf node, one per item_id).

The most granular twin in the hierarchy. Holds demand features derived
from the raw M5 sales data for a single product across all stores.
"""

from __future__ import annotations
import numpy as np
from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)


class ItemAgent(DigitalTwinAgent):
    """Item-level twin tracking per-product demand and stockout behaviour.

    Attributes
    ----------
    mean_sales : float
        Normalized mean daily sales [0, 1].
    sales_cv : float
        Coefficient of variation of daily sales [0, 1].
    zero_rate : float
        Fraction of days with zero sales [0, 1].
    max_zero_streak : float
        Normalized longest consecutive zero-sales run [0, 1].
    price : float
        Normalized current sell price [0, 1].
    price_change_freq : float
        Normalized frequency of price changes [0, 1].
    trend_slope : float
        Normalized demand trend slope [0, 1]. 0.5 = flat.
    snap_sensitivity : float
        Normalized SNAP-day sensitivity [0, 1].
    """

    def __init__(
        self,
        node_id: str,
        mean_sales: float = 0.5,
        sales_cv: float = 0.3,
        zero_rate: float = 0.5,
        max_zero_streak: float = 0.2,
        price: float = 0.5,
        price_change_freq: float = 0.1,
        trend_slope: float = 0.5,
        snap_sensitivity: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(node_id=node_id, layer="item", **kwargs)
        self.mean_sales: float = mean_sales
        self.sales_cv: float = sales_cv
        self.zero_rate: float = zero_rate
        self.max_zero_streak: float = max_zero_streak
        self.price: float = price
        self.price_change_freq: float = price_change_freq
        self.trend_slope: float = trend_slope
        self.snap_sensitivity: float = snap_sensitivity

    def compute_health_score(self) -> float:
        return float(np.clip(
            0.25 * self.capacity
            + 0.25 * (1.0 - self.zero_rate)
            + 0.20 * self.throughput
            + 0.15 * (1.0 - self.max_zero_streak)
            + 0.15 * (1.0 - self.failure_prob),
            0.0, 1.0,
        ))

    def _apply_specific_disruption(self, severity: float, step: int) -> None:
        self.mean_sales = max(0.0, self.mean_sales * (1.0 - severity * 0.5))
        self.zero_rate = min(1.0, self.zero_rate + severity * 0.3)
        self.max_zero_streak = min(1.0, self.max_zero_streak + severity * 0.2)
        self.sales_cv = min(1.0, self.sales_cv + severity * 0.15)
