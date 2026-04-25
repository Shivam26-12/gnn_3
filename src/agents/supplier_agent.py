"""state_agent.py — State-level digital twin (CA, TX, WI)."""

from __future__ import annotations
import numpy as np
from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)


class StateAgent(DigitalTwinAgent):
    """State-level twin tracking aggregate demand and policy sensitivity.

    Attributes
    ----------
    total_demand : float
        Aggregate daily demand across all stores in this state [0, 1] normalized.
    snap_sensitivity : float
        Sales lift on SNAP days in [0, 1].
    event_sensitivity : float
        Sales lift on event days in [0, 1].
    demand_volatility : float
        Coefficient of variation of aggregate demand [0, 1].
    """

    def __init__(
        self,
        node_id: str,
        total_demand: float = 0.5,
        snap_sensitivity: float = 0.1,
        event_sensitivity: float = 0.05,
        demand_volatility: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(node_id=node_id, layer="state", **kwargs)
        self.total_demand: float = total_demand
        self.snap_sensitivity: float = snap_sensitivity
        self.event_sensitivity: float = event_sensitivity
        self.demand_volatility: float = demand_volatility

    def compute_health_score(self) -> float:
        return float(np.clip(
            0.40 * self.capacity
            + 0.30 * (1.0 - self.demand_volatility)
            + 0.20 * self.throughput
            + 0.10 * (1.0 - self.failure_prob),
            0.0, 1.0,
        ))

    def _apply_specific_disruption(self, severity: float, step: int) -> None:
        self.total_demand = max(0.0, self.total_demand * (1.0 - severity * 0.4))
        self.demand_volatility = min(1.0, self.demand_volatility + severity * 0.3)
