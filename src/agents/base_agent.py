"""
base_agent.py — Abstract base class for all M5 retail digital twin agents.

Each node in the supply-chain graph wraps one ``DigitalTwinAgent`` subclass.
The base defines generic operational state (capacity, throughput, failure_prob),
disruption mechanics, and the ``compute_health_score`` / ``apply_disruption`` /
``reset`` contract that all subclasses implement.

Adapted from the industrial DTNet base for M5 Walmart retail context.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

np.random.seed(42)


class DigitalTwinAgent(ABC):
    """Abstract base class for a retail digital-twin agent (one per graph node).

    Attributes
    ----------
    node_id : str
        Unique graph node identifier.
    layer : str
        One of 'state', 'store', 'department', 'category', 'item'.
    capacity : float
        Operational capacity in [0, 1]. 1.0 = fully operational.
    throughput : float
        Current throughput fraction in [0, 1].
    failure_prob : float
        Estimated probability of failure in [0, 1].
    is_disrupted : bool
        True if this agent has been hit by a disruption.
    disruption_severity : float
        Current disruption severity in [0, 1]. 0 = healthy.
    disruption_step : int
        Simulation step when disruption was first applied.
    _initial_state : dict
        Snapshot of all attributes at construction time for ``reset()``.
    """

    def __init__(
        self,
        node_id: str,
        layer: str,
        capacity: float = 1.0,
        throughput: float = 1.0,
        failure_prob: float = 0.0,
    ) -> None:
        self.node_id: str = node_id
        self.layer: str = layer
        self.capacity: float = capacity
        self.throughput: float = throughput
        self.failure_prob: float = failure_prob
        self.is_disrupted: bool = False
        self.disruption_severity: float = 0.0
        self.disruption_step: int = -1

        # Subclass __init__ must finish before _save_initial_state is called
        # (called explicitly by the builder after construction).
        self._initial_state: Dict[str, Any] = {}

    def save_initial_state(self) -> None:
        """Snapshot current attribute values for ``reset()``."""
        self._initial_state = {
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
            if k != "_initial_state"
        }

    @abstractmethod
    def compute_health_score(self) -> float:
        """Return the health score of this agent in [0, 1].

        Must be implemented by every subclass with domain-specific logic.
        """
        ...

    def apply_disruption(self, severity: float, step: int) -> None:
        """Apply a disruption of the given severity at the given step.

        Parameters
        ----------
        severity : float
            Disruption magnitude in [0, 1].
        step : int
            Current simulation timestep.
        """
        self.is_disrupted = True
        self.disruption_severity = max(self.disruption_severity, severity)
        if self.disruption_step < 0:
            self.disruption_step = step

        # Generic degradation: reduce capacity and throughput
        self.capacity = max(0.0, self.capacity * (1.0 - severity * 0.5))
        self.throughput = max(0.0, self.throughput * (1.0 - severity * 0.4))
        self.failure_prob = min(1.0, self.failure_prob + severity * 0.3)

        # Subclass-specific degradation
        self._apply_specific_disruption(severity, step)

    @abstractmethod
    def _apply_specific_disruption(self, severity: float, step: int) -> None:
        """Subclass-specific disruption logic."""
        ...

    def step(self) -> None:
        """Advance internal counters by one simulation timestep."""
        pass

    def reset(self) -> None:
        """Restore all attributes to their initial values."""
        for key, val in self._initial_state.items():
            setattr(self, key, copy.deepcopy(val))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.node_id!r}, "
            f"layer={self.layer!r}, health={self.compute_health_score():.3f})"
        )
