"""model.py — DTNetModel: Mesa simulation for retail disruption cascading.

Wraps NetworkX nodes in Mesa agents for cascading disruption simulation.
Adapted for M5 retail context.
"""

from __future__ import annotations

import numpy as np
import torch
import networkx as nx
import mesa
from mesa import DataCollector
from typing import Any, Dict, List, Set

from src.agents.base_agent import DigitalTwinAgent

np.random.seed(42)
torch.manual_seed(42)

DEFAULT_PROPAGATION_DECAY: float = 0.75
DEFAULT_THRESHOLD: float = 0.05


class TwinMesaAgent(mesa.Agent):
    """Mesa wrapper for a single DigitalTwinAgent node."""

    def __init__(self, model: "DTNetModel", node_id: str, twin: DigitalTwinAgent) -> None:
        super().__init__(model)
        self.node_id = node_id
        self.twin = twin
        self._pending_severity = 0.0

    def step(self) -> None:
        G = self.model.G
        max_incoming = 0.0
        for pred_id in G.predecessors(self.node_id):
            pred_twin = G.nodes[pred_id]["twin"]
            if not pred_twin.is_disrupted:
                continue
            edge_data = G.edges[pred_id, self.node_id]
            criticality = float(edge_data.get("criticality_weight", 1.0))
            incoming = pred_twin.disruption_severity * criticality * self.model.propagation_decay
            vulnerability = 1.0 - self.twin.compute_health_score()
            adjusted = incoming * (1.0 + vulnerability * 0.5)
            max_incoming = max(max_incoming, adjusted)
        self._pending_severity = max_incoming

    def advance(self) -> None:
        if self._pending_severity > self.model.threshold:
            self.twin.apply_disruption(self._pending_severity, self.model.steps)
        self.twin.step()
        self._pending_severity = 0.0


class DTNetModel(mesa.Model):
    """Mesa simulation model for retail disruption cascading."""

    def __init__(self, G: nx.DiGraph, propagation_decay: float = DEFAULT_PROPAGATION_DECAY,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        super().__init__()
        np.random.seed(42)
        self.G = G
        self.propagation_decay = propagation_decay
        self.threshold = threshold
        self._agent_map: Dict[str, TwinMesaAgent] = {}
        self._disrupted_prev: Set[str] = set()
        self._newly_disrupted: List[str] = []

        for _, (node_id, data) in enumerate(G.nodes(data=True)):
            twin = data["twin"]
            agent = TwinMesaAgent(self, node_id, twin)
            self._agent_map[node_id] = agent

        self.datacollector = self._make_datacollector()

    def _make_datacollector(self) -> DataCollector:
        return DataCollector(
            model_reporters={
                "num_disrupted": lambda m: sum(1 for _, d in m.G.nodes(data=True) if d["twin"].is_disrupted),
                "avg_health": lambda m: float(np.mean([d["twin"].compute_health_score() for _, d in m.G.nodes(data=True)])),
                "avg_capacity": lambda m: float(np.mean([d["twin"].capacity for _, d in m.G.nodes(data=True)])),
                "newly_disrupted": lambda m: list(m._newly_disrupted),
            },
            agent_reporters={
                "disruption_severity": lambda a: a.twin.disruption_severity,
                "health_score": lambda a: a.twin.compute_health_score(),
                "capacity": lambda a: a.twin.capacity,
            },
        )

    def step(self) -> None:
        self._disrupted_prev = {nid for nid, d in self.G.nodes(data=True) if d["twin"].is_disrupted}
        self.agents.do("step")
        self.agents.do("advance")
        disrupted_now = {nid for nid, d in self.G.nodes(data=True) if d["twin"].is_disrupted}
        self._newly_disrupted = sorted(disrupted_now - self._disrupted_prev)
        self.datacollector.collect(self)

    def inject_disruption(self, node_id: str, severity: float) -> None:
        if node_id not in self.G.nodes:
            raise KeyError(f"Node '{node_id}' not found.")
        self.G.nodes[node_id]["twin"].apply_disruption(severity, self.steps)

    def reset(self) -> None:
        for _, data in self.G.nodes(data=True):
            data["twin"].reset()
        self._disrupted_prev = set()
        self._newly_disrupted = []
        self.datacollector = self._make_datacollector()
