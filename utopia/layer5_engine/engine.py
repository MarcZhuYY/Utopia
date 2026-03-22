"""Main simulation engine.

Orchestrates the tick-by-tick simulation loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import structlog

from utopia.core.config import SimulationConfig, WorldRules
from utopia.core.models import (
    ExternalEvent,
    SeedMaterial,
)
from utopia.layer1_seed.parser import MaterialParser
from utopia.layer2_world.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from utopia.layer2_world.rule_engine import RuleEngine
from utopia.layer3_cognition.agent import Agent as CognitionAgent
from utopia.layer3_cognition.decision_engine import AgentDecisionEngine, SimulationContext
from utopia.layer4_social.relationships import RelationshipMap
from utopia.layer4_social.propagator import InformationPropagator, create_message
from utopia.layer4_social.dynamics import GroupDynamicsDetector
from utopia.layer5_engine.convergence import ConvergenceDetector
from utopia.layer5_engine.event_injector import ExternalEventInjector

logger = structlog.get_logger()


@dataclass
class WorldState:
    """Complete world state for simulation.

    Contains all mutable state that changes during simulation.
    """

    agents: dict[str, CognitionAgent] = field(default_factory=dict)
    relationship_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    relationship_map: RelationshipMap = field(default_factory=RelationshipMap)
    external_events: list[ExternalEvent] = field(default_factory=list)
    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "agent_count": len(self.agents),
            "events": [e.to_dict() for e in self.external_events],
        }


@dataclass
class SimulationResult:
    """Result of a simulation run.

    Contains all outputs and metrics.
    """

    final_tick: int = 0
    agent_count: int = 0
    domain: str = "general"
    duration_seconds: float = 0.0
    total_llm_calls: int = 0
    estimated_cost: float = 0.0
    converged: bool = False
    convergence_reason: str = ""
    dynamics_history: list[dict[str, Any]] = field(default_factory=list)
    agent_traces: dict[str, list[dict]] = field(default_factory=dict)
    stance_history: dict[str, list[dict]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_tick": self.final_tick,
            "agent_count": self.agent_count,
            "domain": self.domain,
            "duration_seconds": self.duration_seconds,
            "total_llm_calls": self.total_llm_calls,
            "estimated_cost": self.estimated_cost,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "dynamics_history": self.dynamics_history,
            "stance_history": self.stance_history,
        }


class SimulationEngine:
    """Main simulation engine.

    Executes the tick-by-tick simulation loop:
    1. External event injection
    2. Perception phase
    3. Decision phase
    4. Action phase
    5. Interaction phase
    6. World update
    7. Convergence check
    """

    def __init__(self, config: SimulationConfig):
        """Initialize simulation engine.

        Args:
            config: Simulation configuration
        """
        self.config = config
        self.tick_count = 0
        self.world_state: WorldState | None = None
        self.logger = logger
        self.event_injector: ExternalEventInjector | None = None
        self.convergence_detector = ConvergenceDetector(config.world_rules)
        self.group_dynamics = GroupDynamicsDetector()
        self.rule_engine = RuleEngine(config.world_rules, config.domain)

        # State
        self._agent_traces: dict[str, list[dict]] = {}
        self._stance_history: dict[str, list[dict]] = {}
        self._dynamics_history: list[dict[str, Any]] = []
        self._llm_call_count = 0
        self._start_time: float = 0

    def initialize(self, seed: SeedMaterial) -> None:
        """Initialize simulation from seed material.

        Args:
            seed: Seed material
        """
        self.logger.info("Initializing simulation", agent_count=self.config.agent_count)

        # Build world state
        self.world_state = WorldState()

        # Build knowledge graph
        kg_builder = KnowledgeGraphBuilder()
        self.world_state.knowledge_graph = kg_builder.build_from_seed(seed)

        # Generate agents from entities or create defaults
        self._generate_agents(seed)

        # Build relationship network
        self._build_relationship_network()

        # Initialize event injector
        self.event_injector = ExternalEventInjector(self)

        # Initialize traces
        self._agent_traces = {aid: [] for aid in self.world_state.agents}
        self._stance_history = {}

        self.logger.info("Initialization complete", agents=len(self.world_state.agents))

    def _generate_agents(self, seed: SeedMaterial) -> None:
        """Generate agents from seed material.

        Args:
            seed: Seed material
        """
        # Use entities from seed to create agents
        agent_id = 0

        for entity in seed.entities:
            if entity.type.value in ["person", "org"]:
                agent = CognitionAgent.from_entity(
                    entity_id=entity.id,
                    name=entity.name,
                    role=entity.attributes.get("role", "analyst"),
                    expertise=[entity.attributes.get("sector", "general")],
                    base_stances=entity.initial_stance,
                    influence=entity.influence_score,
                )
                self.world_state.agents[entity.id] = agent
                agent_id += 1

        # If not enough agents from entities, create defaults
        if len(self.world_state.agents) < self.config.agent_count:
            remaining = self.config.agent_count - len(self.world_state.agents)
            for i in range(remaining):
                agent_id_str = f"GEN{agent_id + i}"
                agent = CognitionAgent.from_entity(
                    entity_id=agent_id_str,
                    name=f"Agent_{agent_id_str}",
                    role="analyst",
                    expertise=["general"],
                    base_stances={},
                    influence=0.5,
                )
                self.world_state.agents[agent_id_str] = agent

        # Set relationship map reference for all agents
        for agent in self.world_state.agents.values():
            agent.set_relationship_map(self.world_state.relationship_map)

    def _build_relationship_network(self) -> None:
        """Build agent relationship network."""
        agent_ids = list(self.world_state.agents.keys())

        # Build complete graph initially
        self.world_state.relationship_map.build_complete_graph(
            agent_ids, base_trust=0.1
        )

        # Create NetworkX graph for propagation
        self.world_state.relationship_graph = nx.DiGraph()
        for aid in agent_ids:
            self.world_state.relationship_graph.add_node(aid)
        for i, ida in enumerate(agent_ids):
            for idb in agent_ids:
                if ida != idb:
                    edge = self.world_state.relationship_map.get(ida, idb)
                    self.world_state.relationship_graph.add_edge(
                        ida, idb, weight=edge.influence_weight
                    )

    def run(self) -> SimulationResult:
        """Run simulation until convergence or max ticks.

        Returns:
            SimulationResult: Simulation result
        """
        if not self.world_state:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._start_time = time.time()
        max_ticks = self.config.max_ticks

        self.logger.info("Starting simulation", max_ticks=max_ticks)

        while self.tick_count < max_ticks:
            # Pre-tick hook
            self._pre_tick()

            # Execute tick
            self._execute_tick()

            # Detect dynamics
            dynamics = self._detect_dynamics()

            # Log tick
            self._log_tick(dynamics)

            # Check convergence
            conv_result = self.convergence_detector.check(self._dynamics_history)
            if conv_result.converged:
                self.logger.info(
                    "Simulation converged",
                    reason=conv_result.reason,
                    ticks=self.tick_count,
                )
                break

            self.tick_count += 1

        duration = time.time() - self._start_time

        return SimulationResult(
            final_tick=self.tick_count,
            agent_count=len(self.world_state.agents),
            domain=self.config.domain,
            duration_seconds=duration,
            total_llm_calls=self._llm_call_count,
            estimated_cost=self._estimate_cost(),
            converged=conv_result.converged,
            convergence_reason=conv_result.reason,
            dynamics_history=self._dynamics_history,
            agent_traces=self._agent_traces,
            stance_history=self._stance_history,
        )

    def _pre_tick(self) -> None:
        """Pre-tick processing."""
        if self.event_injector:
            self.event_injector.inject_pending()

    def _execute_tick(self) -> None:
        """Execute one simulation tick."""
        # 1. Collect perceptions (messages received)
        all_perceptions = self._collect_perceptions()

        # 2. Decision phase - each agent decides independently
        decisions = self._parallel_decide(all_perceptions)

        # 3. Execute actions
        actions = self._execute_actions(decisions)

        # 4. Propagate information
        propagations = self._propagate_information(actions)

        # 5. Update world state
        self._update_world_state(actions, propagations)

    def _collect_perceptions(self) -> dict[str, list]:
        """Collect perceptions for all agents.

        Returns:
            dict[str, list]: agent_id -> list of perceptions
        """
        # For MVP, no passive perceptions - agents only perceive what they act on
        return {aid: [] for aid in self.world_state.agents}

    def _parallel_decide(
        self,
        perceptions: dict[str, list],
    ) -> dict[str, list]:
        """Make decisions for all agents.

        Args:
            perceptions: Perceptions per agent

        Returns:
            dict[str, list]: agent_id -> actions
        """
        context = SimulationContext(
            current_tick=self.tick_count,
            world_state=self.world_state,
            config=self.config,
            current_situation_summary=self._get_situation_summary(),
        )

        engine = AgentDecisionEngine(self.config)
        results = {}

        for agent_id, agent in self.world_state.agents.items():
            # Simple decision engine for MVP (no LLM)
            from utopia.layer3_cognition.decision_engine import SimpleDecisionEngine
            simple_engine = SimpleDecisionEngine()
            actions = simple_engine.decide(agent, context, perceptions[agent_id])
            results[agent_id] = actions

        return results

    def _execute_actions(self, decisions: dict[str, list]) -> list[dict]:
        """Execute decided actions.

        Args:
            decisions: Decisions per agent

        Returns:
            list[dict]: Executed actions with metadata
        """
        executed = []

        for agent_id, actions in decisions.items():
            for action in actions:
                executed.append({
                    "agent_id": agent_id,
                    "action": action,
                })

        return executed

    def _propagate_information(self, actions: list[dict]) -> list:
        """Propagate information from speak actions.

        Args:
            actions: Executed actions

        Returns:
            list: Propagation results
        """
        propagator = InformationPropagator(self.config.world_rules)
        results = []

        for item in actions:
            if item["action"].action_type == "speak":
                agent = self.world_state.agents[item["agent_id"]]
                message = create_message(
                    content=item["action"].content,
                    sender_id=item["agent_id"],
                    topic_id=item["action"].topic_id,
                    original_stance=agent.get_stance(item["action"].topic_id).position
                    if agent.get_stance(item["action"].topic_id)
                    else 0.0,
                )

                received = propagator.propagate(
                    message=message,
                    sender=agent,
                    graph=self.world_state.relationship_graph,
                    relationships=self.world_state.relationship_map,
                    agents=self.world_state.agents,
                )
                results.extend(received)

        return results

    def _update_world_state(
        self,
        actions: list[dict],
        propagations: list,
    ) -> None:
        """Update world state after tick.

        Args:
            actions: Executed actions
            propagations: Propagation results
        """
        # Decay agent energy
        for agent in self.world_state.agents.values():
            agent.decay_energy(0.02)

        # Record traces
        for item in actions:
            self._agent_traces[item["agent_id"]].append({
                "tick": self.tick_count,
                "action": item["action"].to_dict(),
            })

    def _detect_dynamics(self) -> dict[str, Any]:
        """Detect group dynamics.

        Returns:
            dict: Dynamics metrics
        """
        dynamics = {}

        if self.config.enable_polarization_detection:
            # Detect on first topic
            if self.world_state.knowledge_graph.graph.nodes():
                topic_id = list(self.world_state.knowledge_graph.graph.nodes())[0]
                polarization = self.group_dynamics.detect_polarization(
                    list(self.world_state.agents.values()), topic_id
                )
                dynamics["polarization"] = polarization.to_dict()

        self._dynamics_history.append(dynamics)
        return dynamics

    def _log_tick(self, dynamics: dict[str, Any]) -> None:
        """Log tick progress."""
        if self.tick_count % 10 == 0:
            self.logger.info(
                "Tick progress",
                tick=self.tick_count,
                dynamics=dynamics,
            )

    def _get_situation_summary(self) -> str:
        """Get current situation summary.

        Returns:
            str: Situation summary for context
        """
        if not self.world_state:
            return ""

        return f"Tick {self.tick_count}, {len(self.world_state.agents)} agents active"

    def _estimate_cost(self) -> float:
        """Estimate LLM cost.

        Returns:
            float: Estimated cost in USD
        """
        # Rough estimate: $0.001 per LLM call
        return self._llm_call_count * 0.001

    def save_results(self, output_dir: str) -> None:
        """Save simulation results to files.

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result = SimulationResult(
            final_tick=self.tick_count,
            agent_count=len(self.world_state.agents) if self.world_state else 0,
            domain=self.config.domain,
            duration_seconds=time.time() - self._start_time,
            dynamics_history=self._dynamics_history,
            agent_traces=self._agent_traces,
            stance_history=self._stance_history,
        )

        with open(output_path / "result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        if self.config.save_graph and self.world_state:
            self.world_state.knowledge_graph.save_json(str(output_path / "knowledge_graph.json"))

        self.logger.info("Results saved", output_dir=str(output_path))
