"""CQRS simulation engine with event sourcing.

Rewritten tick lifecycle using CQRS pattern:
1. Query Phase: Issue read-only context
2. Command Phase: Agents decide in parallel, generate events
3. Commit Phase: Batch flush to Neo4j via UNWIND
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from utopia.layer2_world.neo4j_graph_mutator import Neo4jGraphMutator
from utopia.layer2_world.query_service import KnowledgeGraphQueryService, ReadOnlyContext
from utopia.layer2_world.world_event_buffer import WorldEventBuffer
from utopia.layer2_world.world_events import (
    AgentActionEvent,
    OpinionCreateEvent,
    RelationshipUpdateEvent,
    StanceChangeEvent,
    WorldEvent,
)

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent as CognitionAgent


@dataclass
class SimulationConfig:
    """Configuration for CQRS simulation engine."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    max_concurrent_agents: int = 100
    event_buffer_size: int = 10000
    enable_event_sourcing: bool = True


@dataclass
class TickResult:
    """Result of a single tick execution."""

    tick_number: int
    events_generated: int
    events_committed: int
    commit_duration_ms: float
    agent_decisions: dict[str, Any] = field(default_factory=dict)


class SimulationEngineCQRS:
    """
    CQRS architecture simulation engine.

    Three-phase tick lifecycle:
        1. Query Phase: Issue read-only context to all agents
        2. Command Phase: Agents decide in parallel, events buffered
        3. Commit Phase: Batch write to Neo4j via UNWIND

    Benefits:
        - Lock-free agent parallel execution
        - Single transaction per tick (no write contention)
        - Complete audit trail via event sourcing
        - Scales to 100+ agents

    Example:
        >>> config = SimulationConfig()
        >>> engine = SimulationEngineCQRS(config)
        >>> await engine.initialize(agents)
        >>> result = await engine.run_tick(tick_number=1)
        >>> print(f"Processed {result.events_committed} events")
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize CQRS simulation engine.

        Args:
            config: Simulation configuration
        """
        self.config = config

        # CQRS components
        self.event_buffer = WorldEventBuffer(max_size=config.event_buffer_size)
        self.graph_mutator = Neo4jGraphMutator(
            neo4j_uri=config.neo4j_uri,
            neo4j_user=config.neo4j_user,
            neo4j_password=config.neo4j_password,
        )

        # Query service (read-only)
        self.query_service: Optional[KnowledgeGraphQueryService] = None

        # Agent registry
        self.agents: dict[str, "CognitionAgent"] = {}

        # Statistics
        self._tick_count = 0
        self._total_events = 0

    async def initialize(
        self,
        agents: dict[str, "CognitionAgent"],
        knowledge_graph: Optional[Any] = None,
    ) -> None:
        """
        Initialize engine with agents.

        Args:
            agents: Mapping of agent_id to Agent instances
            knowledge_graph: Optional knowledge graph for queries
        """
        self.agents = agents
        self.query_service = KnowledgeGraphQueryService(knowledge_graph)

    async def run_tick(self, tick_number: int) -> TickResult:
        """
        Execute a single tick with CQRS three-phase lifecycle.

        Phase 1: QUERY - Engine issues read-only context
        Phase 2: COMMAND - Agents decide in parallel, generate events
        Phase 3: COMMIT - Batch flush events to Neo4j

        Args:
            tick_number: Current tick number

        Returns:
            TickResult with execution statistics
        """
        # ========== Phase 1: QUERY (Read-only context) ==========
        read_context = await self._prepare_read_context(tick_number)

        # ========== Phase 2: COMMAND (Parallel decisions) ==========
        agent_events_list = await self._execute_agent_commands(read_context)

        # Collect events to buffer
        total_events = 0
        for events in agent_events_list:
            if events:
                await self.event_buffer.append_many(events)
                total_events += len(events)

        # ========== Phase 3: COMMIT (Batch write) ==========
        events_to_flush = await self.event_buffer.drain(tick_number)

        commit_duration = 0.0
        committed_count = 0

        if events_to_flush:
            result = await self.graph_mutator.flush_events(events_to_flush)
            commit_duration = result["duration_ms"]
            committed_count = result["processed"]

        # Advance to next tick
        self.event_buffer.advance_tick()
        self._tick_count += 1
        self._total_events += committed_count

        return TickResult(
            tick_number=tick_number,
            events_generated=total_events,
            events_committed=committed_count,
            commit_duration_ms=commit_duration,
        )

    async def _prepare_read_context(self, tick_number: int) -> ReadOnlyContext:
        """
        Prepare read-only context for agent queries.

        Agents can ONLY read world state through this interface,
        never directly access Neo4j!

        Args:
            tick_number: Current tick number

        Returns:
            ReadOnlyContext snapshot
        """
        if self.query_service is None:
            # Return empty context if not initialized
            return ReadOnlyContext(
                tick_number=tick_number,
                agent_stances={},
                recent_events=[],
                trust_matrix={},
                active_topics=[],
            )

        return await self.query_service.prepare_context(tick_number)

    async def _execute_agent_commands(
        self, context: ReadOnlyContext
    ) -> list[list[WorldEvent]]:
        """
        Execute all agent commands in parallel.

        Agents run completely in parallel with no lock contention.

        Args:
            context: Read-only context for decision making

        Returns:
            List of event lists from each agent
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        async def run_agent_with_limit(agent_id: str, agent: "CognitionAgent"):
            """Run agent with concurrency limit."""
            async with semaphore:
                return await self._agent_tick_command(agent_id, agent, context)

        # Create tasks for all agents
        tasks = [
            asyncio.create_task(run_agent_with_limit(agent_id, agent))
            for agent_id, agent in self.agents.items()
        ]

        # Execute all agents (asyncio parallelizes automatically)
        return await asyncio.gather(*tasks)

    async def _agent_tick_command(
        self,
        agent_id: str,
        agent: "CognitionAgent",
        context: ReadOnlyContext,
    ) -> list[WorldEvent]:
        """
        Single agent's Command Phase.

        Agent makes decisions and generates events,
        without directly modifying the graph.

        Args:
            agent_id: Agent identifier
            agent: Agent instance
            context: Read-only context

        Returns:
            List of domain events generated
        """
        events: list[WorldEvent] = []

        # Get agent decision from its decision engine
        # This is a placeholder - actual implementation depends on Agent class
        decision = await self._get_agent_decision(agent, context)

        if decision is None:
            return events

        # Convert decision to domain events
        if decision.get("action_type") == "change_stance":
            event = StanceChangeEvent(
                event_id=str(uuid.uuid4()),
                tick_number=context.tick_number,
                source_agent_id=agent_id,
                agent_id=agent_id,
                topic_id=decision.get("topic_id", ""),
                old_position=decision.get("old_position", 0.0),
                new_position=decision.get("new_position", 0.0),
                confidence=decision.get("confidence", 0.5),
            )
            events.append(event)

        elif decision.get("action_type") == "speak":
            # Create opinion event
            opinion_event = OpinionCreateEvent(
                event_id=str(uuid.uuid4()),
                tick_number=context.tick_number,
                source_agent_id=agent_id,
                opinion_id=f"op_{agent_id}_{context.tick_number}_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                topic_id=decision.get("topic_id", ""),
                content=decision.get("content", ""),
                stance_position=decision.get("stance_position", 0.0),
                confidence=decision.get("confidence", 0.5),
            )
            events.append(opinion_event)

            # Update influence relationships
            for listener_id in decision.get("target_listeners", []):
                current_influence = await context.get_influence(agent_id, listener_id)
                rel_event = RelationshipUpdateEvent(
                    event_id=str(uuid.uuid4()),
                    tick_number=context.tick_number,
                    source_agent_id=agent_id,
                    from_node_id=agent_id,
                    to_node_id=listener_id,
                    relationship_type="INFLUENCES",
                    old_weight=current_influence,
                    new_weight=min(1.0, current_influence + 0.1),
                    delta=0.1,
                )
                events.append(rel_event)

        elif decision.get("action_type") == "act":
            action_event = AgentActionEvent(
                event_id=str(uuid.uuid4()),
                tick_number=context.tick_number,
                source_agent_id=agent_id,
                action_type=decision.get("action_subtype", "generic"),
                target_agent_id=decision.get("target_agent_id"),
                content=decision.get("content", ""),
                topic_id=decision.get("topic_id"),
                importance=decision.get("importance", 0.5),
            )
            events.append(action_event)

        return events

    async def _get_agent_decision(
        self,
        agent: "CognitionAgent",
        context: ReadOnlyContext,
    ) -> Optional[dict[str, Any]]:
        """
        Get decision from agent.

        This is a bridge method that calls the agent's decision engine.
        Actual implementation depends on Agent class structure.

        Args:
            agent: Agent instance
            context: Read-only context

        Returns:
            Decision dict or None
        """
        # Check if agent has a decide method
        if hasattr(agent, "decide"):
            try:
                # Try async decide
                if asyncio.iscoroutinefunction(agent.decide):
                    return await agent.decide(context)
                else:
                    # Fall back to sync decide
                    return agent.decide(context)
            except Exception as e:
                # Log error but don't crash the tick
                print(f"Agent {agent.agent_id if hasattr(agent, 'agent_id') else 'unknown'} "
                      f"decision error: {e}")
                return None

        # If no decide method, return None
        return None

    async def close(self) -> None:
        """Clean up resources."""
        await self.graph_mutator.close()

    def get_stats(self) -> dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dict with tick_count, total_events, buffer_stats, mutator_stats
        """
        return {
            "tick_count": self._tick_count,
            "total_events": self._total_events,
            "buffer_stats": self.event_buffer.get_stats(),
            "mutator_stats": self.graph_mutator.get_stats(),
        }


# Backward compatibility aliases
SimulationEngine = SimulationEngineCQRS


@dataclass
class WorldState:
    """Backward compatibility WorldState.

    DEPRECATED: Use WorldStateBuffer instead.
    """

    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"tick": self.tick}


@dataclass
class SimulationResult:
    """Backward compatibility SimulationResult.

    DEPRECATED: Use TickResult instead.
    """

    final_tick: int = 0
    agent_count: int = 0
    domain: str = "general"
    duration_seconds: float = 0.0
    converged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_tick": self.final_tick,
            "agent_count": self.agent_count,
            "domain": self.domain,
            "duration_seconds": self.duration_seconds,
            "converged": self.converged,
        }
