"""Read-only query service for CQRS architecture.

Agents use this service to read world state.
All methods are read-only, no writes allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from utopia.layer2_world.knowledge_graph import NodeType, EdgeType

if TYPE_CHECKING:
    from utopia.layer2_world.knowledge_graph import KnowledgeGraph


@dataclass
class AgentStance:
    """Agent stance data transfer object."""

    agent_id: str
    topic_id: str
    position: float
    confidence: float


@dataclass
class TrustRelationship:
    """Trust relationship data transfer object."""

    from_agent: str
    to_agent: str
    weight: float


@dataclass
class RecentEvent:
    """Recent event data transfer object."""

    event_id: str
    event_type: str
    timestamp: str
    description: str


@dataclass
class ReadOnlyContext:
    """
    Read-only context for Agent decision making.

    This is the CQRS Query side - Agents receive this snapshot
    of world state for their tick decision.
    """

    tick_number: int
    agent_stances: dict[str, list[AgentStance]]
    recent_events: list[RecentEvent]
    trust_matrix: dict[tuple[str, str], float]
    active_topics: list[str]

    async def get_influence(self, from_agent: str, to_agent: str) -> float:
        """
        Get influence weight from one agent to another.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID

        Returns:
            Influence weight (0.0 if no relationship)
        """
        return self.trust_matrix.get((from_agent, to_agent), 0.0)

    def get_agent_stance_on_topic(self, agent_id: str, topic_id: str) -> Optional[AgentStance]:
        """
        Get a specific agent's stance on a topic.

        Args:
            agent_id: Agent ID
            topic_id: Topic ID

        Returns:
            AgentStance if found, None otherwise
        """
        stances = self.agent_stances.get(agent_id, [])
        for stance in stances:
            if stance.topic_id == topic_id:
                return stance
        return None


class KnowledgeGraphQueryService:
    """
    Knowledge graph query service - CQRS Query side.

    Responsibilities:
        1. Provide read-only access to world state
        2. Prepare ReadOnlyContext for Agent decision making
        3. No write operations allowed

    Warning:
        This service is READ-ONLY. All writes must go through
        WorldEventBuffer and Neo4jGraphMutator.

    Example:
        >>> service = KnowledgeGraphQueryService(knowledge_graph)
        >>> context = await service.prepare_context(tick_number=42)
        >>> stance = context.get_agent_stance_on_topic("A1", "TOPIC_Tariffs")
    """

    def __init__(self, knowledge_graph: Optional["KnowledgeGraph"] = None):
        """
        Initialize query service.

        Args:
            knowledge_graph: Knowledge graph instance (can be None for testing)
        """
        self._kg = knowledge_graph

    async def prepare_context(
        self,
        tick_number: int,
        agent_filter: Optional[list[str]] = None,
        topic_filter: Optional[list[str]] = None,
    ) -> ReadOnlyContext:
        """
        Prepare read-only context for Agent decision making.

        Args:
            tick_number: Current tick number
            agent_filter: Optional list of agent IDs to include
            topic_filter: Optional list of topic IDs to include

        Returns:
            ReadOnlyContext snapshot
        """
        return ReadOnlyContext(
            tick_number=tick_number,
            agent_stances=await self.get_all_stances(agent_filter, topic_filter),
            recent_events=await self.get_recent_events(limit=100),
            trust_matrix=await self.get_trust_matrix(agent_filter),
            active_topics=await self.get_active_topics(),
        )

    async def get_all_stances(
        self,
        agent_filter: Optional[list[str]] = None,
        topic_filter: Optional[list[str]] = None,
    ) -> dict[str, list[AgentStance]]:
        """
        Get all agent stances.

        Args:
            agent_filter: Optional agent IDs to filter
            topic_filter: Optional topic IDs to filter

        Returns:
            Dict mapping agent_id to list of AgentStance
        """
        if self._kg is None:
            return {}

        result: dict[str, list[AgentStance]] = {}

        # Get all agent nodes from knowledge graph
        for node_id, node_data in self._kg.graph.nodes(data=True):
            if node_data.get("node_type") != NodeType.AGENT.value:
                continue
            if agent_filter and node_id not in agent_filter:
                continue

            stances: list[AgentStance] = []

            # Get stance relationships
            for _, neighbor, edge_data in self._kg.graph.edges(node_id, data=True):
                neighbor_data = self._kg.graph.nodes[neighbor]
                if neighbor_data.get("node_type") != NodeType.TOPIC.value:
                    continue
                if topic_filter and neighbor not in topic_filter:
                    continue

                stance = AgentStance(
                    agent_id=node_id,
                    topic_id=neighbor,
                    position=edge_data.get("weight", 0.0),
                    confidence=edge_data.get("confidence", 0.5),
                )
                stances.append(stance)

            if stances:
                result[node_id] = stances

        return result

    async def get_trust_matrix(
        self, agent_filter: Optional[list[str]] = None
    ) -> dict[tuple[str, str], float]:
        """
        Get trust/influence matrix between agents.

        Args:
            agent_filter: Optional agent IDs to include

        Returns:
            Dict mapping (from_agent, to_agent) to weight
        """
        if self._kg is None:
            return {}

        result: dict[tuple[str, str], float] = {}

        for node_id, node_data in self._kg.graph.nodes(data=True):
            if node_data.get("node_type") != NodeType.AGENT.value:
                continue
            if agent_filter and node_id not in agent_filter:
                continue

            for _, neighbor, edge_data in self._kg.graph.edges(node_id, data=True):
                neighbor_data = self._kg.graph.nodes[neighbor]
                if neighbor_data.get("node_type") != NodeType.AGENT.value:
                    continue
                if agent_filter and neighbor not in agent_filter:
                    continue

                weight = edge_data.get("weight", 0.0)
                result[(node_id, neighbor)] = weight

        return result

    async def get_recent_events(
        self, limit: int = 100, event_types: Optional[list[str]] = None
    ) -> list[RecentEvent]:
        """
        Get recent events from the world.

        Args:
            limit: Maximum number of events to return
            event_types: Optional event types to filter

        Returns:
            List of RecentEvent objects
        """
        if self._kg is None:
            return []

        events: list[RecentEvent] = []

        for node_id, node_data in self._kg.graph.nodes(data=True):
            if node_data.get("node_type") != NodeType.EVENT.value:
                continue
            if event_types and node_data.get("event_type") not in event_types:
                continue

            event = RecentEvent(
                event_id=node_id,
                event_type=node_data.get("event_type", "unknown"),
                timestamp=node_data.get("timestamp", ""),
                description=node_data.get("description", ""),
            )
            events.append(event)

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    async def get_active_topics(self, min_mentions: int = 1) -> list[str]:
        """
        Get currently active topics.

        Args:
            min_mentions: Minimum mentions to be considered active

        Returns:
            List of active topic IDs
        """
        if self._kg is None:
            return []

        topic_mentions: dict[str, int] = {}

        for node_id, node_data in self._kg.graph.nodes(data=True):
            if node_data.get("node_type") != NodeType.TOPIC.value:
                continue

            # Count edges (mentions) to this topic
            mention_count = self._kg.graph.in_degree(node_id)
            topic_mentions[node_id] = mention_count

        # Filter by minimum mentions
        active = [
            topic_id
            for topic_id, count in topic_mentions.items()
            if count >= min_mentions
        ]

        return sorted(active, key=lambda t: topic_mentions[t], reverse=True)

    async def get_agent_neighbors(
        self, agent_id: str, relationship_type: Optional[str] = None
    ) -> list[tuple[str, float]]:
        """
        Get an agent's neighbors with optional relationship filter.

        Args:
            agent_id: Agent ID
            relationship_type: Optional relationship type to filter

        Returns:
            List of (neighbor_id, weight) tuples
        """
        if self._kg is None or agent_id not in self._kg.graph:
            return []

        neighbors: list[tuple[str, float]] = []

        for _, neighbor, edge_data in self._kg.graph.edges(agent_id, data=True):
            if relationship_type and edge_data.get("edge_type") != relationship_type:
                continue

            weight = edge_data.get("weight", 1.0)
            neighbors.append((neighbor, weight))

        return neighbors
