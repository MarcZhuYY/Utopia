"""Relationship network for agents.

Tracks trust, influence, and familiarity between agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from utopia.core.config import WorldRules


@dataclass
class RelationshipEdge:
    """Represents an edge in the relationship graph.

    Attributes:
        trust: Trust level (-1 to 1)
        influence_weight: How much this agent influences the other (0 to 1)
        familiarity: How well they know each other (0 to 1)
        last_interaction: Timestamp of last interaction
    """

    trust: float = 0.0
    influence_weight: float = 0.5
    familiarity: float = 0.0
    last_interaction: datetime = field(default_factory=datetime.now)

    @classmethod
    def default(cls) -> RelationshipEdge:
        """Create default edge.

        Returns:
            RelationshipEdge: Default edge with neutral values
        """
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trust": self.trust,
            "influence_weight": self.influence_weight,
            "familiarity": self.familiarity,
            "last_interaction": self.last_interaction.isoformat(),
        }


@dataclass
class RelationshipDelta:
    """Change to apply to a relationship edge.

    Attributes:
        trust_change: Change in trust (-1 to 1)
        influence_change: Change in influence weight (-1 to 1)
    """

    trust_change: float = 0.0
    influence_change: float = 0.0


class RelationshipMap:
    """Map of relationships between agents.

    Stores as adjacency list: {agent_id: {other_agent_id: RelationshipEdge}}
    """

    def __init__(self):
        """Initialize empty relationship map."""
        self.relations: dict[str, dict[str, RelationshipEdge]] = {}

    def get(self, agent_a: str, agent_b: str) -> RelationshipEdge:
        """Get relationship between two agents.

        Args:
            agent_a: First agent ID
            agent_b: Second agent ID

        Returns:
            RelationshipEdge: Relationship (default if none exists)
        """
        if agent_a not in self.relations:
            return RelationshipEdge.default()
        return self.relations[agent_a].get(agent_b, RelationshipEdge.default())

    def set(self, agent_a: str, agent_b: str, edge: RelationshipEdge) -> None:
        """Set relationship between two agents.

        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            edge: Relationship edge
        """
        if agent_a not in self.relations:
            self.relations[agent_a] = {}
        self.relations[agent_a][agent_b] = edge

    def update(
        self,
        agent_a: str,
        agent_b: str,
        delta: RelationshipDelta,
    ) -> None:
        """Update relationship by applying a delta.

        Args:
            agent_a: First agent ID
            agent_b: Second agent ID
            delta: Change to apply
        """
        edge = self.get(agent_a, agent_b)

        # Apply changes with clamping
        edge.trust = max(-1.0, min(1.0, edge.trust + delta.trust_change))
        edge.influence_weight = max(
            0.0, min(1.0, edge.influence_weight + delta.influence_change)
        )
        edge.familiarity = min(1.0, edge.familiarity + 0.1)
        edge.last_interaction = datetime.now()

        self.set(agent_a, agent_b, edge)

    def initialize_agent(self, agent_id: str) -> None:
        """Initialize empty relations for new agent.

        Args:
            agent_id: Agent ID
        """
        if agent_id not in self.relations:
            self.relations[agent_id] = {}

    def get_all_relations(self, agent_id: str) -> dict[str, RelationshipEdge]:
        """Get all relations for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            dict[str, RelationshipEdge]: Map of other_id -> edge
        """
        return self.relations.get(agent_id, {})

    def get_trusted_agents(self, agent_id: str, threshold: float = 0.3) -> list[str]:
        """Get agents that this agent trusts.

        Args:
            agent_id: Agent ID
            threshold: Trust threshold

        Returns:
            list[str]: List of trusted agent IDs
        """
        relations = self.get_all_relations(agent_id)
        return [oid for oid, edge in relations.items() if edge.trust >= threshold]

    def get_influential_agents(
        self,
        agent_id: str,
        threshold: float = 0.3,
    ) -> list[str]:
        """Get agents that this agent is influenced by.

        Args:
            agent_id: Agent ID
            threshold: Influence threshold

        Returns:
            list[str]: List of influential agent IDs
        """
        relations = self.get_all_relations(agent_id)
        return [oid for oid, edge in relations.items() if edge.influence_weight >= threshold]

    def build_complete_graph(
        self,
        agent_ids: list[str],
        base_trust: float = 0.0,
    ) -> None:
        """Build complete graph with default relationships.

        Args:
            agent_ids: List of agent IDs
            base_trust: Base trust level for all pairs
        """
        for agent_id in agent_ids:
            self.initialize_agent(agent_id)

        # Create edges between all pairs
        for i, ida in enumerate(agent_ids):
            for idb in agent_ids[i + 1 :]:
                edge = RelationshipEdge(
                    trust=base_trust,
                    influence_weight=0.5,
                    familiarity=0.0,
                )
                self.set(ida, idb, edge)
                # Bidirectional
                self.set(idb, ida, RelationshipEdge(
                    trust=base_trust,
                    influence_weight=0.5,
                    familiarity=0.0,
                ))

    def to_dict(self) -> dict[str, Any]:
        """Serialize relationship map.

        Returns:
            dict: Serialized data
        """
        result = {}
        for agent_id, relations in self.relations.items():
            result[agent_id] = {oid: edge.to_dict() for oid, edge in relations.items()}
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipMap:
        """Load relationship map from dictionary.

        Args:
            data: Serialized data

        Returns:
            RelationshipMap: Loaded map
        """
        rmap = cls()
        for agent_id, relations in data.items():
            for other_id, edge_data in relations.items():
                edge = RelationshipEdge(
                    trust=edge_data.get("trust", 0.0),
                    influence_weight=edge_data.get("influence_weight", 0.5),
                    familiarity=edge_data.get("familiarity", 0.0),
                    last_interaction=datetime.fromisoformat(
                        edge_data.get("last_interaction", datetime.now().isoformat())
                    ),
                )
                rmap.set(agent_id, other_id, edge)
        return rmap
