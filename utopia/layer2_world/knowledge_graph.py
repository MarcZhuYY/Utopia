"""Knowledge Graph implementation using NetworkX.

Implements the graph structure from the design document:
- Nodes: Agent, Topic, Event, Opinion
- Edges: INFLUENCES, SUPPORTS/OPPOSES, KNOWS, PARTICIPATED_IN, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import networkx as nx

from utopia.core.models import (
    Entity,
    Event,
    ExternalEvent,
    Persona,
    Relation,
    RelationType,
    SeedMaterial,
    Topic as TopicModel,
)


class NodeType(str, Enum):
    """Node types in the knowledge graph."""

    AGENT = "Agent"
    TOPIC = "Topic"
    EVENT = "Event"
    OPINION = "Opinion"


class EdgeType(str, Enum):
    """Edge types in the knowledge graph."""

    INFLUENCES = "INFLUENCES"
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    NEUTRAL = "NEUTRAL"
    KNOWS = "KNOWS"
    TRUSTS = "TRUSTS"
    DISTRUSTS = "DISTRUSTS"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    CAUSED = "CAUSED"
    ENABLED = "ENABLED"
    PREVENTED = "PREVENTED"
    ABOUT = "ABOUT"
    EXPRESSED = "EXPRESSED"


@dataclass
class GraphNode:
    """Wrapper for graph nodes with metadata."""

    id: str
    node_type: NodeType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "data": self.data,
        }


@dataclass
class GraphEdge:
    """Wrapper for graph edges with metadata."""

    from_node: str
    to_node: str
    edge_type: EdgeType
    weight: float = 1.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_node,
            "to": self.to_node,
            "type": self.edge_type.value,
            "weight": self.weight,
            "data": self.data,
        }


class KnowledgeGraph:
    """Knowledge graph representation of the simulation world.

    Uses NetworkX MultiDiGraph for underlying storage.
    """

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._node_cache: dict[str, dict[str, Any]] = {}

    def add_agent(
        self,
        agent_id: str,
        persona: Persona,
        influence: float,
        domain_expertise: list[str],
        base_stances: dict[str, float],
        memory_capacity: int = 100,
    ) -> None:
        """Add an agent node.

        Args:
            agent_id: Unique agent ID
            persona: Persona object
            influence: Influence score (0-1)
            domain_expertise: List of expertise areas
            base_stances: Base stances on topics
            memory_capacity: Memory capacity
        """
        self.graph.add_node(
            agent_id,
            node_type=NodeType.AGENT.value,
            data={
                "persona_summary": f"{persona.name} ({persona.role})",
                "influence": influence,
                "domain_expertise": domain_expertise,
                "base_stances": base_stances,
                "memory_capacity": memory_capacity,
                "traits": persona.traits,
            },
        )

    def add_topic(self, topic: TopicModel) -> None:
        """Add a topic node.

        Args:
            topic: Topic object
        """
        self.graph.add_node(
            topic.id,
            node_type=NodeType.TOPIC.value,
            data={
                "name": topic.name,
                "description": topic.description,
                "sensitivity": topic.sensitivity,
            },
        )

    def add_event(self, event: Event | ExternalEvent, source_agent: Optional[str] = None) -> None:
        """Add an event node.

        Args:
            event: Event or ExternalEvent object
            source_agent: ID of agent that triggered this event
        """
        if isinstance(event, ExternalEvent):
            self.graph.add_node(
                event.id,
                node_type=NodeType.EVENT.value,
                data={
                    "description": event.description,
                    "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else event.timestamp,
                    "importance": event.importance,
                    "source_agent": source_agent or "external",
                    "topic_id": event.topic_id,
                    "external": True,
                },
            )
        else:
            self.graph.add_node(
                event.id,
                node_type=NodeType.EVENT.value,
                data={
                    "description": event.description,
                    "timestamp": event.timestamp,
                    "importance": event.importance,
                    "source_agent": source_agent,
                    "participants": event.participants,
                    "external": False,
                },
            )

    def add_opinion(
        self,
        opinion_id: str,
        content: str,
        stance: float,
        topic_id: str,
        agent_id: str,
        confidence: float = 0.5,
    ) -> None:
        """Add an opinion node.

        Args:
            opinion_id: Unique opinion ID
            content: Opinion text
            stance: Stance position (-1 to 1)
            topic_id: Related topic ID
            agent_id: Agent who expressed this
            confidence: Confidence level
        """
        self.graph.add_node(
            opinion_id,
            node_type=NodeType.OPINION.value,
            data={
                "content": content,
                "stance": stance,
                "topic_id": topic_id,
                "confidence": confidence,
                "agent_id": agent_id,
            },
        )

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        **attrs,
    ) -> None:
        """Add an edge between nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            edge_type: Type of edge
            weight: Edge weight
            **attrs: Additional attributes
        """
        self.graph.add_edge(
            from_id,
            to_id,
            edge_type=edge_type.value,
            weight=weight,
            **attrs,
        )

    def get_agent_stance(self, agent_id: str, topic_id: str) -> Optional[float]:
        """Get agent's current stance on a topic.

        Args:
            agent_id: Agent ID
            topic_id: Topic ID

        Returns:
            Optional[float]: Stance position or None
        """
        # Look for OPINION node from this agent about this topic
        for _, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.OPINION.value:
                if data.get("agent_id") == agent_id and data.get("topic_id") == topic_id:
                    return data.get("stance")
        return None

    def get_topic_agents(self, topic_id: str) -> list[str]:
        """Get all agents related to a topic.

        Args:
            topic_id: Topic ID

        Returns:
            list[str]: List of agent IDs
        """
        agents = []
        for agent_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.AGENT.value:
                # Check if agent has any stance on this topic
                if self.get_agent_stance(agent_id, topic_id) is not None:
                    agents.append(agent_id)
        return agents

    def get_influence_network(self, agent_id: str, max_depth: int = 2) -> dict[str, float]:
        """Get agents influenced by this agent.

        Args:
            agent_id: Starting agent ID
            max_depth: Maximum propagation depth

        Returns:
            dict[str, float]: Map of agent_id -> influence weight
        """
        influenced = {}
        try:
            edges = nx.bfs_edges(self.graph, agent_id, depth_limit=max_depth)
            for from_node, to_node in edges:
                edge_dict = self.graph.get_edge_data(from_node, to_node)
                if not edge_dict:
                    continue
                for edge_data in edge_dict.values():
                    if edge_data.get("edge_type") == EdgeType.INFLUENCES.value:
                        weight = edge_data.get("weight", 1.0)
                        influenced[to_node] = influenced.get(to_node, 0) + weight
        except nx.NetworkXError:
            pass
        return influenced

    def to_dict(self) -> dict[str, Any]:
        """Export graph as dictionary.

        Returns:
            dict: Serialized graph
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **data})

        edges = []
        for from_node, to_node, data in self.graph.edges(data=True):
            edges.append({"from": from_node, "to": to_node, **data})

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def save_json(self, filepath: str) -> None:
        """Save graph to JSON file.

        Args:
            filepath: Output file path
        """
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        """Load graph from dictionary.

        Args:
            data: Serialized graph data

        Returns:
            KnowledgeGraph: Loaded graph
        """
        graph = cls()
        for node in data.get("nodes", []):
            node = dict(node)  # avoid mutating input
            node_id = node.pop("id")
            graph.graph.add_node(node_id, **node)
        for edge in data.get("edges", []):
            edge = dict(edge)  # avoid mutating input
            from_id = edge.pop("from")
            to_id = edge.pop("to")
            graph.graph.add_edge(from_id, to_id, **edge)
        return graph


class KnowledgeGraphBuilder:
    """Builder for constructing knowledge graph from seed material."""

    def __init__(self):
        """Initialize builder."""
        self.graph = KnowledgeGraph()

    def build_from_seed(self, seed: SeedMaterial) -> KnowledgeGraph:
        """Build complete knowledge graph from seed material.

        Args:
            seed: Parsed seed material

        Returns:
            KnowledgeGraph: Built graph
        """
        self.graph = KnowledgeGraph()

        # 1. Create agent nodes from entities
        for entity in seed.entities:
            if entity.type.value in ["person", "org"]:
                self._create_agent_node(entity)

        # 2. Create topic nodes from events
        topics = self._extract_topics(seed.events)
        for topic in topics:
            self.graph.add_topic(topic)

        # 3. Create initial relationship edges
        for relation in seed.relationships:
            self._create_edge(relation)

        # 4. Set initial stances
        for entity in seed.entities:
            if entity.initial_stance:
                self._set_initial_stances(entity.id, entity.initial_stance)

        return self.graph

    def _create_agent_node(self, entity: Entity) -> None:
        """Create agent node from entity.

        Args:
            entity: Source entity
        """
        persona = Persona(
            name=entity.name,
            role=entity.attributes.get("role", "analyst"),
            expertise=[entity.attributes.get("sector", "general")],
            influence_base=entity.influence_score,
        )

        self.graph.add_agent(
            agent_id=entity.id,
            persona=persona,
            influence=entity.influence_score,
            domain_expertise=persona.expertise,
            base_stances=entity.initial_stance,
        )

    def _extract_topics(self, events: list[Event]) -> list[TopicModel]:
        """Extract topics from events.

        Args:
            events: List of events

        Returns:
            list[TopicModel]: Extracted topics
        """
        # For MVP, extract simple topics from event descriptions
        # Phase 2: Use LLM to abstract topics
        topics = []
        topic_names = set()

        for event in events:
            # Simple extraction: first 5 words as topic name
            words = event.description.split()[:5]
            topic_name = " ".join(words)

            if topic_name and topic_name not in topic_names:
                topic_names.add(topic_name)
                topic = TopicModel(
                    id=f"T{len(topics)}",
                    name=topic_name,
                    description=event.description,
                    sensitivity=event.importance,
                )
                topics.append(topic)

        return topics

    def _create_edge(self, relation: Relation) -> None:
        """Create edge from relation.

        Args:
            relation: Source relation
        """
        # Map relation type to edge type
        type_mapping = {
            RelationType.WORKS_FOR: EdgeType.INFLUENCES,
            RelationType.OPPOSES: EdgeType.OPPOSES,
            RelationType.ALLIES: EdgeType.SUPPORTS,
            RelationType.COMPETES: EdgeType.OPPOSES,
            RelationType.BELONGS_TO: EdgeType.INFLUENCES,
            RelationType.PART_OF: EdgeType.INFLUENCES,
            RelationType.KNOWS: EdgeType.KNOWS,
            RelationType.TRUSTS: EdgeType.TRUSTS,
            RelationType.DISTRUSTS: EdgeType.DISTRUSTS,
        }

        edge_type = type_mapping.get(relation.type, EdgeType.KNOWS)

        self.graph.add_edge(
            from_id=relation.from_entity,
            to_id=relation.to_entity,
            edge_type=edge_type,
            weight=relation.strength,
        )

    def _set_initial_stances(self, entity_id: str, stances: dict[str, float]) -> None:
        """Set initial stances for an entity.

        Args:
            entity_id: Entity ID
            stances: Dict of topic_id -> stance position
        """
        for topic_id, position in stances.items():
            opinion_id = f"{entity_id}_opinion_{topic_id}"
            self.graph.add_opinion(
                opinion_id=opinion_id,
                content=f"Initial stance on {topic_id}",
                stance=position,
                topic_id=topic_id,
                agent_id=entity_id,
                confidence=0.5,
            )
