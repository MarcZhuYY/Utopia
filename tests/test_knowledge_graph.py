"""Tests for Layer 2 (World Model) - Knowledge Graph."""

import pytest

from utopia.core.models import Entity, EntityType, SeedMaterial
from utopia.layer2_world.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    NodeType,
)


class TestKnowledgeGraph:
    """Test knowledge graph."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = KnowledgeGraph()
        assert len(graph.graph.nodes) == 0

    def test_add_agent(self):
        """Test adding agent node."""
        from utopia.core.models import Persona

        graph = KnowledgeGraph()
        persona = Persona(name="Test Agent", role="analyst")

        graph.add_agent(
            agent_id="A1",
            persona=persona,
            influence=0.7,
            domain_expertise=["tech"],
            base_stances={"topic1": 0.5},
        )

        assert "A1" in graph.graph.nodes
        node_data = graph.graph.nodes["A1"]
        assert node_data["node_type"] == NodeType.AGENT.value
        assert node_data["data"]["influence"] == 0.7

    def test_add_topic(self):
        """Test adding topic node."""
        from utopia.core.models import Topic

        graph = KnowledgeGraph()
        topic = Topic(
            id="T1",
            name="AI Regulation",
            description="Government regulation of AI",
            sensitivity=0.8,
        )

        graph.add_topic(topic)

        assert "T1" in graph.graph.nodes
        node_data = graph.graph.nodes["T1"]
        assert node_data["node_type"] == NodeType.TOPIC.value
        assert node_data["data"]["sensitivity"] == 0.8

    def test_graph_serialization(self):
        """Test graph to_dict."""
        from utopia.core.models import Persona, Topic

        graph = KnowledgeGraph()
        persona = Persona(name="Agent")

        graph.add_agent("A1", persona, 0.5, [], {})
        graph.add_topic(Topic(id="T1", name="Topic"))

        data = graph.to_dict()

        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 0


class TestKnowledgeGraphBuilder:
    """Test knowledge graph builder."""

    def test_build_from_seed(self):
        """Test building graph from seed material."""
        seed = SeedMaterial(
            raw_text="Test text",
        )
        seed.entities = [
            Entity(id="E1", name="Person A", type=EntityType.PERSON),
            Entity(id="E2", name="Person B", type=EntityType.PERSON),
        ]

        builder = KnowledgeGraphBuilder()
        graph = builder.build_from_seed(seed)

        # Should have 2 agent nodes
        agent_nodes = [
            n for n, d in graph.graph.nodes(data=True)
            if d.get("node_type") == NodeType.AGENT.value
        ]
        assert len(agent_nodes) == 2

    def test_build_with_relations(self):
        """Test building graph with relations."""
        from utopia.core.models import Relation, RelationType

        seed = SeedMaterial(raw_text="Test")
        seed.entities = [
            Entity(id="E1", name="A", type=EntityType.PERSON),
            Entity(id="E2", name="B", type=EntityType.PERSON),
        ]
        seed.relationships = [
            Relation(from_entity="E1", to_entity="E2", type=RelationType.KNOWS, strength=0.8),
        ]

        builder = KnowledgeGraphBuilder()
        graph = builder.build_from_seed(seed)

        assert len(graph.graph.edges) >= 1
