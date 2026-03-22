"""Tests for Layer 4 (Social Interaction) - Relationships."""

import pytest
from datetime import datetime

from utopia.layer4_social.relationships import (
    RelationshipMap,
    RelationshipEdge,
    RelationshipDelta,
)


class TestRelationshipMap:
    """Test relationship map."""

    def test_empty_map(self):
        """Test empty map."""
        rmap = RelationshipMap()
        edge = rmap.get("A", "B")
        assert edge.trust == 0.0
        assert edge.influence_weight == 0.5

    def test_set_and_get(self):
        """Test setting and getting relationship."""
        rmap = RelationshipMap()
        edge = RelationshipEdge(trust=0.7, influence_weight=0.8)

        rmap.set("A", "B", edge)

        retrieved = rmap.get("A", "B")
        assert retrieved.trust == 0.7
        assert retrieved.influence_weight == 0.8

    def test_update(self):
        """Test updating relationship."""
        rmap = RelationshipMap()
        rmap.set("A", "B", RelationshipEdge(trust=0.3))

        delta = RelationshipDelta(trust_change=0.2)
        rmap.update("A", "B", delta)

        edge = rmap.get("A", "B")
        assert edge.trust == 0.5

    def test_trust_clamping(self):
        """Test trust stays within bounds."""
        rmap = RelationshipMap()
        rmap.set("A", "B", RelationshipEdge(trust=0.8))

        delta = RelationshipDelta(trust_change=0.5)
        rmap.update("A", "B", delta)

        edge = rmap.get("A", "B")
        assert edge.trust <= 1.0
        assert edge.trust >= -1.0

    def test_build_complete_graph(self):
        """Test building complete graph."""
        agents = ["A", "B", "C"]
        rmap = RelationshipMap()
        rmap.build_complete_graph(agents, base_trust=0.5)

        # Check all pairs exist
        for a in agents:
            for b in agents:
                if a != b:
                    edge = rmap.get(a, b)
                    assert edge.trust == 0.5

    def test_serialization(self):
        """Test serialization."""
        rmap = RelationshipMap()
        rmap.set("A", "B", RelationshipEdge(trust=0.7))

        data = rmap.to_dict()
        loaded = RelationshipMap.from_dict(data)

        edge = loaded.get("A", "B")
        assert edge.trust == 0.7
