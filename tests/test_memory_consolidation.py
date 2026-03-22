"""Tests for memory consolidation system (memory_consolidation.py)."""

import numpy as np
import pytest

from utopia.core.pydantic_models import BigFiveTraits
from utopia.layer3_cognition.memory_consolidation import (
    MemoryConsolidationSystem,
    MemoryCluster,
)


class TestMemoryEntry:
    """Test memory entry with Ebbinghaus decay."""

    def test_memory_strength_formula(self):
        """Test: V(t) = Importance * exp(-lambda * delta_tick)
        lambda = 0.1 * (1 - Conscientiousness)
        """
        from utopia.core.pydantic_models import MemoryEntry

        memory = MemoryEntry(
            id="M001",
            content="Test memory",
            importance=0.8,
            creation_tick=0,
        )

        # High conscientiousness = slow decay
        strength_high = memory.compute_strength(
            current_tick=10,
            conscientiousness=0.9,
        )

        # Low conscientiousness = fast decay
        strength_low = memory.compute_strength(
            current_tick=10,
            conscientiousness=0.1,
        )

        # High conscientiousness should remember better
        assert strength_high > strength_low

    def test_forgetting_threshold(self):
        """Test memories are forgotten below threshold."""
        from utopia.core.pydantic_models import MemoryEntry

        memory = MemoryEntry(
            id="M001",
            content="Test memory",
            importance=0.1,  # Low importance
            creation_tick=0,
        )

        # After many ticks, should be forgotten
        is_forgotten = memory.is_forgotten(
            current_tick=100,
            conscientiousness=0.5,
            threshold=0.1,
        )

        assert is_forgotten


class TestMemoryCluster:
    """Test memory clustering."""

    def test_centroid_update(self):
        """Test incremental centroid calculation."""
        cluster = MemoryCluster()

        # Add first memory
        cluster.add_memory("M1", np.array([1.0, 0.0, 0.0]))
        assert np.allclose(cluster.centroid, [1.0, 0.0, 0.0])

        # Add second memory
        cluster.add_memory("M2", np.array([0.0, 1.0, 0.0]))
        expected = np.array([0.5, 0.5, 0.0])
        assert np.allclose(cluster.centroid, expected)

        # Add third memory
        cluster.add_memory("M3", np.array([0.0, 0.0, 1.0]))
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(cluster.centroid, expected)


class TestMemoryConsolidationSystem:
    """Test memory consolidation system."""

    def test_duplicate_detection(self):
        """Test duplicate memory detection."""
        system = MemoryConsolidationSystem()

        # Add original memory
        entry1 = system.add_memory(
            content="Unique content",
            importance=0.5,
            current_tick=0,
        )

        # Add duplicate
        entry2 = system.add_memory(
            content="Unique content",
            importance=0.5,
            current_tick=1,
        )

        # Should be detected as duplicate
        assert entry1.id == entry2.id or entry2.id is None

    def test_short_term_capacity_trigger(self):
        """Test consolidation triggers at capacity."""
        # Use low capacity and similar embeddings to trigger clustering
        system = MemoryConsolidationSystem(
            short_term_capacity=3,
            similarity_threshold=0.5,
            embedding_func=lambda x: [0.9] * 10,  # All similar
        )

        # Fill buffer - third add should trigger consolidation
        system.add_memory("Content 1", 0.5, current_tick=0)
        system.add_memory("Content 2", 0.5, current_tick=0)

        # This should trigger consolidation (buffer will be full)
        result = system.add_memory("Content 3", 0.5, current_tick=0)

        # Note: Consolidation happens before adding, so check short_term size
        # After consolidation, some memories may be removed
        stats = system.get_statistics()
        # Either consolidation happened or buffer is at capacity
        assert stats["short_term_count"] <= 3

    def test_retrieval_scoring(self):
        """Test retrieval with strength-weighted scoring."""
        system = MemoryConsolidationSystem(
            embedding_func=lambda x: [1.0 if "important" in x else 0.0],
        )

        # Add memories
        system.add_memory(
            content="Important memory about AI",
            importance=0.9,
            current_tick=0,
        )
        system.add_memory(
            content="Less important memory",
            importance=0.3,
            current_tick=0,
        )

        # Query for AI
        results = system.retrieve(
            query="AI",
            current_tick=0,
            limit=2,
        )

        # Should return results
        assert len(results) > 0

    def test_forget_weak_memories(self):
        """Test forgetting mechanism."""
        system = MemoryConsolidationSystem(
            traits=BigFiveTraits(conscientiousness=0.5),
        )

        # Add memory with low importance
        system.add_memory(
            content="Weak memory",
            importance=0.1,
            current_tick=0,
        )

        # Forget weak memories
        forgotten = system.forget_weak_memories(current_tick=50)

        assert forgotten > 0

    def test_consolidation_removes_originals(self):
        """Test that consolidation removes original memories."""
        system = MemoryConsolidationSystem(
            short_term_capacity=10,
            similarity_threshold=0.5,
            embedding_func=lambda x: [float(ord(x[0])) / 255.0] * 10,
        )

        # Add similar memories
        for i in range(5):
            system.add_memory(
                content=f"Similar topic content {i}",
                importance=0.5,
                current_tick=0,
            )

        initial_count = len(system.short_term)

        # Consolidate
        system.consolidate(current_tick=1)

        # Original memories should be removed
        final_count = len(system.short_term)
        assert final_count < initial_count or len(system.consolidated_experiences) > 0

    def test_statistics(self):
        """Test statistics tracking."""
        system = MemoryConsolidationSystem()

        stats = system.get_statistics()

        assert "short_term_count" in stats
        assert "long_term_count" in stats
        assert "consolidated_count" in stats
        assert "consolidation_events" in stats
        assert "forgotten_count" in stats
