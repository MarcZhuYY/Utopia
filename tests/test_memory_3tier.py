"""Tests for 3-tier memory system.

Tests Hot/Warm/Cold memory architecture and Smart RAG retrieval.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from utopia.layer3_cognition.warm_memory_models import (
    ColdMemory,
    HotMemoryItem,
    WarmMemoryItem,
    RetrievedMemory,
)
from utopia.layer3_cognition.memory import MemorySystem3Tier


@pytest.fixture
def cold_memory():
    """Create a cold memory fixture."""
    return ColdMemory(
        persona_summary="Test agent persona",
        high_confidence_stances={"topic1": 0.95, "topic2": 0.90},
        static_knowledge={"domain": "finance"},
        core_goals=["goal1", "goal2"],
    )


@pytest.fixture
def memory_system(cold_memory):
    """Create a 3-tier memory system fixture."""
    return MemorySystem3Tier(
        agent_id="test_agent",
        cold_memory=cold_memory,
        hot_memory_maxlen=5,
        warm_memory_max_size=10,
    )


class TestThreeTierStructure:
    """Test the 3-tier memory structure."""

    def test_cold_memory_initialization(self, cold_memory):
        """Test Cold Memory initialization."""
        assert cold_memory.persona_summary == "Test agent persona"
        assert cold_memory.high_confidence_stances == {"topic1": 0.95, "topic2": 0.90}
        assert len(cold_memory.core_goals) == 2

    def test_memory_system_initialization(self, memory_system):
        """Test Memory System initialization."""
        assert memory_system.agent_id == "test_agent"
        assert len(memory_system.hot) == 0
        assert len(memory_system.warm) == 0
        assert len(memory_system.pending_embeddings) == 0

    def test_hot_memory_maxlen(self, memory_system):
        """Test Hot Memory maxlen enforcement."""
        # Add 7 items (maxlen=5)
        for i in range(7):
            memory_system.add_experience(
                content=f"Experience {i}",
                topic_id="topic",
                importance=0.5,
            )

        # Only last 5 should remain
        assert len(memory_system.hot) == 5
        # Should contain experiences 2-6
        contents = [item.content for item in memory_system.hot]
        assert "Experience 2" in contents
        assert "Experience 6" in contents
        assert "Experience 0" not in contents


class TestHotMemory:
    """Test Hot Memory functionality."""

    def test_add_to_hot_memory(self, memory_system):
        """Test adding experience creates Hot Memory item."""
        memory_system.add_experience(
            content="Test content",
            topic_id="test_topic",
            importance=0.5,
        )

        assert len(memory_system.hot) == 1
        item = memory_system.hot[0]
        assert item.content == "Test content"
        assert item.topic_id == "test_topic"
        assert item.importance == 0.5

    def test_hot_memory_keywords(self, memory_system):
        """Test Hot Memory keywords."""
        memory_system.add_experience(
            content="Test content",
            topic_id="test_topic",
            importance=0.5,
            keywords=["test", "content"],
        )

        item = memory_system.hot[0]
        assert item.keywords == ["test", "content"]

    def test_search_hot_memory_by_topic(self, memory_system):
        """Test Hot Memory search by topic."""
        memory_system.add_experience(
            content="Content A",
            topic_id="topic_A",
            importance=0.5,
        )
        memory_system.add_experience(
            content="Content B",
            topic_id="topic_B",
            importance=0.6,
        )

        results = memory_system._search_hot_memory("", topic_id="topic_A")
        assert len(results) == 1
        assert results[0].content == "Content A"

    def test_search_hot_memory_by_keyword(self, memory_system):
        """Test Hot Memory keyword search."""
        memory_system.add_experience(
            content="The quick brown fox",
            topic_id="topic",
            importance=0.5,
        )
        memory_system.add_experience(
            content="The lazy dog",
            topic_id="topic",
            importance=0.6,
        )

        results = memory_system._search_hot_memory("quick fox")
        assert len(results) == 1
        assert results[0].content == "The quick brown fox"

    def test_search_hot_memory_empty_query(self, memory_system):
        """Test Hot Memory search with empty query returns all."""
        memory_system.add_experience(
            content="Content A",
            topic_id="topic_A",
            importance=0.5,
        )
        memory_system.add_experience(
            content="Content B",
            topic_id="topic_B",
            importance=0.6,
        )

        results = memory_system._search_hot_memory("")
        assert len(results) == 2


class TestWarmMemory:
    """Test Warm Memory functionality."""

    def test_pending_embeddings_threshold(self, memory_system):
        """Test that only high-importance items enter pending queue."""
        # Add low importance (below threshold 0.3)
        memory_system.add_experience(
            content="Low importance",
            topic_id="topic",
            importance=0.2,
        )
        assert len(memory_system.pending_embeddings) == 0

        # Add high importance (at/above threshold)
        memory_system.add_experience(
            content="High importance",
            topic_id="topic",
            importance=0.3,
        )
        assert len(memory_system.pending_embeddings) == 1

    def test_on_batch_embeddings_received(self, memory_system):
        """Test receiving batch embedding results."""
        # Simulate receiving embeddings
        embeddings = [
            (
                "Test text",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "test_topic",
                    "importance": 0.7,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]

        memory_system.on_batch_embeddings_received(embeddings)

        assert len(memory_system.warm) == 1
        warm_item = memory_system.warm[0]
        assert warm_item.text == "Test text"
        assert warm_item.importance_score == 0.7
        assert warm_item.vector is not None
        assert warm_item.vector.shape == (768,)

    def test_warm_memory_size_limit(self, memory_system):
        """Test Warm Memory size limit enforcement."""
        # Add 15 warm items (max_size=10)
        for i in range(15):
            embeddings = [
                (
                    f"Text {i}",
                    np.random.randn(768).astype(np.float32),
                    {
                        "topic_id": "topic",
                        "importance": 0.5 + (i * 0.01),  # Varying importance
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ]
            memory_system.on_batch_embeddings_received(embeddings)

        # Should be limited to 10
        assert len(memory_system.warm) == 10

    def test_warm_memory_lru_eviction(self, memory_system):
        """Test Warm Memory LRU + importance eviction."""
        # Create items with varying importance
        for i in range(10):
            embeddings = [
                (
                    f"Text {i}",
                    np.random.randn(768).astype(np.float32),
                    {
                        "topic_id": "topic",
                        "importance": 0.1 if i < 5 else 0.9,  # First 5 are low importance
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ]
            memory_system.on_batch_embeddings_received(embeddings)

        # Add one more to trigger eviction
        embeddings = [
            (
                "New text",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "topic",
                    "importance": 0.95,  # High importance
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]
        memory_system.on_batch_embeddings_received(embeddings)

        # Should still be 10 items
        assert len(memory_system.warm) == 10

        # Low importance items should be evicted
        texts = [item.text for item in memory_system.warm]
        assert "New text" in texts
        # At least some low importance items should be gone
        assert sum(1 for t in texts if t.startswith("Text ") and int(t.split()[1]) < 5) < 5


class TestWarmMemorySearch:
    """Test Warm Memory vector search."""

    def test_search_warm_memory_by_vector(self, memory_system):
        """Test Warm Memory vector similarity search."""
        # Create items with different vectors
        vectors = [
            np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=np.float32),  # Similar to query
            np.array([0.0, 1.0, 0.0] + [0.0] * 765, dtype=np.float32),  # Different
            np.array([0.9, 0.1, 0.0] + [0.0] * 765, dtype=np.float32),  # Very similar
        ]

        for i, vec in enumerate(vectors):
            embeddings = [
                (
                    f"Text {i}",
                    vec,
                    {
                        "topic_id": "topic",
                        "importance": 0.5,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ]
            memory_system.on_batch_embeddings_received(embeddings)

        # Search with query vector similar to vector 0 and 2
        query = np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=np.float32)
        results = memory_system._search_warm_memory(query_vector=query, k=2)

        assert len(results) == 2
        # First result should be Text 0 or Text 2 (most similar)
        assert results[0].text in ["Text 0", "Text 2"]

    def test_search_warm_memory_with_topic_filter(self, memory_system):
        """Test Warm Memory search with topic filter."""
        vectors = [
            np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=np.float32),
            np.array([0.9, 0.1, 0.0] + [0.0] * 765, dtype=np.float32),
        ]

        for i, vec in enumerate(vectors):
            embeddings = [
                (
                    f"Text {i}",
                    vec,
                    {
                        "topic_id": f"topic_{i}",
                        "importance": 0.5,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ]
            memory_system.on_batch_embeddings_received(embeddings)

        query = np.array([1.0, 0.0, 0.0] + [0.0] * 765, dtype=np.float32)
        results = memory_system._search_warm_memory(
            query_vector=query, topic_id="topic_0", k=2
        )

        assert len(results) == 1
        assert results[0].text == "Text 0"

    def test_search_warm_memory_fallback(self, memory_system):
        """Test Warm Memory text fallback search (no vector)."""
        embeddings = [
            (
                "The quick brown fox",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "topic",
                    "importance": 0.5,
                    "timestamp": datetime.now().isoformat(),
                },
            ),
            (
                "The lazy dog sleeps",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "topic",
                    "importance": 0.6,
                    "timestamp": datetime.now().isoformat(),
                },
            ),
        ]
        memory_system.on_batch_embeddings_received(embeddings)

        results = memory_system._search_warm_memory(query_text="quick fox")
        assert len(results) == 1
        assert results[0].text == "The quick brown fox"


class TestSmartRAGRetrieval:
    """Test Smart RAG retrieval with Hot -> Warm priority."""

    def test_retrieve_hot_priority(self, memory_system):
        """Test that Hot Memory results are returned first."""
        # Add Hot Memory
        memory_system.add_experience(
            content="Hot memory content",
            topic_id="topic",
            importance=0.3,
        )

        # Add Warm Memory
        embeddings = [
            (
                "Warm memory content",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "topic",
                    "importance": 0.9,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]
        memory_system.on_batch_embeddings_received(embeddings)

        results = memory_system.retrieve_relevant(
            query="memory content",
            k=2,
        )

        assert len(results) == 2
        # Hot should come first
        assert results[0].source == "hot"
        assert results[1].source == "warm"

    def test_retrieve_early_return_on_hot_match(self, memory_system):
        """Test early return when Hot Memory has enough results."""
        # Add multiple Hot Memories
        for i in range(5):
            memory_system.add_experience(
                content=f"Hot memory {i}",
                topic_id="topic",
                importance=0.3,
            )

        # Request only 2 results
        results = memory_system.retrieve_relevant(query="Hot memory", k=2)

        # Should return only Hot results, no need to query Warm
        assert len(results) == 2
        assert all(r.source == "hot" for r in results)

    def test_retrieve_combines_hot_and_warm(self, memory_system):
        """Test retrieval combines Hot and Warm when needed."""
        # Add 1 Hot Memory
        memory_system.add_experience(
            content="Hot content",
            topic_id="topic",
            importance=0.3,
        )

        # Add 3 Warm Memories
        for i in range(3):
            embeddings = [
                (
                    f"Warm content {i}",
                    np.random.randn(768).astype(np.float32),
                    {
                        "topic_id": "topic",
                        "importance": 0.8,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            ]
            memory_system.on_batch_embeddings_received(embeddings)

        # Request 4 results
        results = memory_system.retrieve_relevant(
            query="content",
            k=4,
        )

        assert len(results) == 4
        # First should be Hot
        assert results[0].source == "hot"
        # Rest should be Warm
        assert all(r.source == "warm" for r in results[1:])


class TestColdMemory:
    """Test Cold Memory functionality."""

    def test_cold_memory_to_system_prompt(self, cold_memory):
        """Test Cold Memory to system prompt conversion."""
        prompt = cold_memory.to_system_prompt_section()

        assert "Core Identity" in prompt
        assert "Test agent persona" in prompt
        assert "High Confidence Beliefs" in prompt
        assert "topic1: +0.95" in prompt
        assert "Core Goals" in prompt
        assert "goal1" in prompt


class TestMemoryStats:
    """Test memory statistics."""

    def test_get_stats(self, memory_system):
        """Test memory statistics."""
        # Add some data
        memory_system.add_experience(
            content="Test",
            topic_id="topic",
            importance=0.5,
        )

        stats = memory_system.get_stats()

        assert stats["agent_id"] == "test_agent"
        assert stats["hot_memory_size"] == 1
        assert stats["warm_memory_size"] == 0
        assert stats["pending_embeddings"] == 1
        assert stats["cold_memory_size"] == 4  # 2 stances + 1 knowledge + 2 goals

    def test_len(self, memory_system):
        """Test __len__ method."""
        assert len(memory_system) == 0

        memory_system.add_experience(
            content="Test",
            topic_id="topic",
            importance=0.5,
        )
        assert len(memory_system) == 1

        # Add Warm Memory
        embeddings = [
            (
                "Warm text",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "topic",
                    "importance": 0.5,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]
        memory_system.on_batch_embeddings_received(embeddings)
        assert len(memory_system) == 2


class TestClear:
    """Test memory clearing."""

    def test_clear(self, memory_system):
        """Test clearing all memories."""
        memory_system.add_experience(
            content="Test",
            topic_id="topic",
            importance=0.5,
        )

        embeddings = [
            (
                "Warm text",
                np.random.randn(768).astype(np.float32),
                {
                    "topic_id": "topic",
                    "importance": 0.5,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ]
        memory_system.on_batch_embeddings_received(embeddings)

        assert len(memory_system) == 2

        memory_system.clear()

        assert len(memory_system) == 0
        assert len(memory_system.hot) == 0
        assert len(memory_system.warm) == 0
        assert len(memory_system.pending_embeddings) == 0
