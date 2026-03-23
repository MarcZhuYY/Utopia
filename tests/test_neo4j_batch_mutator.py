"""Tests for Neo4jGraphMutator UNWIND batch processing.

Core test: 50 agents × 50 StanceChangeEvent → 1 mock transaction call.

Verifies CQRS architecture's core benefit:
- No matter how many agents produce how many events, only 1 Neo4j transaction
- Eliminates concurrent write lock contention
"""

from __future__ import annotations

import pytest

from utopia.layer2_world import (
    Neo4jBatchError,
    Neo4jGraphMutator,
    RelationshipCreateEvent,
    StanceChangeEvent,
)


class MockNeo4jDriver:
    """Mock Neo4j driver for testing."""

    def __init__(self):
        self.session_call_count = 0

    def session(self):
        self.session_call_count += 1
        return MockSession()


class MockSession:
    """Mock Neo4j session."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def begin_transaction(self):
        return MockTransactionContext()


class MockTransactionContext:
    """Mock transaction context manager."""

    async def __aenter__(self):
        return MockTransaction()

    async def __aexit__(self, *args):
        pass


class MockTransaction:
    """Mock Neo4j transaction that tracks calls."""

    run_count = 0
    commit_count = 0
    rollback_count = 0
    last_cypher = ""
    last_params = None

    @classmethod
    def reset(cls):
        cls.run_count = 0
        cls.commit_count = 0
        cls.rollback_count = 0
        cls.last_cypher = ""
        cls.last_params = None

    async def run(self, cypher, params=None):
        MockTransaction.run_count += 1
        MockTransaction.last_cypher = cypher
        MockTransaction.last_params = params

    async def commit(self):
        MockTransaction.commit_count += 1

    async def rollback(self):
        MockTransaction.rollback_count += 1


@pytest.fixture
def mock_mutator():
    """Create mutator with mocked driver."""
    mutator = Neo4jGraphMutator()
    mutator._driver = MockNeo4jDriver()  # type: ignore
    MockTransaction.reset()
    return mutator


@pytest.mark.asyncio
async def test_batch_mutator_single_transaction(mock_mutator):
    """
    Core test: 50 Agent × 1 StanceChangeEvent → 1 transaction.

    This verifies CQRS architecture's core benefit:
    - Regardless of how many agents/events, only 1 Neo4j transaction
    - Write lock contention eliminated
    """
    agent_count = 50
    topic = "TOPIC_Tariffs"

    # Each agent produces 1 StanceChangeEvent
    events = []
    for i in range(agent_count):
        agent_id = f"AGENT_{i:03d}"
        event = StanceChangeEvent(
            event_id=f"evt_{i}",
            tick_number=1,
            source_agent_id=agent_id,
            agent_id=agent_id,
            topic_id=topic,
            old_position=0.0,
            new_position=0.5 + (i * 0.01),  # Different new positions
            confidence=0.8,
        )
        events.append(event)

    # Execute batch write
    result = await mock_mutator.flush_events(events)

    # ===== Key assertions =====
    assert MockTransaction.commit_count == 1, (
        f"Expected 1 transaction, got {MockTransaction.commit_count}. "
        "CQRS batching failed - multiple transactions detected!"
    )

    assert result["processed"] == 50
    assert result["errors"] == 0

    # Verify UNWIND query was used
    assert "UNWIND" in MockTransaction.last_cypher
    # Verify all 50 events in single params batch
    assert MockTransaction.last_params is not None
    assert len(MockTransaction.last_params["events"]) == 50


@pytest.mark.asyncio
async def test_empty_events(mock_mutator):
    """Test with empty event list."""
    result = await mock_mutator.flush_events([])

    assert result["processed"] == 0
    assert result["errors"] == 0
    assert result["duration_ms"] == 0
    assert MockTransaction.commit_count == 0


@pytest.mark.asyncio
async def test_mixed_event_types(mock_mutator):
    """Test batch processing with multiple event types."""
    events = []

    # Add stance changes
    for i in range(10):
        events.append(
            StanceChangeEvent(
                event_id=f"stance_{i}",
                tick_number=1,
                source_agent_id=f"AGENT_{i:03d}",
                agent_id=f"AGENT_{i:03d}",
                topic_id="TOPIC_Test",
                old_position=0.0,
                new_position=0.5,
                confidence=0.8,
            )
        )

    # Add relationships
    for i in range(5):
        events.append(
            RelationshipCreateEvent(
                event_id=f"rel_{i}",
                tick_number=1,
                source_agent_id="AGENT_001",
                from_node_id="AGENT_001",
                to_node_id=f"AGENT_{i+1:03d}",
                relationship_type="TRUSTS",
                weight=0.8,
            )
        )

    result = await mock_mutator.flush_events(events)

    assert result["processed"] == 15
    assert MockTransaction.commit_count == 1


@pytest.mark.asyncio
async def test_transaction_rollback_on_error(mock_mutator):
    """Test rollback on error."""

    class FailingTransaction:
        async def run(self, cypher, params=None):
            raise RuntimeError("Neo4j connection failed")

        async def commit(self):
            pass

        async def rollback(self):
            MockTransaction.rollback_count += 1

    class FailingContext:
        async def __aenter__(self):
            return FailingTransaction()

        async def __aexit__(self, *args):
            pass

    class FailingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def begin_transaction(self):
            return FailingContext()

    class FailingDriver:
        def session(self):
            return FailingSession()

    mock_mutator._driver = FailingDriver()  # type: ignore
    MockTransaction.reset()

    events = [
        StanceChangeEvent(
            event_id="evt_fail",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Test",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )
    ]

    with pytest.raises(Neo4jBatchError):
        await mock_mutator.flush_events(events)

    # CRITICAL FIX: With retry mechanism (3 attempts), rollback is called 3 times
    # (once per failed attempt before retry)
    assert MockTransaction.rollback_count == 3


@pytest.mark.asyncio
async def test_mutator_singleton():
    """Test that Neo4jGraphMutator is a singleton."""
    mutator1 = Neo4jGraphMutator()
    mutator2 = Neo4jGraphMutator()

    assert mutator1 is mutator2


@pytest.mark.asyncio
async def test_mutator_stats(mock_mutator):
    """Test statistics tracking."""
    # Reset stats for clean test (singleton persists across tests)
    mock_mutator._transaction_count = 0
    mock_mutator._events_processed = 0

    events = [
        StanceChangeEvent(
            event_id=f"evt_{i}",
            tick_number=1,
            source_agent_id=f"AGENT_{i:03d}",
            agent_id=f"AGENT_{i:03d}",
            topic_id="TOPIC_Test",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )
        for i in range(10)
    ]

    await mock_mutator.flush_events(events)

    stats = mock_mutator.get_stats()
    assert stats["transaction_count"] == 1
    assert stats["events_processed"] == 10
    assert stats["avg_events_per_tx"] == 10.0

    # Second batch
    await mock_mutator.flush_events(events)

    stats = mock_mutator.get_stats()
    assert stats["transaction_count"] == 2
    assert stats["events_processed"] == 20
    assert stats["avg_events_per_tx"] == 10.0
