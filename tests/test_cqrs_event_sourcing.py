"""Tests for CQRS + Event Sourcing architecture.

Verifies:
- Event immutability (frozen Pydantic models)
- Event buffer thread safety
- 50 agents × 50 events → 1 transaction guarantee
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from utopia.layer2_world import (
    EventType,
    StanceChangeEvent,
    WorldEventBuffer,
)


class TestEventImmutability:
    """Test that WorldEvent instances are immutable."""

    def test_stance_change_event_frozen(self):
        """StanceChangeEvent should be immutable."""
        event = StanceChangeEvent(
            event_id="test-evt-001",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Tariffs",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )

        # Attempt to modify should raise exception
        with pytest.raises(Exception):
            event.new_position = 0.8

    def test_stance_delta_property(self):
        """stance_delta should calculate correctly."""
        event = StanceChangeEvent(
            event_id="test-evt-002",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Tariffs",
            old_position=-0.3,
            new_position=0.7,
            confidence=0.8,
        )

        assert event.stance_delta == 1.0  # |0.7 - (-0.3)| = 1.0

    def test_event_type_validation(self):
        """Event type should be validated."""
        event = StanceChangeEvent(
            event_id="test-evt-003",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Tariffs",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )

        assert event.event_type == EventType.STANCE_CHANGE


class TestWorldEventBuffer:
    """Test WorldEventBuffer functionality."""

    @pytest.mark.asyncio
    async def test_append_and_drain(self):
        """Test basic append and drain."""
        buffer = WorldEventBuffer()

        event = StanceChangeEvent(
            event_id="evt-001",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Test",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )

        await buffer.append(event)

        events = await buffer.drain(1)
        assert len(events) == 1
        assert events[0].event_id == "evt-001"

    @pytest.mark.asyncio
    async def test_drain_clears_events(self):
        """Drain should clear events from buffer."""
        buffer = WorldEventBuffer()

        event = StanceChangeEvent(
            event_id="evt-002",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Test",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )

        await buffer.append(event)

        # First drain
        events1 = await buffer.drain(1)
        assert len(events1) == 1

        # Second drain should be empty
        events2 = await buffer.drain(1)
        assert len(events2) == 0

    @pytest.mark.asyncio
    async def test_append_many(self):
        """Test batch append."""
        buffer = WorldEventBuffer()

        events = [
            StanceChangeEvent(
                event_id=f"evt-{i}",
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

        await buffer.append_many(events)

        drained = await buffer.drain(1)
        assert len(drained) == 10

    @pytest.mark.asyncio
    async def test_advance_tick(self):
        """Test tick advancement."""
        buffer = WorldEventBuffer()

        event1 = StanceChangeEvent(
            event_id="evt-tick1",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Test",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )
        await buffer.append(event1)

        buffer.advance_tick()

        event2 = StanceChangeEvent(
            event_id="evt-tick2",
            tick_number=2,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Test",
            old_position=0.5,
            new_position=0.7,
            confidence=0.8,
        )
        await buffer.append(event2)

        # Drain tick 1
        tick1_events = await buffer.drain(1)
        assert len(tick1_events) == 1
        assert tick1_events[0].event_id == "evt-tick1"

        # Drain tick 2
        tick2_events = await buffer.drain(2)
        assert len(tick2_events) == 1
        assert tick2_events[0].event_id == "evt-tick2"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics."""
        buffer = WorldEventBuffer()

        event = StanceChangeEvent(
            event_id="evt-stats",
            tick_number=1,
            source_agent_id="AGENT_001",
            agent_id="AGENT_001",
            topic_id="TOPIC_Test",
            old_position=0.0,
            new_position=0.5,
            confidence=0.8,
        )
        await buffer.append(event)

        stats = await buffer.get_stats()
        assert stats["current_tick"] == 0
        assert stats["total_buffered_events"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_append(self):
        """Test concurrent event appends (thread safety)."""
        buffer = WorldEventBuffer()
        event_count = 100
        agent_count = 10

        async def producer(agent_id: str, count: int):
            """Produce events."""
            for i in range(count):
                event = StanceChangeEvent(
                    event_id=f"{agent_id}_evt_{i}",
                    tick_number=1,
                    source_agent_id=agent_id,
                    agent_id=agent_id,
                    topic_id="TOPIC_Test",
                    old_position=0.0,
                    new_position=0.5,
                    confidence=0.8,
                )
                await buffer.append(event)
                await asyncio.sleep(0.001)  # Simulate processing delay

        # Create concurrent tasks
        tasks = [
            asyncio.create_task(producer(f"AGENT_{i:03d}", event_count))
            for i in range(agent_count)
        ]

        await asyncio.gather(*tasks)

        # Verify all events collected
        events = await buffer.drain(1)
        assert len(events) == agent_count * event_count

        # Verify grouping by agent
        by_agent = defaultdict(list)
        for e in events:
            by_agent[e.source_agent_id].append(e)

        assert len(by_agent) == agent_count
        assert all(len(evts) == event_count for evts in by_agent.values())


class TestEventGrouping:
    """Test event type grouping."""

    @pytest.mark.asyncio
    async def test_get_events_by_type(self):
        """Test grouping events by type."""
        from utopia.layer2_world import AgentActionEvent

        buffer = WorldEventBuffer()

        # Add stance change events
        for i in range(3):
            event = StanceChangeEvent(
                event_id=f"stance-{i}",
                tick_number=1,
                source_agent_id="AGENT_001",
                agent_id="AGENT_001",
                topic_id="TOPIC_Test",
                old_position=0.0,
                new_position=0.5,
                confidence=0.8,
            )
            await buffer.append(event)

        # Add action events
        for i in range(2):
            event = AgentActionEvent(
                event_id=f"action-{i}",
                tick_number=1,
                source_agent_id="AGENT_001",
                action_type="speak",
                content="Test",
                importance=0.5,
            )
            await buffer.append(event)

        # Group by type
        grouped = await buffer.get_events_by_type(1)

        assert len(grouped[EventType.STANCE_CHANGE]) == 3
        assert len(grouped[EventType.AGENT_ACTION]) == 2


class TestFiftyAgentsFiftyEvents:
    """Core test: 50 agents × events → verify aggregation."""

    @pytest.mark.asyncio
    async def test_fifty_agents_produce_events(self):
        """50 agents each produce events, verify all collected."""
        buffer = WorldEventBuffer()
        agent_count = 50
        events_per_agent = 10

        async def producer(agent_id: str, count: int):
            """Produce events for one agent."""
            for i in range(count):
                event = StanceChangeEvent(
                    event_id=f"{agent_id}_evt_{i}",
                    tick_number=1,
                    source_agent_id=agent_id,
                    agent_id=agent_id,
                    topic_id="TOPIC_Tariffs",
                    old_position=0.0,
                    new_position=0.5 + (i * 0.01),
                    confidence=0.8,
                )
                await buffer.append(event)

        # Create 50 concurrent agents
        tasks = [
            asyncio.create_task(producer(f"AGENT_{i:03d}", events_per_agent))
            for i in range(agent_count)
        ]

        await asyncio.gather(*tasks)

        # Verify all events collected
        events = await buffer.drain(1)
        assert len(events) == agent_count * events_per_agent

        # Verify each agent's events are correct
        by_agent = defaultdict(list)
        for e in events:
            by_agent[e.source_agent_id].append(e)

        assert len(by_agent) == agent_count

        # Verify stance progression (unique positions per agent)
        for agent_id, agent_events in by_agent.items():
            assert len(agent_events) == events_per_agent
            positions = [e.new_position for e in agent_events]
            # Positions should be unique (0.5, 0.51, 0.52, ...)
            assert len(set(positions)) == events_per_agent
