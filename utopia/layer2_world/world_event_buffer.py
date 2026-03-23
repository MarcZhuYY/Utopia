"""World event buffer for CQRS write-side.

Agents append events by tick, Neo4jGraphMutator consumes them in batches.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from utopia.layer2_world.world_events import EventType, WorldEvent

logger = logging.getLogger(__name__)


class WorldEventBuffer:
    """World event buffer - CQRS write-side.

    Responsibilities:
        1. Receive events from all Agents (concurrent-safe)
        2. Accumulate events by Tick
        3. Support batch drain for Neo4jGraphMutator consumption

    Example:
        >>> buffer = WorldEventBuffer()
        >>> await buffer.append(event)  # From Agent
        >>> events = await buffer.drain(tick_number=1)  # From Mutator
    """

    def __init__(self, max_size: int = 10000, timeout_seconds: float = 5.0):
        """Initialize event buffer.

        Args:
            max_size: Maximum events per tick (soft limit, warns on overflow)
            timeout_seconds: Timeout for lock acquisition
        """
        self._tick_events: dict[int, list["WorldEvent"]] = defaultdict(list)
        self._event_count = 0
        self._current_tick = 0
        self._tick_lock = asyncio.Lock()
        self._timeout_seconds = timeout_seconds
        self._max_size = max_size

    async def append(self, event: "WorldEvent") -> None:
        """Append a single event.

        Args:
            event: Domain event to buffer
        """
        async with self._tick_lock:
            self._tick_events[event.tick_number].append(event)
            self._event_count += 1

    async def append_many(self, events: list["WorldEvent"]) -> None:
        """Batch append events (single lock acquisition for efficiency).

        Args:
            events: List of domain events to buffer
        """
        async with self._tick_lock:
            for event in events:
                self._tick_events[event.tick_number].append(event)
                self._event_count += 1

    async def drain(self, tick_number: Optional[int] = None) -> list["WorldEvent"]:
        """Extract events for a specific tick.

        Args:
            tick_number: Specific tick to drain, None for current tick

        Returns:
            List of all events for the specified tick
        """
        target_tick = tick_number if tick_number is not None else self._current_tick

        async with self._tick_lock:
            events = self._tick_events.get(target_tick, [])
            if target_tick in self._tick_events:
                del self._tick_events[target_tick]
            return events

    async def drain_all(self) -> list["WorldEvent"]:
        """Extract all pending events.

        Returns:
            List of all buffered events
        """
        async with self._tick_lock:
            all_events: list["WorldEvent"] = []
            for events in self._tick_events.values():
                all_events.extend(events)
            self._tick_events.clear()
            return all_events

    def advance_tick(self) -> None:
        """Advance to next tick."""
        self._current_tick += 1

    async def get_stats(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dict with current_tick, pending_ticks, total_buffered_events
        """
        async with self._tick_lock:
            return {
                "current_tick": self._current_tick,
                "pending_ticks": len(self._tick_events),
                "total_buffered_events": sum(
                    len(e) for e in self._tick_events.values()
                ),
            }

    async def get_events_by_type(self, tick_number: int) -> dict["EventType", list["WorldEvent"]]:
        """Group events by type.

        Args:
            tick_number: Tick to group

        Returns:
            Dict mapping event types to event lists
        """
        from utopia.layer2_world.world_events import EventType

        async with self._tick_lock:
            events = self._tick_events.get(tick_number, [])
            grouped: dict[EventType, list["WorldEvent"]] = defaultdict(list)
            for event in events:
                grouped[event.event_type].append(event)
            return dict(grouped)
