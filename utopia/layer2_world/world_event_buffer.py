"""World event buffer for CQRS write-side.

Implements lock-free event collection using asyncio.Queue.
Agents append events, Neo4jGraphMutator consumes them in batches.
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
    """
    World event buffer - CQRS write-side.

    Responsibilities:
        1. Receive events from all Agents (lock-free concurrent)
        2. Accumulate events by Tick
        3. Support batch drain for Neo4jGraphMutator consumption

    Design:
        - Uses asyncio.Queue for async lock-free append
        - Agents only append, never write to Neo4j directly
        - Single consumer pattern (Neo4jGraphMutator)
        - Async locking prevents race between append and drain

    Example:
        >>> buffer = WorldEventBuffer()
        >>> await buffer.append(event)  # From Agent
        >>> events = await buffer.drain(tick_number=1)  # From Mutator
    """

    def __init__(self, max_size: int = 10000, timeout_seconds: float = 5.0):
        """
        Initialize event buffer.

        Args:
            max_size: Maximum queue size before blocking
            timeout_seconds: Timeout for queue operations
        """
        self._queue: asyncio.Queue["WorldEvent"] = asyncio.Queue(maxsize=max_size)
        self._tick_events: dict[int, list["WorldEvent"]] = defaultdict(list)
        self._event_count = 0
        self._current_tick = 0
        # CRITICAL FIX: Add lock for tick_events access (prevents race with drain)
        self._tick_lock = asyncio.Lock()
        self._timeout_seconds = timeout_seconds

    async def append(self, event: "WorldEvent") -> None:
        """
        Agents call this to append events.

        Warning:
            Agents MUST NOT directly operate on Neo4j,
            only submit events through this method!

        Args:
            event: Domain event to buffer
        """
        # CRITICAL FIX: Queue put with timeout to prevent blocking forever
        try:
            await asyncio.wait_for(
                self._queue.put(event),
                timeout=self._timeout_seconds
            )
        except asyncio.TimeoutError:
            # Log and drop event rather than blocking indefinitely
            logger.warning(
                f"Event queue full after {self._timeout_seconds}s, dropping event {event.event_id}"
            )
            return

        # CRITICAL FIX: Atomic tick events update under lock
        async with self._tick_lock:
            self._tick_events[event.tick_number].append(event)
            self._event_count += 1

    async def append_many(self, events: list["WorldEvent"]) -> None:
        """
        Batch append events.

        Args:
            events: List of domain events to buffer
        """
        for event in events:
            await self.append(event)

    async def drain(self, tick_number: Optional[int] = None) -> list["WorldEvent"]:
        """
        Extract events for Neo4jGraphMutator consumption.

        CRITICAL FIX: Now async with proper locking to prevent race with append().

        Args:
            tick_number: Specific tick to drain, None for current tick

        Returns:
            List of all events for the specified tick
        """
        target_tick = tick_number if tick_number is not None else self._current_tick

        # CRITICAL FIX: Atomic read-and-clear under lock (prevents race with append)
        async with self._tick_lock:
            events = self._tick_events.get(target_tick, [])
            if target_tick in self._tick_events:
                del self._tick_events[target_tick]
            return events

    async def drain_all(self) -> list["WorldEvent"]:
        """
        Extract all pending events (for testing/shutdown).

        CRITICAL FIX: Now async with proper locking.

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
        """
        Get buffer statistics.

        CRITICAL FIX: Now async with proper locking for thread-safe reads.

        Returns:
            Dict with current_tick, pending_ticks, total_buffered_events, queue_size
        """
        async with self._tick_lock:
            return {
                "current_tick": self._current_tick,
                "pending_ticks": len(self._tick_events),
                "total_buffered_events": sum(
                    len(e) for e in self._tick_events.values()
                ),
                "queue_size": self._queue.qsize(),
            }

    async def get_events_by_type(self, tick_number: int) -> dict["EventType", list["WorldEvent"]]:
        """
        Group events by type.

        CRITICAL FIX: Now async with proper locking for thread-safe reads.

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
