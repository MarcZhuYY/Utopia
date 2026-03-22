"""Mailbox module implementing Pub/Sub message pattern for agent wake/sleep mechanism.

This module provides:
- Mailbox: Per-agent message queue with priority support
- MessageBroker: Central message routing with pub/sub semantics
- ActiveTaskPool: Batch processing pool for tick execution
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

from utopia.core.models import Message, ReceivedMessage
from utopia.core.pydantic_models import ActivityStatus

logger = structlog.get_logger()


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(order=True)
class PrioritizedMessage:
    """Message with priority for queue ordering."""

    priority: int = field(compare=True)
    timestamp: datetime = field(compare=True)
    message: ReceivedMessage = field(compare=False)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8], compare=False)


class Mailbox:
    """Per-agent message queue.

    Implements priority-based message storage with async support.
    Each agent has one mailbox for incoming messages.

    Attributes:
        agent_id: Owner agent ID
        max_size: Maximum queue size (drops lowest priority when full)
        _queue: Internal priority queue
    """

    def __init__(self, agent_id: str, max_size: int = 100):
        """Initialize mailbox.

        Args:
            agent_id: Owner agent ID
            max_size: Maximum queue size
        """
        self.agent_id = agent_id
        self.max_size = max_size
        self._queue: asyncio.PriorityQueue[PrioritizedMessage] = asyncio.PriorityQueue()
        self._message_count = 0
        self._lock = asyncio.Lock()

    async def put(
        self,
        message: ReceivedMessage,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """Add message to mailbox.

        Args:
            message: Message to add
            priority: Message priority

        Returns:
            str: Message ID
        """
        async with self._lock:
            # Check if we need to drop messages (mailbox full)
            if self._queue.qsize() >= self.max_size:
                # Remove lowest priority message
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            prioritized = PrioritizedMessage(
                priority=priority.value,
                timestamp=datetime.now(),
                message=message,
            )
            await self._queue.put(prioritized)
            self._message_count += 1

            return prioritized.message_id

    async def get(self) -> Optional[ReceivedMessage]:
        """Get next message from mailbox.

        Returns:
            Optional[ReceivedMessage]: Message or None if empty
        """
        try:
            prioritized = self._queue.get_nowait()
            return prioritized.message
        except asyncio.QueueEmpty:
            return None

    async def get_all(self) -> list[ReceivedMessage]:
        """Get all messages from mailbox.

        Returns:
            list[ReceivedMessage]: All messages in priority order
        """
        messages = []
        while not self._queue.empty():
            msg = await self.get()
            if msg:
                messages.append(msg)
        return messages

    def empty(self) -> bool:
        """Check if mailbox is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def clear(self) -> None:
        """Clear all messages from mailbox."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class MessageBroker:
    """Central message broker for Pub/Sub pattern.

    Routes messages from publishers to subscriber mailboxes.
    Supports broadcast and targeted messaging.

    Attributes:
        mailboxes: Dict of agent_id -> Mailbox
        _lock: Async lock for thread safety
    """

    def __init__(self):
        """Initialize message broker."""
        self.mailboxes: dict[str, Mailbox] = {}
        self._lock = asyncio.Lock()
        self.logger = logger

    async def register_agent(self, agent_id: str, max_mailbox_size: int = 100) -> Mailbox:
        """Register an agent and create its mailbox.

        Args:
            agent_id: Agent ID
            max_mailbox_size: Maximum mailbox size

        Returns:
            Mailbox: Created mailbox
        """
        async with self._lock:
            if agent_id not in self.mailboxes:
                self.mailboxes[agent_id] = Mailbox(agent_id, max_mailbox_size)
                self.logger.debug("Registered agent mailbox", agent_id=agent_id)
            return self.mailboxes[agent_id]

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and remove its mailbox.

        Args:
            agent_id: Agent ID
        """
        async with self._lock:
            if agent_id in self.mailboxes:
                self.mailboxes[agent_id].clear()
                del self.mailboxes[agent_id]
                self.logger.debug("Unregistered agent mailbox", agent_id=agent_id)

    async def publish(
        self,
        message: ReceivedMessage,
        target_agent_ids: Optional[list[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> list[str]:
        """Publish message to target agents.

        Args:
            message: Message to publish
            target_agent_ids: Target agent IDs (None = broadcast)
            priority: Message priority

        Returns:
            list[str]: List of message IDs
        """
        message_ids = []

        async with self._lock:
            targets = target_agent_ids or list(self.mailboxes.keys())

            for agent_id in targets:
                if agent_id in self.mailboxes:
                    msg_id = await self.mailboxes[agent_id].put(message, priority)
                    message_ids.append(msg_id)

        return message_ids

    async def broadcast(
        self,
        message: ReceivedMessage,
        exclude_sender: bool = True,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> list[str]:
        """Broadcast message to all agents.

        Args:
            message: Message to broadcast
            exclude_sender: Whether to exclude sender from broadcast
            priority: Message priority

        Returns:
            list[str]: List of message IDs
        """
        targets = list(self.mailboxes.keys())

        if exclude_sender and message.from_agent in targets:
            targets.remove(message.from_agent)

        return await self.publish(message, targets, priority)

    def get_mailbox(self, agent_id: str) -> Optional[Mailbox]:
        """Get mailbox for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Optional[Mailbox]: Mailbox or None if not registered
        """
        return self.mailboxes.get(agent_id)

    def get_all_mailboxes(self) -> dict[str, Mailbox]:
        """Get all mailboxes.

        Returns:
            dict[str, Mailbox]: All mailboxes
        """
        return self.mailboxes.copy()


@dataclass
class ActiveTaskPool:
    """Active task pool for batch processing during ticks.

    Tracks which agents need to be processed in the current tick
    based on their mailbox state and activity status.

    Attributes:
        active_agent_ids: Set of agent IDs to process this tick
        force_wake_ids: Set of agent IDs forced to wake up
    """

    active_agent_ids: set[str] = field(default_factory=set)
    force_wake_ids: set[str] = field(default_factory=set)

    def add(self, agent_id: str, force_wake: bool = False) -> None:
        """Add agent to active pool.

        Args:
            agent_id: Agent ID
            force_wake: Whether this is a forced wake
        """
        self.active_agent_ids.add(agent_id)
        if force_wake:
            self.force_wake_ids.add(agent_id)

    def remove(self, agent_id: str) -> None:
        """Remove agent from active pool.

        Args:
            agent_id: Agent ID
        """
        self.active_agent_ids.discard(agent_id)
        self.force_wake_ids.discard(agent_id)

    def is_active(self, agent_id: str) -> bool:
        """Check if agent is in active pool.

        Args:
            agent_id: Agent ID

        Returns:
            bool: True if active
        """
        return agent_id in self.active_agent_ids

    def is_force_wake(self, agent_id: str) -> bool:
        """Check if agent was forced to wake.

        Args:
            agent_id: Agent ID

        Returns:
            bool: True if force wake
        """
        return agent_id in self.force_wake_ids

    def clear(self) -> None:
        """Clear active pool."""
        self.active_agent_ids.clear()
        self.force_wake_ids.clear()

    def get_batch(self, batch_size: Optional[int] = None) -> list[str]:
        """Get batch of agents to process.

        Args:
            batch_size: Maximum batch size (None = all)

        Returns:
            list[str]: Agent IDs
        """
        agents = list(self.active_agent_ids)
        if batch_size:
            return agents[:batch_size]
        return agents

    def __len__(self) -> int:
        """Get number of active agents."""
        return len(self.active_agent_ids)


class TickProcessor:
    """Tick processor with mailbox pattern and active task pool.

    Orchestrates per-tick processing:
    1. Collect messages from all mailboxes
    2. Determine agent activity status
    3. Batch process active agents
    4. Handle wake/sleep transitions

    Attributes:
        broker: MessageBroker instance
        active_pool: ActiveTaskPool instance
        max_silent_ticks: Maximum ticks before force wake
    """

    def __init__(self, broker: MessageBroker, max_silent_ticks: int = 10):
        """Initialize tick processor.

        Args:
            broker: MessageBroker instance
            max_silent_ticks: Maximum silent ticks before force wake
        """
        self.broker = broker
        self.active_pool = ActiveTaskPool()
        self.max_silent_ticks = max_silent_ticks
        self.logger = logger

    async def prepare_tick(self, current_tick: int) -> set[str]:
        """Prepare for tick processing.

        Collects all agents with non-empty mailboxes and adds them
        to the active pool. Also checks for force wake conditions.

        Args:
            current_tick: Current simulation tick

        Returns:
            set[str]: Set of active agent IDs
        """
        self.active_pool.clear()

        for agent_id, mailbox in self.broker.get_all_mailboxes().items():
            # Check if agent has messages
            if not mailbox.empty():
                self.active_pool.add(agent_id, force_wake=False)

        self.logger.debug(
            "Tick prepared",
            tick=current_tick,
            active_agents=len(self.active_pool),
        )

        return self.active_pool.active_agent_ids.copy()

    async def process_agent_mailbox(
        self,
        agent_id: str,
        activity_status: ActivityStatus,
    ) -> list[ReceivedMessage]:
        """Process an agent's mailbox based on activity status.

        Args:
            agent_id: Agent ID
            activity_status: Current activity status

        Returns:
            list[ReceivedMessage]: Messages to process
        """
        mailbox = self.broker.get_mailbox(agent_id)
        if not mailbox:
            return []

        if activity_status == ActivityStatus.SLEEPING:
            # In SLEEPING state, return messages for silent processing
            # but don't consume them (they stay for when agent wakes)
            return await mailbox.get_all()

        # In active states, consume messages
        return await mailbox.get_all()

    def force_wake_check(self, agent_id: str, silent_ticks: int) -> bool:
        """Check if agent should be force-woken.

        Args:
            agent_id: Agent ID
            silent_ticks: Number of consecutive silent ticks

        Returns:
            bool: True if should force wake
        """
        return silent_ticks >= self.max_silent_ticks

    def get_active_pool(self) -> ActiveTaskPool:
        """Get current active pool."""
        return self.active_pool
