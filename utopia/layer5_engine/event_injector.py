"""External event injector for simulation.

Allows dynamic injection of events during simulation.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from utopia.core.models import ExternalEvent as ExternalEventModel

if TYPE_CHECKING:
    from utopia.layer5_engine.engine import SimulationEngine


@dataclass
class ExternalEvent:
    """External event to inject into simulation.

    Attributes:
        id: Unique event ID
        description: Event description
        topic_id: Related topic ID
        importance: Event importance (0-1)
        timestamp: When event occurs
        source: Event source
    """

    id: str = field(default_factory=lambda: f"EXT{uuid.uuid4().hex[:8]}")
    description: str = ""
    topic_id: str = ""
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "external"


class ExternalEventInjector:
    """Injects external events into simulation.

    External events can be:
    - Scheduled in advance
    - Injected dynamically during simulation
    - Triggered by real-world data feeds
    """

    def __init__(self, engine: Optional["SimulationEngine"] = None):
        """Initialize event injector.

        Args:
            engine: Simulation engine reference
        """
        self.engine = engine
        self.pending_events: deque[ExternalEvent] = deque()
        self.injected_events: list[ExternalEvent] = []

    def schedule(self, event: ExternalEvent) -> None:
        """Schedule an event for future injection.

        Args:
            event: Event to schedule
        """
        self.pending_events.append(event)

    def schedule_at_tick(
        self,
        tick: int,
        description: str,
        topic_id: str,
        importance: float = 0.5,
        source: str = "scheduled",
    ) -> None:
        """Schedule event to inject at specific tick.

        Args:
            tick: Target tick
            description: Event description
            topic_id: Related topic
            importance: Event importance
            source: Event source
        """
        event = ExternalEvent(
            description=description,
            topic_id=topic_id,
            importance=importance,
            source=source,
        )
        self.pending_events.append(event)

    def inject(self, event: ExternalEvent) -> None:
        """Inject event into simulation immediately.

        Args:
            event: Event to inject
        """
        if not self.engine or not self.engine.world_state:
            return

        # Add to world state
        self.engine.world_state.external_events.append(event)
        self.injected_events.append(event)

        # Update knowledge graph
        self.engine.world_state.knowledge_graph.add_event(
            event, source_agent="external"
        )

        # Find relevant agents and inject perceptions
        relevant_agents = self.engine.world_state.knowledge_graph.get_topic_agents(
            event.topic_id
        )

        for agent_id in relevant_agents:
            agent = self.engine.world_state.agents.get(agent_id)
            if agent:
                # Add perception directly to agent
                agent.add_memory(
                    content=f"[External Event] {event.description}",
                    topic_id=event.topic_id,
                    importance=event.importance,
                    emotional_valence=0.0,
                    source_agent="external_event",
                )

        # Log injection
        if self.engine.logger:
            self.engine.logger.info(
                "External event injected",
                event_id=event.id,
                topic=event.topic_id,
                importance=event.importance,
                affected_agents=len(relevant_agents),
            )

    def inject_pending(self) -> None:
        """Inject all pending events.

        Called at the start of each tick.
        """
        while self.pending_events:
            event = self.pending_events.popleft()
            self.inject(event)

    def clear_pending(self) -> None:
        """Clear all pending events."""
        self.pending_events.clear()

    def get_injected_count(self) -> int:
        """Get count of injected events.

        Returns:
            int: Number of injected events
        """
        return len(self.injected_events)

    def create_news_event(
        self,
        headline: str,
        topic_id: str,
        importance: float = 0.5,
    ) -> ExternalEvent:
        """Create event from news headline.

        Args:
            headline: News headline
            topic_id: Topic ID
            importance: Event importance

        Returns:
            ExternalEvent: Created event
        """
        return ExternalEvent(
            description=headline,
            topic_id=topic_id,
            importance=importance,
            source="news",
        )

    def create_market_event(
        self,
        description: str,
        topic_id: str,
        importance: float = 0.7,
    ) -> ExternalEvent:
        """Create market-related event.

        Args:
            description: Event description
            topic_id: Topic ID
            importance: Event importance

        Returns:
            ExternalEvent: Created event
        """
        return ExternalEvent(
            description=description,
            topic_id=topic_id,
            importance=importance,
            source="market",
        )
