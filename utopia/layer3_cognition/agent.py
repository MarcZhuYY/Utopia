"""Agent core class - the 'brain' of each simulation agent.

Integrates persona, memory, beliefs, relationships, and state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from utopia.core.models import (
    Action,
    AgentState,
    MemoryItem,
    Persona,
    Stance,
)
from utopia.core.pydantic_models import ActivityStatus
from utopia.layer3_cognition.memory import MemorySystem
from utopia.layer3_cognition.beliefs import BeliefSystem

if TYPE_CHECKING:
    from utopia.layer4_social.relationships import RelationshipMap


@dataclass
class Agent:
    """Individual agent with cognitive capabilities.

    This is the main agent class that combines all cognitive components.

    Attributes:
        id: Unique agent identifier
        persona: Persona/human identity
        memory: Memory system
        beliefs: Belief system
        state: Current state
        relationship_map: Reference to relationship network
        activity_status: Current activity status (SLEEPING/ROUTINE/AWAKE_CRITICAL)
        silent_ticks: Number of consecutive silent ticks
    """

    id: str = field(default_factory=lambda: f"A{uuid.uuid4().hex[:8]}")
    persona: Persona = field(default_factory=Persona)
    memory: MemorySystem = field(default_factory=MemorySystem)
    beliefs: BeliefSystem = field(default_factory=BeliefSystem)
    state: AgentState = field(default_factory=AgentState)
    _relationship_map: Optional[Any] = field(default=None, repr=False)
    activity_status: ActivityStatus = field(default=ActivityStatus.ROUTINE)
    silent_ticks: int = field(default=0)

    def __post_init__(self):
        """Initialize defaults if not set."""
        if not self.persona.name:
            self.persona.name = f"Agent_{self.id}"

    @property
    def name(self) -> str:
        """Get agent name."""
        return self.persona.name

    @property
    def influence(self) -> float:
        """Get agent's base influence."""
        return self.persona.influence_base

    def set_relationship_map(self, rel_map: Any) -> None:
        """Set reference to relationship map.

        Args:
            rel_map: RelationshipMap instance
        """
        self._relationship_map = rel_map

    def get_stance(self, topic_id: str) -> Optional[Stance]:
        """Get current stance on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            Optional[Stance]: Current stance or None
        """
        return self.beliefs.get_stance(topic_id)

    def get_trust(self, other_agent_id: str) -> float:
        """Get trust level toward another agent.

        Args:
            other_agent_id: Target agent ID

        Returns:
            float: Trust level (-1 to 1)
        """
        if self._relationship_map:
            edge = self._relationship_map.get(self.id, other_agent_id)
            return edge.trust
        return 0.0

    def add_memory(
        self,
        content: str,
        topic_id: Optional[str] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        source_agent: Optional[str] = None,
    ) -> None:
        """Add a memory item.

        Args:
            content: Memory content
            topic_id: Related topic
            importance: Importance score (0-1)
            emotional_valence: Emotional tone (-1 to 1)
            source_agent: Source agent ID
        """
        item = MemoryItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            emotional_valence=emotional_valence,
            source_agent=source_agent or "",
            verified=False,
        )
        self.memory.add(item, self.id, topic_id)

    def retrieve_memories(
        self,
        query: str,
        topic_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve relevant memories.

        Args:
            query: Query string
            topic_id: Filter by topic
            limit: Maximum results

        Returns:
            list[MemoryItem]: Retrieved memories
        """
        return self.memory.retrieve(query, topic_id, limit)

    def update_belief(
        self,
        topic_id: str,
        new_info: str,
        direction: str = "neutral",
        strength: float = 0.5,
    ) -> None:
        """Update belief on a topic.

        Args:
            topic_id: Topic identifier
            new_info: New information
            direction: pro/con/neutral
            strength: Evidence strength (0-1)
        """
        self.beliefs.update(topic_id, new_info, self, direction, strength)

    def decay_energy(self, amount: float = 0.05) -> None:
        """Decay agent energy.

        Args:
            amount: Amount to decay
        """
        self.state.energy = max(0.0, self.state.energy - amount)

    def recover_energy(self, amount: float = 0.1) -> None:
        """Recover agent energy.

        Args:
            amount: Amount to recover
        """
        self.state.energy = min(1.0, self.state.energy + amount)

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent to dictionary.

        Returns:
            dict: Agent data
        """
        return {
            "id": self.id,
            "persona": self.persona.to_dict(),
            "state": self.state.to_dict(),
            "influence": self.influence,
        }

    @classmethod
    def from_entity(
        cls,
        entity_id: str,
        name: str,
        role: str,
        expertise: list[str],
        base_stances: dict[str, float],
        influence: float = 0.5,
    ) -> Agent:
        """Create agent from entity data.

        Args:
            entity_id: Entity ID
            name: Agent name
            role: Agent role
            expertise: Expertise areas
            base_stances: Initial stances
            influence: Base influence

        Returns:
            Agent: Created agent
        """
        persona = Persona(
            name=name,
            role=role,
            expertise=expertise,
            influence_base=influence,
        )

        agent = cls(id=entity_id, persona=persona)
        agent.beliefs.initialize_stances(base_stances)

        return agent

    def set_activity_status(self, status: ActivityStatus, current_tick: int = 0) -> None:
        """Set agent activity status.

        Args:
            status: New activity status
            current_tick: Current simulation tick
        """
        old_status = self.activity_status
        self.activity_status = status

        if status == ActivityStatus.SLEEPING:
            self.silent_ticks += 1
        else:
            if old_status == ActivityStatus.SLEEPING:
                # Waking up from sleep
                self.silent_ticks = 0

    def should_force_wake(self, max_silent_ticks: int = 10) -> bool:
        """Check if agent should be force-woken.

        Args:
            max_silent_ticks: Maximum silent ticks before force wake

        Returns:
            bool: True if should force wake
        """
        return self.silent_ticks >= max_silent_ticks

    def silent_update_beliefs(self, topic_id: str, delta: float, sender_trust: float) -> None:
        """Silently update beliefs (for SLEEPING state).

        Performs Bayesian update without LLM call.

        Args:
            topic_id: Topic to update
            delta: Stance delta
            sender_trust: Trust in message sender
        """
        from utopia.core.pydantic_models import BigFiveTraits

        # Get current stance
        stance = self.get_stance(topic_id)
        current_pos = stance.position if stance else 0.0
        current_conf = stance.confidence if stance else 0.5

        # Get openness from BigFive traits (default 0.5)
        if hasattr(self.persona, 'big_five') and self.persona.big_five:
            openness = self.persona.big_five.openness
        else:
            openness = 0.5

        # Apply delta with openness and confidence modulation
        new_pos = current_pos + delta * (1 - current_conf) * openness
        new_pos = max(-1.0, min(1.0, new_pos))

        # Update confidence
        if delta * current_pos > 0:  # Same direction
            new_conf = current_conf + (1 - current_conf) * sender_trust * 0.1
        else:  # Opposite direction
            new_conf = current_conf - (1 - current_conf) * (1 - openness) * 0.05
        new_conf = max(0.0, min(1.0, new_conf))

        # Update via belief system if method exists
        if hasattr(self.beliefs, 'update_stance'):
            self.beliefs.update_stance(topic_id, new_pos, new_conf)
        elif hasattr(self.beliefs, 'set_stance'):
            self.beliefs.set_stance(topic_id, new_pos, new_conf)
