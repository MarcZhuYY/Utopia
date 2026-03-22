"""Belief system for agents.

Implements stance management with Bayesian-style updates and LLM reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from utopia.core.models import Stance
from utopia.core.config import WorldRules

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


@dataclass
class BeliefDelta:
    """Change in belief state.

    Attributes:
        topic_id: Topic that changed
        old_position: Previous position
        new_position: New position
        old_confidence: Previous confidence
        new_confidence: New confidence
        reason: Reason for change
    """

    topic_id: str
    old_position: float
    new_position: float
    old_confidence: float
    new_confidence: float
    reason: str = ""


class BeliefSystem:
    """Manages agent beliefs and stances on topics.

    Implements:
    - Stance storage and retrieval
    - Bayesian-style belief updating
    - LLM-driven impact analysis
    - Stance change momentum and limits
    """

    def __init__(
        self,
        max_change_per_tick: float = 0.15,
        momentum: float = 0.3,
        confidence_boost: float = 0.1,
    ):
        """Initialize belief system.

        Args:
            max_change_per_tick: Max stance change per tick
            momentum: Stance change momentum coefficient
            confidence_boost: Confidence boost rate
        """
        self.stances: dict[str, Stance] = {}
        self._max_change = max_change_per_tick
        self._momentum = momentum
        self._confidence_boost = confidence_boost

    def get_stance(self, topic_id: str) -> Optional[Stance]:
        """Get stance on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            Optional[Stance]: Current stance or None
        """
        return self.stances.get(topic_id)

    def get_position(self, topic_id: str) -> float:
        """Get stance position on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            float: Position (-1 to 1), 0 if no stance
        """
        stance = self.stances.get(topic_id)
        return stance.position if stance else 0.0

    def initialize_stances(self, stances: dict[str, float]) -> None:
        """Initialize stances from dictionary.

        Args:
            stances: Dict of topic_id -> position
        """
        for topic_id, position in stances.items():
            self.stances[topic_id] = Stance(
                topic_id=topic_id,
                position=max(-1.0, min(1.0, position)),
                confidence=0.5,
                evidence=[],
                counter_arguments=[],
                last_updated=datetime.now(),
            )

    def update(
        self,
        topic_id: str,
        new_info: str,
        agent: Agent,
        direction: str = "neutral",
        strength: float = 0.5,
    ) -> BeliefDelta:
        """Update belief on a topic.

        For MVP, uses direct position update with LLM analysis.
        Phase 2: Full Bayesian update with LLM evidence analysis.

        Args:
            topic_id: Topic identifier
            new_info: New information received
            agent: Agent context
            direction: pro/con/neutral (from LLM analysis)
            strength: Evidence strength (0-1)

        Returns:
            BeliefDelta: Change that occurred
        """
        # Get or create current stance
        current = self.stances.get(topic_id)
        if current is None:
            current = Stance(
                topic_id=topic_id,
                position=0.0,
                confidence=0.5,
                evidence=[],
                counter_arguments=[],
                last_updated=datetime.now(),
            )
            self.stances[topic_id] = current

        old_position = current.position
        old_confidence = current.confidence

        # Calculate new position based on direction
        new_position = self._calculate_new_position(
            current.position, direction, strength, current.confidence
        )

        # Clamp to max change per tick
        clamped_position = self._clamp_change(old_position, new_position)

        # Update confidence
        new_confidence = min(1.0, current.confidence + self._confidence_boost * strength)

        # Update stance
        current.position = clamped_position
        current.confidence = new_confidence
        current.evidence.append(new_info)
        current.last_updated = datetime.now()

        return BeliefDelta(
            topic_id=topic_id,
            old_position=old_position,
            new_position=clamped_position,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            reason=new_info,
        )

    def _calculate_new_position(
        self,
        current: float,
        direction: str,
        strength: float,
        confidence: float,
    ) -> float:
        """Calculate new position based on direction and evidence.

        Args:
            current: Current position
            direction: pro/con/neutral
            strength: Evidence strength
            confidence: Current confidence

        Returns:
            float: Calculated new position
        """
        if direction == "neutral":
            return current

        # Direction multiplier: pro = +1, con = -1
        multiplier = 1.0 if direction == "pro" else -1.0

        # Base change scaled by evidence strength and inverse of confidence
        # (less confident agents update more)
        base_change = multiplier * strength * (1.0 - confidence * 0.5)

        # Apply momentum from previous changes
        momentum_effect = self._momentum * base_change

        return current + base_change + momentum_effect

    def _clamp_change(self, old: float, new: float) -> float:
        """Clamp new position to max change per tick.

        Args:
            old: Old position
            new: Calculated new position

        Returns:
            float: Clamped new position
        """
        change = new - old
        if abs(change) > self._max_change:
            return old + (self._max_change if change > 0 else -self._max_change)
        return new

    def analyze_impact(
        self,
        new_info: str,
        topic_id: str,
        agent: Agent,
    ) -> dict[str, Any]:
        """Analyze impact of new information using LLM.

        This is a placeholder for LLM-based impact analysis.
        When LLM is integrated, this will call the LLM to determine
        direction and strength of impact.

        Args:
            new_info: New information
            topic_id: Topic being updated
            agent: Agent context

        Returns:
            dict: Analysis result with direction and strength
        """
        # TODO: Replace with actual LLM call
        # For MVP, return neutral
        return {
            "claimed_direction": "neutral",
            "claimed_strength": 0.0,
            "reasoning": "LLM not integrated",
        }

    def get_all_stances(self) -> dict[str, float]:
        """Get all stance positions.

        Returns:
            dict: topic_id -> position
        """
        return {tid: s.position for tid, s in self.stances.items()}

    def get_confidence(self, topic_id: str) -> float:
        """Get confidence on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            float: Confidence (0-1)
        """
        stance = self.stances.get(topic_id)
        return stance.confidence if stance else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize beliefs.

        Returns:
            dict: Belief data
        """
        return {
            "stances": {tid: s.to_dict() for tid, s in self.stances.items()},
        }
