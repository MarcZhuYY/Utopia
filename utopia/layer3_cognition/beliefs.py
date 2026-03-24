"""Bayesian belief system implementation.

This module implements the mathematically rigorous Bayesian stance update
formula as specified in the optimization requirements:

Delta = (S_m - S_i) * T_ij * I_m
S_new = S_i + Delta * (1 - C_i) * O

Confidence evolution:
- If sign(Delta) == sign(S_i): C_new = C_i + (1 - C_i) * T_ij * 0.1
- Else: C_new = C_i - (1 - C_i) * (1 - O) * 0.05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from utopia.core.pydantic_models import (
    BeliefUpdateInput,
    BeliefUpdateResult,
    BigFiveTraits,
    StanceState,
)

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


def _sanitize_float(
    value: float,
    default: float = 0.0,
    bounds: tuple[float, float] = (-float("inf"), float("inf")),
    name: str = "value",
) -> float:
    """Sanitize float value: replace NaN/Inf with default, then clip.

    SAFETY FIX: Prevents NaN/Inf from contaminating belief system.

    Args:
        value: Input value to sanitize
        default: Default value if NaN or Inf (when bounds are finite)
        bounds: Valid range (min, max)
        name: Parameter name for logging

    Returns:
        Sanitized float value
    """
    # Check for NaN
    if np.isnan(value):
        return float(np.clip(default, bounds[0], bounds[1]))

    # Check for Inf (if bounds are finite)
    if np.isinf(value):
        if bounds[0] > -float("inf") and bounds[1] < float("inf"):
            return float(np.clip(default, bounds[0], bounds[1]))
        # If bounds are infinite, clip to something reasonable
        return float(np.clip(0.0 if np.isnan(default) else default, -1e6, 1e6))

    # Clip to bounds
    return float(np.clip(value, bounds[0], bounds[1]))


@dataclass
class BayesianBeliefDelta:
    """Change in belief state from Bayesian update.

    Attributes:
        topic_id: Topic that was updated
        old_position: Previous stance position
        new_position: Updated stance position
        old_confidence: Previous confidence
        new_confidence: Updated confidence
        delta_stance: Computed delta before openness/clamping
        trust_factor: Trust in information source
        intensity: Message intensity
        openness: Openness trait that moderated the update
    """

    topic_id: str
    old_position: float
    new_position: float
    old_confidence: float
    new_confidence: float
    delta_stance: float
    trust_factor: float
    intensity: float
    openness: float

    def to_result(self, reasoning: str = "") -> BeliefUpdateResult:
        """Convert to Pydantic result model."""
        return BeliefUpdateResult(
            topic_id=self.topic_id,
            old_stance=self.old_position,
            new_stance=self.new_position,
            old_confidence=self.old_confidence,
            new_confidence=self.new_confidence,
            delta_stance=self.new_position - self.old_position,
            delta_confidence=self.new_confidence - self.old_confidence,
            reasoning=reasoning,
        )


class BayesianBeliefSystem:
    """Belief system with mathematically rigorous Bayesian updates.

    This implementation strictly follows the mathematical formulas:
    - Position update with openness moderation
    - Confidence evolution based on consistency
    - Personality trait integration (Openness, Neuroticism)
    """

    # Maximum change per tick to prevent sudden shifts
    MAX_CHANGE_PER_TICK: float = 0.15

    # Confidence adjustment rates
    CONFIDENCE_BOOST_RATE: float = 0.1
    CONFIDENCE_PENALTY_RATE: float = 0.05

    def __init__(
        self,
        traits: Optional[BigFiveTraits] = None,
        max_change_per_tick: float = MAX_CHANGE_PER_TICK,
    ):
        """Initialize Bayesian belief system.

        Args:
            traits: Big Five personality traits (uses defaults if None)
            max_change_per_tick: Maximum stance change per update
        """
        self.stances: dict[str, StanceState] = {}
        self.traits = traits or BigFiveTraits()
        self._max_change = max_change_per_tick

    def get_stance(self, topic_id: str) -> Optional[StanceState]:
        """Get current stance on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            Current stance state or None if no stance exists
        """
        return self.stances.get(topic_id)

    def get_position(self, topic_id: str) -> float:
        """Get stance position on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            Position in [-1, 1], 0.0 if no stance exists
        """
        stance = self.stances.get(topic_id)
        return stance.position if stance else 0.0

    def get_confidence(self, topic_id: str) -> float:
        """Get confidence on a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            Confidence in [0, 1], 0.0 if no stance exists
        """
        stance = self.stances.get(topic_id)
        return stance.confidence if stance else 0.0

    def initialize_stance(
        self,
        topic_id: str,
        position: float,
        confidence: float = 0.5,
    ) -> StanceState:
        """Initialize a new stance.

        SAFETY FIX: Sanitizes inputs to prevent NaN/Inf initialization.

        Args:
            topic_id: Topic identifier
            position: Initial position in [-1, 1]
            confidence: Initial confidence in [0, 1]

        Returns:
            Created stance state
        """
        # SAFETY FIX: Sanitize initialization parameters
        position = _sanitize_float(
            position, default=0.0, bounds=(-1.0, 1.0), name="position"
        )
        confidence = _sanitize_float(
            confidence, default=0.5, bounds=(0.0, 1.0), name="confidence"
        )

        stance = StanceState(
            topic_id=topic_id,
            position=position,
            confidence=confidence,
            evidence_count=0,
            counter_count=0,
            last_updated_tick=0,
        )
        self.stances[topic_id] = stance
        return stance

    def bayesian_update(
        self,
        topic_id: str,
        message_stance: float,
        intensity: float,
        trust_in_sender: float,
        current_tick: int,
    ) -> BayesianBeliefDelta:
        """Perform Bayesian stance update.

        This is the core algorithm implementing:
        Delta = (S_m - S_i) * T_ij * I_m
        S_new = S_i + Delta * (1 - C_i) * O

        Args:
            topic_id: Topic being updated
            message_stance: Message stance S_m in [-1, 1]
            intensity: Message intensity I_m in [0, 1]
            trust_in_sender: Trust in sender T_ij in [0, 1]
            current_tick: Current simulation tick

        Returns:
            BeliefDelta with old and new values
        """
        # SAFETY FIX: Sanitize all inputs to prevent NaN/Inf contamination
        message_stance = _sanitize_float(
            message_stance, default=0.0, bounds=(-1.0, 1.0), name="message_stance"
        )
        intensity = _sanitize_float(
            intensity, default=0.5, bounds=(0.0, 1.0), name="intensity"
        )
        trust_in_sender = _sanitize_float(
            trust_in_sender, default=0.5, bounds=(0.0, 1.0), name="trust_in_sender"
        )
        current_tick = max(0, int(_sanitize_float(current_tick, default=0, name="current_tick")))

        # Get or create current stance
        current = self.stances.get(topic_id)
        if current is None:
            current = self.initialize_stance(topic_id, 0.0, 0.5)

        # Store old values
        old_position = _sanitize_float(
            current.position, default=0.0, bounds=(-1.0, 1.0), name="old_position"
        )
        old_confidence = _sanitize_float(
            current.confidence, default=0.5, bounds=(0.0, 1.0), name="old_confidence"
        )

        # Get openness from personality traits (sanitize trait value)
        openness = _sanitize_float(
            self.traits.openness, default=0.5, bounds=(0.0, 1.0), name="openness"
        )

        # Compute Delta = (S_m - S_i) * T_ij * I_m
        delta = (message_stance - old_position) * trust_in_sender * intensity

        # Compute new position: S_new = S_i + Delta * (1 - C_i) * O
        # Higher confidence = less open to change
        # Higher openness = more receptive to new info
        openness_factor = openness * (1.0 - old_confidence)
        raw_new_position = old_position + delta * openness_factor

        # Clamp change to max per tick
        new_position = self._clamp_change(old_position, raw_new_position)

        # Confidence evolution
        # If sign(Delta) == sign(S_i): consistent with current stance
        # Else: contradictory information
        if abs(delta) < 1e-10:
            # No meaningful update — keep confidence unchanged
            new_confidence = old_confidence
        elif np.sign(delta) == np.sign(old_position):
            # Consistent: boost confidence
            # C_new = C_i + (1 - C_i) * T_ij * 0.1
            confidence_boost = (
                old_confidence + (1.0 - old_confidence) * trust_in_sender * self.CONFIDENCE_BOOST_RATE
            )
            new_confidence = min(1.0, confidence_boost)
        else:
            # Contradictory: slight decrease based on neuroticism
            # Higher neuroticism = more shaken by contradiction
            # C_new = C_i - (1 - C_i) * (1 - O) * 0.05
            neuroticism_factor = 1.0 - openness
            confidence_penalty = (
                old_confidence - (1.0 - old_confidence) * neuroticism_factor * self.CONFIDENCE_PENALTY_RATE
            )
            new_confidence = max(0.0, confidence_penalty)

        # Update stance state
        self.stances[topic_id] = StanceState(
            topic_id=topic_id,
            position=new_position,
            confidence=new_confidence,
            evidence_count=current.evidence_count + (1 if delta > 0 else 0),
            counter_count=current.counter_count + (1 if delta < 0 else 0),
            last_updated_tick=current_tick,
        )

        return BayesianBeliefDelta(
            topic_id=topic_id,
            old_position=old_position,
            new_position=new_position,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            delta_stance=delta,
            trust_factor=trust_in_sender,
            intensity=intensity,
            openness=openness,
        )

    def update_from_input(
        self,
        input_data: BeliefUpdateInput,
        current_tick: int,
    ) -> BeliefUpdateResult:
        """Update from validated Pydantic input.

        Args:
            input_data: Validated update parameters
            current_tick: Current simulation tick

        Returns:
            Update result with reasoning
        """
        delta = self.bayesian_update(
            topic_id=input_data.topic_id,
            message_stance=input_data.message_stance,
            intensity=input_data.message_intensity,
            trust_in_sender=input_data.trust_in_sender,
            current_tick=current_tick,
        )

        # Generate reasoning string
        consistency = "一致" if np.sign(delta.delta_stance) == np.sign(delta.old_position) else "矛盾"
        reasoning = (
            f"接收到立场={input_data.message_stance:.2f}的信息，"
            f"强度={input_data.message_intensity:.2f}，"
            f"来源信任度={input_data.trust_in_sender:.2f}。"
            f"Delta={delta.delta_stance:.3f}，"
            f"与当前立场{consistency}。"
            f"开放度={delta.openness:.2f}调节后，"
            f"立场从{delta.old_position:.2f}变为{delta.new_position:.2f}，"
            f"置信度从{delta.old_confidence:.2f}变为{delta.new_confidence:.2f}。"
        )

        return delta.to_result(reasoning)

    def _clamp_change(self, old: float, new: float) -> float:
        """Clamp new position to max change per tick.

        Args:
            old: Previous position
            new: Computed new position

        Returns:
            Clamped position
        """
        change = new - old
        if abs(change) > self._max_change:
            return old + (self._max_change if change > 0 else -self._max_change)
        return float(np.clip(new, -1.0, 1.0))

    def get_all_stances(self) -> dict[str, float]:
        """Get all stance positions.

        Returns:
            Dict mapping topic_id to position
        """
        return {tid: s.position for tid, s in self.stances.items()}

    def compute_stance_variance(self) -> float:
        """Compute variance of all stance positions.

        Returns:
            Variance of positions (0.0 if no stances)
        """
        if not self.stances:
            return 0.0
        positions = [s.position for s in self.stances.values()]
        return float(np.var(positions))

    def to_dict(self) -> dict:
        """Serialize to dictionary.

        Returns:
            Dictionary with all stance data
        """
        return {
            "stances": {
                tid: {
                    "position": s.position,
                    "confidence": s.confidence,
                    "evidence_count": s.evidence_count,
                    "counter_count": s.counter_count,
                    "last_updated_tick": s.last_updated_tick,
                }
                for tid, s in self.stances.items()
            },
            "traits": self.traits.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianBeliefSystem":
        """Load from dictionary.

        Args:
            data: Serialized data

        Returns:
            Restored belief system
        """
        traits_data = data.get("traits", {})
        traits = BigFiveTraits(**traits_data)
        system = cls(traits=traits)

        for tid, sdata in data.get("stances", {}).items():
            system.stances[tid] = StanceState(
                topic_id=tid,
                position=sdata["position"],
                confidence=sdata["confidence"],
                evidence_count=sdata.get("evidence_count", 0),
                counter_count=sdata.get("counter_count", 0),
                last_updated_tick=sdata.get("last_updated_tick", 0),
            )

        return system


def compute_belief_distance(
    stance_a: float,
    stance_b: float,
    confidence_a: float = 0.5,
    confidence_b: float = 0.5,
) -> float:
    """Compute weighted belief distance between two agents.

    Distance considers both stance difference and confidence.
    Higher confidence = less willing to change (higher effective distance).

    Args:
        stance_a: Agent A's stance
        stance_b: Agent B's stance
        confidence_a: Agent A's confidence
        confidence_b: Agent B's confidence

    Returns:
        Weighted distance in [0, 2]
    """
    stance_diff = abs(stance_a - stance_b)
    avg_confidence = (confidence_a + confidence_b) / 2.0
    # Higher confidence amplifies perceived difference
    return stance_diff * (1.0 + avg_confidence)
