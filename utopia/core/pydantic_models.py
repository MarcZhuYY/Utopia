"""Pydantic v2 data models for Utopia simulation engine.

This module defines type-safe, validated data models for the optimized
three-layer architecture (L3 Cognition, L4 Social, L5 Engine).

All models use strict validation to ensure data integrity across the
simulation pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# L3: Individual Cognition Models
# =============================================================================


class BigFiveTraits(BaseModel):
    """Big Five personality traits (OCEAN model).

    All traits range from 0.0 (low) to 1.0 (high).
    These traits influence agent behavior and belief updating.
    """

    model_config = ConfigDict(frozen=True)

    openness: float = Field(default=0.5, ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Conscientiousness (affects memory decay)"
    )
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraversion")
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0, description="Agreeableness")
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0, description="Neuroticism (affects confidence)")


class StanceState(BaseModel):
    """Immutable stance state for a specific topic.

    Position ranges from -1.0 (strongly against) to 1.0 (strongly support).
    Confidence ranges from 0.0 (uncertain) to 1.0 (certain).
    """

    model_config = ConfigDict(frozen=True)

    topic_id: str = Field(..., description="Unique topic identifier")
    position: float = Field(..., ge=-1.0, le=1.0, description="Stance position [-1, 1]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level [0, 1]")
    evidence_count: int = Field(default=0, ge=0, description="Number of supporting evidences")
    counter_count: int = Field(default=0, ge=0, description="Number of counter-arguments")
    last_updated_tick: int = Field(default=0, ge=0, description="Last update tick")

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: float) -> float:
        """Clamp position to valid range."""
        return float(np.clip(v, -1.0, 1.0))

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp confidence to valid range."""
        return float(np.clip(v, 0.0, 1.0))


class MemoryEntry(BaseModel):
    """A single memory entry with Ebbinghaus decay model.

    Memory strength V(t) = Importance * exp(-lambda * (t - t0))
    where lambda = 0.1 * (1 - Conscientiousness)
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(..., description="Unique memory ID")
    content: str = Field(..., min_length=1, description="Memory content")
    importance: float = Field(..., ge=0.0, le=1.0, description="Initial importance [0, 1]")
    creation_tick: int = Field(..., ge=0, description="Tick when memory was created")
    source_agent_id: Optional[str] = Field(default=None, description="Source agent ID")
    topic_id: Optional[str] = Field(default=None, description="Related topic")
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0, description="Emotional tone [-1, 1]")
    is_consolidated: bool = Field(default=False, description="Whether this is a consolidated memory")
    original_ids: list[str] = Field(default_factory=list, description="Original memory IDs if consolidated")

    def compute_strength(self, current_tick: int, conscientiousness: float) -> float:
        """Compute current memory strength using Ebbinghaus decay.

        Args:
            current_tick: Current simulation tick
            conscientiousness: Agent's conscientiousness trait (affects decay rate)

        Returns:
            Current memory strength V(t)
        """
        delta_tick = current_tick - self.creation_tick
        lambda_decay = 0.1 * (1.0 - conscientiousness)
        return float(self.importance * np.exp(-lambda_decay * delta_tick))

    def is_forgotten(self, current_tick: int, conscientiousness: float, threshold: float = 0.1) -> bool:
        """Check if memory should be forgotten.

        Args:
            current_tick: Current simulation tick
            conscientiousness: Agent's conscientiousness trait
            threshold: Strength threshold below which memory is forgotten

        Returns:
            True if memory strength is below threshold
        """
        return self.compute_strength(current_tick, conscientiousness) < threshold


class MemoryVector(BaseModel):
    """Vector representation of memory for similarity computation."""

    model_config = ConfigDict(frozen=True)

    memory_id: str = Field(..., description="Reference to memory entry")
    vector: list[float] = Field(..., description="Embedding vector")

    def cosine_similarity(self, other: MemoryVector) -> float:
        """Compute cosine similarity with another memory vector."""
        v1 = np.array(self.vector)
        v2 = np.array(other.vector)
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(dot / norm) if norm > 0 else 0.0


class ConsolidatedExperience(BaseModel):
    """A consolidated experience from multiple similar memories.

    Created when short-term buffer overflows and similar memories
    (similarity > 0.85) are clustered together.
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(..., description="Unique experience ID")
    summary: str = Field(..., min_length=1, description="LLM-generated summary")
    original_memory_ids: list[str] = Field(..., min_length=2, description="Source memory IDs")
    average_importance: float = Field(..., ge=0.0, le=1.0, description="Weighted average importance")
    creation_tick: int = Field(..., ge=0, description="Consolidation tick")
    topic_ids: list[str] = Field(default_factory=list, description="Related topics")


class BeliefUpdateInput(BaseModel):
    """Input parameters for Bayesian belief update.

    All values are strictly validated to ensure mathematical correctness.
    """

    model_config = ConfigDict(frozen=True)

    topic_id: str = Field(..., description="Topic being updated")
    current_stance: float = Field(..., ge=-1.0, le=1.0, description="Current stance S_i")
    current_confidence: float = Field(..., ge=0.0, le=1.0, description="Current confidence C_i")
    message_stance: float = Field(..., ge=-1.0, le=1.0, description="Message stance S_m")
    message_intensity: float = Field(..., ge=0.0, le=1.0, description="Message intensity I_m")
    trust_in_sender: float = Field(..., ge=0.0, le=1.0, description="Trust in sender T_ij")
    openness: float = Field(..., ge=0.0, le=1.0, description="Openness trait O")

    @field_validator("current_stance", "message_stance")
    @classmethod
    def clamp_stance(cls, v: float) -> float:
        return float(np.clip(v, -1.0, 1.0))


class BeliefUpdateResult(BaseModel):
    """Result of Bayesian belief update.

    Contains new stance and confidence values, plus metadata about the update.
    """

    model_config = ConfigDict(frozen=True)

    topic_id: str = Field(..., description="Topic identifier")
    old_stance: float = Field(..., ge=-1.0, le=1.0, description="Previous stance")
    new_stance: float = Field(..., ge=-1.0, le=1.0, description="Updated stance")
    old_confidence: float = Field(..., ge=0.0, le=1.0, description="Previous confidence")
    new_confidence: float = Field(..., ge=0.0, le=1.0, description="Updated confidence")
    delta_stance: float = Field(..., description="Change in stance")
    delta_confidence: float = Field(..., description="Change in confidence")
    reasoning: str = Field(default="", description="Update reasoning")


class PersonaAnchor(BaseModel):
    """Core persona constraints to prevent character collapse.

    Used as System Prompt置顶约束 in Agent.decide() to maintain
    consistent personality across multiple rounds of interaction.
    """

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Social role")
    core_motivations: list[str] = Field(..., min_length=1, description="Core motivation list")
    constraints: list[str] = Field(default_factory=list, description="Behavior constraints")
    traits: BigFiveTraits = Field(default_factory=BigFiveTraits, description="Personality traits")
    expertise_domains: list[str] = Field(default_factory=list, description="Expertise areas")

    def generate_system_prompt(self) -> str:
        """Generate System Prompt置顶约束 text.

        This prompt is placed at the top of every decision context
        to prevent persona collapse.

        Returns:
            System prompt text in Chinese (matching existing prompts)
        """
        lines = [
            f"【核心人设约束】",
            f"你是 {self.name}，角色：{self.role}",
            f"",
            f"【核心动机】",
        ]
        for motivation in self.core_motivations:
            lines.append(f"- {motivation}")

        lines.extend([
            f"",
            f"【人格特质】",
            f"- 开放度: {self.traits.openness:.1f}",
            f"- 尽责性: {self.traits.conscientiousness:.1f} (影响记忆衰减)",
            f"- 外向性: {self.traits.extraversion:.1f}",
            f"- 宜人性: {self.traits.agreeableness:.1f}",
            f"- 神经质: {self.traits.neuroticism:.1f} (影响置信度波动)",
        ])

        if self.constraints:
            lines.extend([f"", f"【行为约束】"])
            for constraint in self.constraints:
                lines.append(f"- {constraint}")

        if self.expertise_domains:
            lines.extend([f"", f"【专业领域】"])
            lines.append(f"{', '.join(self.expertise_domains)}")

        lines.extend([
            f"",
            f"【重要】请在所有决策中保持以上人设一致性，不要演变为AI助手。",
        ])

        return "\n".join(lines)


class ReasoningTrace(BaseModel):
    """Complete reasoning trace for an action.

    Provides full auditability of agent decision-making process
    for L6 analysis layer.
    """

    model_config = ConfigDict(frozen=False)

    trace_id: str = Field(..., description="Unique trace ID")
    agent_id: str = Field(..., description="Acting agent")
    tick: int = Field(..., ge=0, description="Decision tick")
    situation_analysis: str = Field(..., description="Situation assessment")
    attention_focus: list[str] = Field(default_factory=list, description="Focused topics")
    retrieved_memories: list[str] = Field(default_factory=list, description="Retrieved memory IDs")
    considered_stances: dict[str, float] = Field(default_factory=dict, description="Relevant stances")
    reasoning_chain: list[str] = Field(..., min_length=1, description="Step-by-step reasoning")
    chosen_action: str = Field(..., description="Selected action type")
    expected_impact: str = Field(default="", description="Expected outcome")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Decision confidence")


class ActionWithTrace(BaseModel):
    """Action with full reasoning trace attached.

    Replaces the basic Action model to provide complete traceability.
    """

    model_config = ConfigDict(frozen=False)

    action_id: str = Field(..., description="Unique action ID")
    agent_id: str = Field(..., description="Acting agent")
    action_type: Literal["speak", "private_message", "change_belief", "act", "silent"] = Field(
        ..., description="Action type"
    )
    target_agent_id: Optional[str] = Field(default=None, description="Target agent")
    content: str = Field(default="", description="Action content")
    topic_id: str = Field(default="", description="Related topic")
    tick: int = Field(..., ge=0, description="Action tick")
    trace: ReasoningTrace = Field(..., description="Complete reasoning trace")


# =============================================================================
# L4: Social Interaction Models
# =============================================================================


class HomophilyUpdateInput(BaseModel):
    """Input for homophily-based trust update.

    Implements the formula: Affinity = 1 - |S_a - S_b| / 2
    """

    model_config = ConfigDict(frozen=True)

    agent_a_id: str = Field(..., description="First agent")
    agent_b_id: str = Field(..., description="Second agent")
    current_trust: float = Field(..., ge=-1.0, le=1.0, description="Current trust level")
    stance_a: float = Field(..., ge=-1.0, le=1.0, description="Agent A's stance")
    stance_b: float = Field(..., ge=-1.0, le=1.0, description="Agent B's stance")
    interaction_quality: float = Field(..., ge=0.0, le=1.0, description="Quality of interaction")

    def compute_affinity(self) -> float:
        """Compute stance affinity between agents."""
        return 1.0 - (abs(self.stance_a - self.stance_b) / 2.0)

    def compute_trust_delta(self) -> float:
        """Compute trust change using homophily formula.

        Delta_Trust = (Affinity - 0.5) * Interaction_Quality * 0.1
        """
        affinity = self.compute_affinity()
        return (affinity - 0.5) * self.interaction_quality * 0.1


class TrustUpdateResult(BaseModel):
    """Result of trust update operation."""

    model_config = ConfigDict(frozen=True)

    agent_a_id: str = Field(..., description="First agent")
    agent_b_id: str = Field(..., description="Second agent")
    old_trust: float = Field(..., ge=-1.0, le=1.0, description="Previous trust")
    new_trust: float = Field(..., ge=-1.0, le=1.0, description="Updated trust")
    delta: float = Field(..., description="Trust change")
    affinity: float = Field(..., ge=0.0, le=1.0, description="Stance affinity")
    reason: str = Field(default="", description="Update reason")


class PropagationBatch(BaseModel):
    """Batch of messages for vectorized propagation.

    Enables efficient NumPy-based BFS propagation.
    """

    model_config = ConfigDict(frozen=True)

    sender_ids: list[str] = Field(..., description="Sender agent IDs")
    receiver_ids: list[str] = Field(..., description="Receiver agent IDs")
    message_contents: list[str] = Field(..., description="Message contents")
    topic_ids: list[str] = Field(..., description="Topic IDs")
    depths: list[int] = Field(..., description="Propagation depths")
    trust_levels: list[float] = Field(..., description="Trust levels for each edge")


# =============================================================================
# L5: Engine Models
# =============================================================================


class WorldStateSnapshot(BaseModel):
    """Immutable snapshot of world state at a specific tick.

    Used for double-buffering to prevent causal confusion.
    """

    model_config = ConfigDict(frozen=True)

    tick: int = Field(..., ge=0, description="Snapshot tick")
    agent_states: dict[str, StanceState] = Field(..., description="Agent stance states")
    relationship_matrix: dict[tuple[str, str], float] = Field(
        default_factory=dict, description="Trust matrix as dict"
    )
    event_log: list[dict[str, Any]] = Field(default_factory=list, description="Events this tick")


class ActionBufferEntry(BaseModel):
    """Buffered action waiting to be committed."""

    model_config = ConfigDict(frozen=True)

    action: ActionWithTrace = Field(..., description="The action")
    source_tick: int = Field(..., ge=0, description="Tick when action was generated")
    priority: int = Field(default=0, description="Execution priority")


class AsyncLLMCall(BaseModel):
    """Represents an async LLM API call.

    Tracks retry state and backoff timing.
    """

    model_config = ConfigDict(frozen=False)

    call_id: str = Field(..., description="Unique call ID")
    prompt: str = Field(..., description="LLM prompt")
    agent_id: str = Field(..., description="Requesting agent")
    tick: int = Field(..., ge=0, description="Request tick")
    priority: int = Field(default=0, description="Call priority")
    attempt_count: int = Field(default=0, ge=0, description="Retry attempts")
    last_attempt_time: Optional[datetime] = Field(default=None, description="Last attempt timestamp")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", description="Call status"
    )
    result: Optional[str] = Field(default=None, description="LLM response")
    error_message: Optional[str] = Field(default=None, description="Error if failed")

    def compute_backoff_delay(self, base_delay: float = 1.0) -> float:
        """Compute exponential backoff delay.

        Delay = base_delay * 2^attempt_count

        Args:
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds
        """
        return base_delay * (2 ** self.attempt_count)


class SimulationMetrics(BaseModel):
    """Performance metrics for simulation analysis."""

    model_config = ConfigDict(frozen=False)

    total_ticks: int = Field(default=0, ge=0, description="Ticks executed")
    total_agents: int = Field(default=0, ge=0, description="Agent count")
    llm_calls_made: int = Field(default=0, ge=0, description="LLM API calls")
    llm_errors: int = Field(default=0, ge=0, description="Failed LLM calls")
    average_decision_time_ms: float = Field(default=0.0, ge=0.0, description="Avg decision latency")
    memory_consolidations: int = Field(default=0, ge=0, description="Consolidation events")
    propagation_events: int = Field(default=0, ge=0, description="Message propagations")


# =============================================================================
# Utility Functions
# =============================================================================


def clip_stance(value: float) -> float:
    """Clamp stance value to [-1, 1] range."""
    return float(np.clip(value, -1.0, 1.0))


def clip_confidence(value: float) -> float:
    """Clamp confidence value to [0, 1] range."""
    return float(np.clip(value, 0.0, 1.0))


def clip_trust(value: float) -> float:
    """Clamp trust value to [-1, 1] range."""
    return float(np.clip(value, -1.0, 1.0))
