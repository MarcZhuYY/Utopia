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


class ActivityStatus(str, Enum):
    """Agent活动状态 - 用于唤醒/睡眠机制

    状态转换规则:
    - SLEEPING: Delta < 0.15, 仅执行静默贝叶斯更新
    - ROUTINE: 0.15 <= Delta < 0.4, 标准LLM调用
    - AWAKE_CRITICAL: Delta >= 0.4, 深度推理 (ToT/CoT)
    """

    SLEEPING = "sleeping"
    ROUTINE = "routine"
    AWAKE_CRITICAL = "awake_critical"


class AgentRuntimeState(BaseModel):
    """Agent运行时状态 - 扩展AgentState用于唤醒机制

    Attributes:
        activity_status: 当前活动状态
        silent_ticks: 连续睡眠tick数
        last_active_tick: 上次活跃tick
        mailbox_message_count: 当前邮箱消息数
    """

    model_config = ConfigDict(frozen=False)

    activity_status: ActivityStatus = Field(
        default=ActivityStatus.ROUTINE, description="当前活动状态"
    )
    silent_ticks: int = Field(default=0, ge=0, description="连续睡眠tick数")
    last_active_tick: int = Field(default=0, ge=0, description="上次活跃tick")
    mailbox_message_count: int = Field(default=0, ge=0, description="邮箱消息数")


class CognitiveDissonanceInput(BaseModel):
    """认知失调度计算输入

    公式: Delta = |Message_Stance - Agent_Stance| * Sender_Trust * Message_Importance
    """

    model_config = ConfigDict(frozen=True)

    message_stance: float = Field(..., ge=-1.0, le=1.0, description="消息立场 S_m")
    agent_stance: float = Field(..., ge=-1.0, le=1.0, description="Agent立场 S_i")
    sender_trust: float = Field(..., ge=0.0, le=1.0, description="发送者信任度 T_ij")
    message_importance: float = Field(..., ge=0.0, le=1.0, description="消息重要性 I_m")

    def compute_delta(self) -> float:
        """计算认知失调度 Delta"""
        stance_diff = abs(self.message_stance - self.agent_stance)
        return stance_diff * self.sender_trust * self.message_importance

    def determine_activity_status(self, threshold_low: float = 0.15, threshold_high: float = 0.4) -> ActivityStatus:
        """根据Delta确定活动状态"""
        delta = self.compute_delta()
        if delta < threshold_low:
            return ActivityStatus.SLEEPING
        elif delta < threshold_high:
            return ActivityStatus.ROUTINE
        else:
            return ActivityStatus.AWAKE_CRITICAL


class WakeUpDecision(BaseModel):
    """唤醒决策结果"""

    model_config = ConfigDict(frozen=True)

    target_status: ActivityStatus = Field(..., description="目标活动状态")
    max_delta: float = Field(..., ge=0.0, description="最大认知失调度")
    critical_messages: list[str] = Field(default_factory=list, description="需深度处理的消息ID")
    force_wake: bool = Field(default=False, description="是否强制唤醒")
    reason: str = Field(default="", description="决策原因")


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
# L5: LLM Router Models (Capability-Based Routing)
# =============================================================================


class LLMModel(str, Enum):
    """全旗舰模型池 - 各模型均为顶级配置，按专长分配

    Model Pool:
    - MINIMAX_M27: 主力底座，默认路由，处理复杂Agent协作
    - DEEPSEEK_R1: 数学与逻辑审核，专攻纯逻辑推理
    - KIMI_K25: 记忆档案馆，专攻超长文本无损压缩
    - GLM_5: 世界法则裁判，担任RuleValidator
    - QWEN_35_PLUS: 神经末梢，处理高频浅层交互
    """

    MINIMAX_M27 = "minimax-m2.7"
    DEEPSEEK_R1 = "deepseek-r1"
    KIMI_K25 = "kimi-k2.5"
    GLM_5 = "glm-5"
    QWEN_35_PLUS = "qwen-3.5-plus"


class TaskType(str, Enum):
    """任务类型决定模型路由

    Routing Strategy:
    - DEFAULT: 默认路由至 MINIMAX_M27
    - BAYESIAN_STANCE_UPDATE: 贝叶斯立场更新 → DEEPSEEK_R1
    - COGNITIVE_DISSONANCE: 认知失调处理 → DEEPSEEK_R1
    - MEMORY_CONSOLIDATION: 记忆归并 → KIMI_K25
    - LONG_CONTEXT_PARSE: 超长文本解析 → KIMI_K25 (context > 30K)
    - RULE_VALIDATION: 规则校验 → GLM_5
    - HIGH_FREQ_INTERACT: 高频浅层交互 → QWEN_35_PLUS
    - INFO_PROPAGATION: 信息传播 → QWEN_35_PLUS
    """

    DEFAULT = "default"
    BAYESIAN_STANCE_UPDATE = "bayesian_stance_update"
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    LONG_CONTEXT_PARSE = "long_context_parse"
    RULE_VALIDATION = "rule_validation"
    HIGH_FREQ_INTERACT = "high_freq_interact"
    INFO_PROPAGATION = "info_propagation"


class TaskRequest(BaseModel):
    """LLM任务请求 (Capability-Based Routing)

    Attributes:
        task_id: 唯一任务ID
        task_type: 任务类型决定路由
        agent_id: 请求Agent ID
        agent_importance: Agent重要性 [0, 1]
        prompt: 提示词内容
        context_length: 上下文长度(字节)
        priority: 优先级(数值越大越优先)
        timeout: 超时秒数
        require_validation: 是否需要规则校验
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="唯一任务ID")
    task_type: TaskType = Field(default=TaskType.DEFAULT, description="任务类型决定路由")
    agent_id: str = Field(..., description="请求Agent ID")
    agent_importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Agent重要性")
    prompt: str = Field(..., min_length=1, description="提示词")
    context_length: int = Field(default=0, ge=0, description="上下文长度")
    priority: int = Field(default=0, description="优先级(高优先先执行)")
    timeout: float = Field(default=30.0, gt=0.0, description="超时秒数")
    require_validation: bool = Field(default=False, description="是否需要规则校验")

    @field_validator("context_length")
    @classmethod
    def auto_detect_context_length(cls, v: int, info) -> int:
        """如果未提供context_length，自动从prompt计算"""
        if v == 0 and "prompt" in info.data:
            return len(info.data["prompt"].encode("utf-8"))
        return v


class LLMResponse(BaseModel):
    """LLM响应结果

    Attributes:
        task_id: 对应任务ID
        success: 是否成功
        model_used: 实际使用的模型
        content: 响应内容
        latency_ms: 延迟毫秒
        tokens_used: Token使用量
        retries: 重试次数
        fallback_used: 是否使用了降级
        error_message: 错误信息(失败时)
    """

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(..., description="对应任务ID")
    success: bool = Field(..., description="是否成功")
    model_used: LLMModel = Field(..., description="实际使用的模型")
    content: str = Field(default="", description="响应内容")
    latency_ms: float = Field(..., ge=0.0, description="延迟毫秒")
    tokens_used: int = Field(default=0, ge=0, description="Token使用量")
    retries: int = Field(default=0, ge=0, description="重试次数")
    fallback_used: bool = Field(default=False, description="是否使用了降级")
    error_message: str = Field(default="", description="错误信息")


class ModelCapability(BaseModel):
    """模型能力配置

    Attributes:
        model: 模型枚举
        specialties: 专长任务类型列表
        context_window: 上下文窗口大小
        max_concurrent: 最大并发数
        provider: API提供商名称
        api_key_env: API Key环境变量名
    """

    model_config = ConfigDict(frozen=True)

    model: LLMModel = Field(..., description="模型")
    specialties: list[TaskType] = Field(default_factory=list, description="专长任务类型")
    context_window: int = Field(..., ge=1000, description="上下文窗口大小")
    max_concurrent: int = Field(..., ge=1, description="最大并发数")
    provider: str = Field(..., description="API提供商")
    api_key_env: str = Field(..., description="API Key环境变量名")


class RouterStats(BaseModel):
    """LLM Router统计信息"""

    model_config = ConfigDict(frozen=False)

    total_requests: int = Field(default=0, ge=0, description="总请求数")
    successful_requests: int = Field(default=0, ge=0, description="成功请求数")
    failed_requests: int = Field(default=0, ge=0, description="失败请求数")
    fallback_requests: int = Field(default=0, ge=0, description="降级请求数")
    total_latency_ms: float = Field(default=0.0, ge=0.0, description="总延迟")
    model_distribution: dict[LLMModel, int] = Field(
        default_factory=dict, description="模型使用分布"
    )
    task_type_distribution: dict[TaskType, int] = Field(
        default_factory=dict, description="任务类型分布"
    )


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
