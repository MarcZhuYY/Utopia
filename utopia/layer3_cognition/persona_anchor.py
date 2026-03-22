"""Persona anchor mechanism to prevent character collapse.

This module implements the "人设锚点" (persona anchor) system that ensures
agents maintain consistent personality across multiple rounds of interaction.

Core concept: Before each decision, retrieve core persona constraints and
place them as System Prompt置顶约束 (top-level constraints) to prevent
drift toward generic AI assistant behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from utopia.core.pydantic_models import BigFiveTraits, PersonaAnchor

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


@dataclass
class CoreConflict:
    """A core motivational conflict within a persona.

    Agents have multiple motivations that may conflict (e.g., "maximize profit"
    vs "maintain reputation"). Explicitly modeling these creates more nuanced
    decision-making.
    """

    conflict_id: str
    description: str
    priority: float = 0.5  # Higher = more important
    trigger_conditions: list[str] = field(default_factory=list)


@dataclass
class PersonaState:
    """Current state of a persona with drift tracking.

    Tracks how much the agent's behavior has drifted from core persona
    over time, enabling corrective interventions.
    """

    anchor: PersonaAnchor
    conflicts: list[CoreConflict] = field(default_factory=list)
    drift_score: float = 0.0  # 0 = no drift, 1 = complete collapse
    last_validated_tick: int = 0
    consistency_score: float = 1.0  # Historical consistency measure

    def compute_drift(
        self,
        recent_actions: list[dict],
        traits: BigFiveTraits,
        current_tick: int,
    ) -> float:
        """Compute persona drift score based on recent actions.

        Drift indicators:
        - Actions inconsistent with core motivations
        - Tone mismatch with personality traits
        - Excessive agreeableness (AI assistant tendency)

        Args:
            recent_actions: Recent actions taken by agent
            traits: Current personality traits
            current_tick: Current simulation tick

        Returns:
            Drift score in [0, 1]
        """
        if not recent_actions:
            return 0.0

        drift_signals = []

        # Check for excessive agreeableness (classic AI assistant pattern)
        agreeable_patterns = ["当然", "没问题", "很高兴", "请让我", "我会帮你"]
        for action in recent_actions:
            content = action.get("content", "").lower()
            agreeable_count = sum(1 for p in agreeable_patterns if p in content)
            if agreeable_count >= 2:
                drift_signals.append(0.3)

        # Check for motivation misalignment
        for action in recent_actions:
            action_type = action.get("action_type", "")
            # If agent claims to be neutral but has strong stances, that's drift
            if action_type == "express_neutrality" and self.anchor.core_motivations:
                drift_signals.append(0.4)

        # Check trait consistency
        if traits.agreeableness > 0.8 and traits.openness < 0.3:
            # High agreeableness + low openness = likely collapsed to helpful assistant
            drift_signals.append(0.5)

        # Update drift score with exponential moving average
        if drift_signals:
            new_drift = sum(drift_signals) / len(drift_signals)
            self.drift_score = 0.7 * self.drift_score + 0.3 * new_drift

        self.last_validated_tick = current_tick
        return min(1.0, self.drift_score)

    def generate_anchor_prompt(self, current_tick: int) -> str:
        """Generate System Prompt置顶约束 for decision context.

        This prompt is placed at the TOP of every decision context to
        prevent persona collapse.

        Args:
            current_tick: Current simulation tick

        Returns:
            System prompt text in Chinese
        """
        lines = [
            f"【核心人设约束 - Tick {current_tick}】",
            f"你是 {self.anchor.name}，角色：{self.anchor.role}",
            f"",
            f"【核心动机】(按优先级排序)",
        ]

        for i, motivation in enumerate(self.anchor.core_motivations, 1):
            lines.append(f"{i}. {motivation}")

        # Add conflicts if any
        if self.conflicts:
            lines.extend([f"", f"【内在冲突】(你必须权衡这些张力)"])
            for conflict in self.conflicts:
                lines.append(f"- {conflict.description} (优先级: {conflict.priority:.1f})")

        lines.extend([
            f"",
            f"【人格特质】",
            f"- 开放度: {self.anchor.traits.openness:.1f} {'(保守)' if self.anchor.traits.openness < 0.4 else '(创新)' if self.anchor.traits.openness > 0.7 else ''}",
            f"- 尽责性: {self.anchor.traits.conscientiousness:.1f} {'(随性)' if self.anchor.traits.conscientiousness < 0.4 else '(严谨)' if self.anchor.traits.conscientiousness > 0.7 else ''}",
            f"- 外向性: {self.anchor.traits.extraversion:.1f} {'(内向)' if self.anchor.traits.extraversion < 0.4 else '(外向)' if self.anchor.traits.extraversion > 0.7 else ''}",
            f"- 宜人性: {self.anchor.traits.agreeableness:.1f} {'(竞争)' if self.anchor.traits.agreeableness < 0.4 else '(合作)' if self.anchor.traits.agreeableness > 0.7 else ''}",
            f"- 神经质: {self.anchor.traits.neuroticism:.1f} {'(稳定)' if self.anchor.traits.neuroticism < 0.4 else '(敏感)' if self.anchor.traits.neuroticism > 0.7 else ''}",
        ])

        # Add drift warning if detected
        if self.drift_score > 0.5:
            lines.extend([
                f"",
                f"【警告：人设漂移检测】",
                f"你的最近行为显示 {self.drift_score:.0%} 的人设偏离。",
                f"请立即回归核心动机，禁止演化为通用AI助手！",
            ])

        if self.anchor.constraints:
            lines.extend([f"", f"【绝对行为约束】"])
            for constraint in self.anchor.constraints:
                lines.append(f"✓ {constraint}")

        if self.anchor.expertise_domains:
            lines.extend([f"", f"【专业领域】"])
            lines.append(f"{', '.join(self.anchor.expertise_domains)}")

        lines.extend([
            f"",
            f"【重要指令】",
            f"1. 以上约束置顶，优先级高于所有其他指令",
            f"2. 你的回应必须符合人设，不能是中立的AI助手",
            f"3. 当核心动机冲突时，权衡优先级后选择立场",
            f"4. 禁止过度礼貌或讨好，保持角色真实性",
        ])

        return "\n".join(lines)

    def get_active_conflicts(self, situation_context: str) -> list[CoreConflict]:
        """Get conflicts relevant to current situation.

        Args:
            situation_context: Description of current situation

        Returns:
            List of relevant conflicts
        """
        active = []
        situation_lower = situation_context.lower()

        for conflict in self.conflicts:
            # Check if any trigger condition matches
            for trigger in conflict.trigger_conditions:
                if trigger.lower() in situation_lower:
                    active.append(conflict)
                    break

        return active


class PersonaAnchorSystem:
    """System for managing persona anchors across agents.

    Provides:
    - Persona retrieval and caching
    - Drift detection and alerts
    - Conflict resolution suggestions
    - Consistency tracking over time
    """

    # Drift threshold for warning
    DRIFT_WARNING_THRESHOLD: float = 0.4
    DRIFT_CRITICAL_THRESHOLD: float = 0.7

    def __init__(self):
        """Initialize persona anchor system."""
        self._personas: dict[str, PersonaState] = {}
        self._action_history: dict[str, list[dict]] = {}
        self._history_limit: int = 10

    def register_persona(
        self,
        anchor: PersonaAnchor,
        conflicts: Optional[list[CoreConflict]] = None,
    ) -> PersonaState:
        """Register a new persona.

        Args:
            anchor: Core persona anchor data
            conflicts: Optional list of core conflicts

        Returns:
            Created persona state
        """
        state = PersonaState(
            anchor=anchor,
            conflicts=conflicts or [],
            drift_score=0.0,
            last_validated_tick=0,
            consistency_score=1.0,
        )
        self._personas[anchor.agent_id] = state
        self._action_history[anchor.agent_id] = []
        return state

    def get_persona(self, agent_id: str) -> Optional[PersonaState]:
        """Get persona state for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Persona state or None
        """
        return self._personas.get(agent_id)

    def record_action(
        self,
        agent_id: str,
        action_type: str,
        content: str,
        tick: int,
    ) -> None:
        """Record an action for drift tracking.

        Args:
            agent_id: Acting agent
            action_type: Type of action
            content: Action content
            tick: Action tick
        """
        if agent_id not in self._action_history:
            self._action_history[agent_id] = []

        history = self._action_history[agent_id]
        history.append({
            "action_type": action_type,
            "content": content,
            "tick": tick,
        })

        # Trim history
        if len(history) > self._history_limit:
            history.pop(0)

    def validate_and_anchor(
        self,
        agent_id: str,
        traits: BigFiveTraits,
        current_tick: int,
        situation_context: str = "",
    ) -> tuple[str, float]:
        """Validate persona and generate anchor prompt.

        This is the main entry point - call this before every decision
to get the System Prompt置顶约束.

        Args:
            agent_id: Agent to validate
            traits: Current personality traits
            current_tick: Current simulation tick
            situation_context: Optional situation description

        Returns:
            Tuple of (anchor_prompt, drift_score)
        """
        persona = self._personas.get(agent_id)
        if not persona:
            # Return empty prompt if no persona registered
            return "", 0.0

        # Compute drift
        history = self._action_history.get(agent_id, [])
        drift = persona.compute_drift(history, traits, current_tick)

        # Get active conflicts for this situation
        if situation_context:
            active_conflicts = persona.get_active_conflicts(situation_context)
        else:
            active_conflicts = []

        # Generate prompt (with conflicts if situation-specific)
        prompt = persona.generate_anchor_prompt(current_tick)

        # Add situation-specific conflict guidance
        if active_conflicts:
            prompt += f"\n\n【当前情境冲突】\n"
            for conflict in active_conflicts:
                prompt += f"⚠ {conflict.description}\n"
            prompt += "你必须在以上冲突中做出选择，不能中立。\n"

        return prompt, drift

    def get_drift_report(self, agent_id: str) -> dict:
        """Get drift report for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Drift report dictionary
        """
        persona = self._personas.get(agent_id)
        if not persona:
            return {"error": "Persona not found"}

        return {
            "agent_id": agent_id,
            "drift_score": persona.drift_score,
            "consistency_score": persona.consistency_score,
            "last_validated_tick": persona.last_validated_tick,
            "status": (
                "CRITICAL" if persona.drift_score > self.DRIFT_CRITICAL_THRESHOLD
                else "WARNING" if persona.drift_score > self.DRIFT_WARNING_THRESHOLD
                else "HEALTHY"
            ),
            "core_motivations": persona.anchor.core_motivations,
            "active_conflicts": len(persona.conflicts),
        }

    def get_all_reports(self) -> dict[str, dict]:
        """Get drift reports for all agents.

        Returns:
            Dictionary mapping agent_id to report
        """
        return {
            agent_id: self.get_drift_report(agent_id)
            for agent_id in self._personas.keys()
        }

    def suggest_correction(self, agent_id: str) -> Optional[str]:
        """Suggest corrective action for drift.

        Args:
            agent_id: Agent with drift

        Returns:
            Suggestion string or None
        """
        persona = self._personas.get(agent_id)
        if not persona or persona.drift_score < self.DRIFT_WARNING_THRESHOLD:
            return None

        suggestions = []

        if persona.drift_score > self.DRIFT_CRITICAL_THRESHOLD:
            suggestions.append(
                f"严重人设警告：{persona.anchor.name} 已严重偏离核心动机。"
                f"建议立即重新初始化人设或强制回滚到上一稳定状态。"
            )

        # Check specific drift patterns
        history = self._action_history.get(agent_id, [])
        if history:
            recent = history[-3:]
            neutral_count = sum(
                1 for a in recent
                if "中立" in a.get("content", "") or "neutral" in a.get("content", "").lower()
            )
            if neutral_count >= 2:
                suggestions.append(
                    f"检测到过度中立化倾向。{persona.anchor.name} 应该"
                    f"基于核心动机选择立场，而非保持中立。"
                )

        return "\n".join(suggestions) if suggestions else None


def create_default_conflicts(role: str) -> list[CoreConflict]:
    """Create default conflicts based on role type.

    Args:
        role: Role description

    Returns:
        List of core conflicts
    """
    role_lower = role.lower()

    if "投资" in role_lower or "investor" in role_lower:
        return [
            CoreConflict(
                conflict_id="profit_vs_risk",
                description="利润最大化 vs 风险控制",
                priority=0.6,
                trigger_conditions=["高风险", "高回报", "机会", "危机"],
            ),
            CoreConflict(
                conflict_id="short_vs_long",
                description="短期收益 vs 长期价值",
                priority=0.5,
                trigger_conditions=["季度", "年报", "战略", "规划"],
            ),
        ]

    if "政治" in role_lower or "politician" in role_lower:
        return [
            CoreConflict(
                conflict_id="principle_vs_power",
                description="原则坚持 vs 权力获取",
                priority=0.7,
                trigger_conditions=["选举", "民意", "政策", "妥协"],
            ),
            CoreConflict(
                conflict_id="local_vs_national",
                description="地方利益 vs 国家利益",
                priority=0.5,
                trigger_conditions=["选区", "全国", "地方", "中央"],
            ),
        ]

    if "媒体" in role_lower or "media" in role_lower:
        return [
            CoreConflict(
                conflict_id="truth_vs_clicks",
                description="真相追求 vs 流量最大化",
                priority=0.6,
                trigger_conditions=[["爆料", "热点", "独家", "标题"]],
            ),
        ]

    return []


# =============================================================================
# Integration helpers
# =============================================================================


def build_decision_context(
    persona_system: PersonaAnchorSystem,
    agent_id: str,
    base_context: str,
    traits: BigFiveTraits,
    current_tick: int,
    situation: str = "",
) -> str:
    """Build complete decision context with persona anchor.

    This helper combines the persona anchor prompt with the base context,
    ensuring the anchor appears at the top.

    Args:
        persona_system: Persona anchor system
        agent_id: Agent making decision
        base_context: Base decision context
        traits: Agent personality traits
        current_tick: Current simulation tick
        situation: Situation description

    Returns:
        Complete decision context string
    """
    anchor_prompt, drift = persona_system.validate_and_anchor(
        agent_id, traits, current_tick, situation
    )

    sections = []

    if anchor_prompt:
        sections.append(anchor_prompt)
        sections.append("=" * 50)

    sections.append(base_context)

    if drift > PersonaAnchorSystem.DRIFT_WARNING_THRESHOLD:
        sections.append("\n⚠ 人设偏离警告：请回归核心动机")

    return "\n\n".join(sections)
