"""Agent decision engine using CoT/ToT reasoning.

Implements the perceive -> attend -> reason -> action cycle.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from utopia.core.config import SimulationConfig
from utopia.core.models import Action, Message, ReceivedMessage
from utopia.core.pydantic_models import (
    ActivityStatus,
    CognitiveDissonanceInput,
    WakeUpDecision,
)
from utopia.layer3_cognition.memory import MemorySystem

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


@dataclass
class AttentionItem:
    """Item representing a focus of attention.

    Attributes:
        topic_id: Topic being focused on
        description: Description of the situation
        urgency: Urgency score (0-1)
        source: Source of this attention (perception/event/memory)
    """

    topic_id: str
    description: str
    urgency: float = 0.5
    source: str = ""


@dataclass
class Decision:
    """Represents a decision made by an agent.

    Attributes:
        situation_analysis: Analysis of current situation
        chosen_action: Type of action chosen
        action_target: Target agent ID (if applicable)
        action_content: Content of the action
        reasoning: Reasoning process
        expected_impact: Expected impact
    """

    situation_analysis: str = ""
    chosen_action: str = "silent"
    action_target: str = ""
    action_content: str = ""
    reasoning: str = ""
    expected_impact: str = ""


@dataclass
class SimulationContext:
    """Context for agent decision-making.

    Passed to decision engine to provide world state.
    """

    current_tick: int
    world_state: Any  # WorldState - set via TYPE_CHECKING in real usage
    config: SimulationConfig
    current_situation_summary: str = ""


class AgentDecisionEngine:
    """Decision engine for agents.

    Implements the perceive -> attend -> reason -> action cycle:
    1. Perceive: Collect relevant information
    2. Attend: Focus on most important items
    3. Reason: Use CoT/ToT to make decisions
    4. Act: Generate validated actions
    """

    MAX_ATTENTION_ITEMS = 3
    MAX_ACTIONS_PER_TICK = 3

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize decision engine.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()

        # Wake/Sleep mechanism thresholds
        self.WAKE_THRESHOLD_LOW = 0.15  # SLEEPING -> ROUTINE
        self.WAKE_THRESHOLD_HIGH = 0.4  # ROUTINE -> AWAKE_CRITICAL

    def perceive_and_filter(
        self,
        agent: Agent,
        inbox_messages: list[ReceivedMessage],
        force_wake: bool = False,
    ) -> WakeUpDecision:
        """Perceive messages and determine activity status before LLM call.

        This is the wake/sleep interception point. It calculates cognitive
        dissonance for each message and determines if agent should:
        - SLEEPING: Delta < 0.15, silent Bayesian update only
        - ROUTINE: 0.15 <= Delta < 0.4, standard LLM call
        - AWAKE_CRITICAL: Delta >= 0.4, deep reasoning

        Formula: Delta = |Message_Stance - Agent_Stance| * Sender_Trust * Message_Importance

        Args:
            agent: Agent receiving messages
            inbox_messages: Messages in agent's mailbox
            force_wake: Whether to force wake (external event or max silent ticks)

        Returns:
            WakeUpDecision: Target activity status and critical messages
        """
        if not inbox_messages:
            return WakeUpDecision(
                target_status=ActivityStatus.SLEEPING,
                max_delta=0.0,
                critical_messages=[],
                force_wake=force_wake,
                reason="No messages",
            )

        max_delta = 0.0
        critical_message_ids = []
        message_deltas = []

        for msg in inbox_messages:
            # Get agent's current stance on the topic
            agent_stance = agent.get_stance(msg.message.topic_id)
            agent_stance_pos = agent_stance.position if agent_stance else 0.0

            # Get sender trust
            sender_trust = agent.get_trust(msg.from_agent)

            # Message importance (default to 0.5 if not specified)
            msg_importance = getattr(msg, "importance", 0.5)

            # Calculate cognitive dissonance
            dissonance_input = CognitiveDissonanceInput(
                message_stance=msg.message.original_stance,
                agent_stance=agent_stance_pos,
                sender_trust=sender_trust,
                message_importance=msg_importance,
            )

            delta = dissonance_input.compute_delta()
            max_delta = max(max_delta, delta)
            message_deltas.append((msg, delta))

            # Track critical messages (Delta >= 0.4)
            if delta >= self.WAKE_THRESHOLD_HIGH:
                # Use message_id if available, otherwise use hash of content
                msg_id = getattr(msg.message, 'message_id', None) or f"msg_{hash(msg.message.content)}"
                critical_message_ids.append(msg_id)

        # Determine target status
        if force_wake:
            target_status = ActivityStatus.AWAKE_CRITICAL
            reason = f"Force wake (max_delta={max_delta:.3f})"
        elif max_delta < self.WAKE_THRESHOLD_LOW:
            target_status = ActivityStatus.SLEEPING
            reason = f"Low delta ({max_delta:.3f} < {self.WAKE_THRESHOLD_LOW})"
        elif max_delta < self.WAKE_THRESHOLD_HIGH:
            target_status = ActivityStatus.ROUTINE
            reason = f"Medium delta ({max_delta:.3f})"
        else:
            target_status = ActivityStatus.AWAKE_CRITICAL
            reason = f"High delta ({max_delta:.3f} >= {self.WAKE_THRESHOLD_HIGH})"

        return WakeUpDecision(
            target_status=target_status,
            max_delta=max_delta,
            critical_messages=critical_message_ids,
            force_wake=force_wake,
            reason=reason,
        )

    def silent_update(
        self,
        agent: Agent,
        messages: list[ReceivedMessage],
    ) -> None:
        """Perform silent Bayesian updates without LLM calls.

        Used when agent is in SLEEPING state. Updates beliefs but
        doesn't generate responses.

        Args:
            agent: Agent to update
            messages: Messages to process
        """
        from utopia.layer3_cognition.beliefs_v2 import BayesianBeliefSystem

        for msg in messages:
            # Get current stance
            agent_stance = agent.get_stance(msg.message.topic_id)
            agent_stance_pos = agent_stance.position if agent_stance else 0.0
            agent_confidence = agent_stance.confidence if agent_stance else 0.5

            # Get sender trust
            sender_trust = agent.get_trust(msg.from_agent)

            # Get openness from BigFive traits (handle both traits dict and big_five object)
            if hasattr(agent.persona, 'big_five') and agent.persona.big_five:
                openness = agent.persona.big_five.openness
            elif hasattr(agent.persona, 'traits') and agent.persona.traits:
                openness = agent.persona.traits.get('openness', 0.5)
            else:
                openness = 0.5

            # Calculate delta for Bayesian update
            delta = (msg.message.original_stance - agent_stance_pos) * sender_trust * 0.3

            # Silent update with reduced intensity (0.3 factor)
            new_stance = agent_stance_pos + delta * (1 - agent_confidence) * openness
            new_stance = max(-1.0, min(1.0, new_stance))

            # Update confidence
            if delta * agent_stance_pos > 0:  # Same direction
                new_confidence = agent_confidence + (1 - agent_confidence) * sender_trust * 0.1
            else:  # Opposite direction
                new_confidence = agent_confidence - (1 - agent_confidence) * (1 - openness) * 0.05
            new_confidence = max(0.0, min(1.0, new_confidence))

            # Apply update via belief system
            if hasattr(agent.beliefs, 'update_stance'):
                agent.beliefs.update_stance(
                    topic_id=msg.message.topic_id,
                    new_position=new_stance,
                    new_confidence=new_confidence,
                )

            # Add to memory with reduced importance
            agent.add_memory(
                content=msg.message.content,
                topic_id=msg.message.topic_id,
                importance=0.3 * getattr(msg, "importance", 0.5),
                source_agent=msg.from_agent,
            )

    def decide(
        self,
        agent: Agent,
        context: SimulationContext,
        perceptions: list[ReceivedMessage],
    ) -> list[Action]:
        """Make decisions based on perceptions.

        Args:
            agent: Agent making decisions
            context: Simulation context
            perceptions: List of perceived messages

        Returns:
            list[Action]: List of actions to execute
        """
        # Step 1: Perceive - update memory with new information
        self._perceive(agent, perceptions)

        # Step 2: Attend - determine focus areas
        attention = self._allocate_attention(agent, context)

        # Step 3: Reason - make decisions on each focus area
        decisions = []
        for focus in attention:
            decision = self._reason(agent, focus, context)
            if decision:
                decisions.append(decision)

        # Step 4: Generate and validate actions
        actions = self._generate_actions(agent, decisions, context)

        return actions[: self.MAX_ACTIONS_PER_TICK]

    def _perceive(
        self,
        agent: Agent,
        perceptions: list[ReceivedMessage],
    ) -> None:
        """Process perceptions and add to memory.

        Args:
            agent: Agent perceiving
            perceptions: Perceived messages
        """
        for perception in perceptions:
            # Add to memory
            agent.add_memory(
                content=perception.message.content,
                topic_id=perception.message.topic_id,
                importance=perception.trust_at_reception,
                emotional_valence=0.0,
                source_agent=perception.from_agent,
            )

            # Update beliefs if significant
            if perception.depth == 0 and perception.trust_at_reception > 0.5:
                direction = "pro" if perception.message.original_stance > 0 else "con"
                agent.update_belief(
                    topic_id=perception.message.topic_id,
                    new_info=perception.message.content,
                    direction=direction,
                    strength=perception.trust_at_reception * 0.1,
                )

    def _allocate_attention(
        self,
        agent: Agent,
        context: SimulationContext,
    ) -> list[AttentionItem]:
        """Allocate attention across perception items.

        Args:
            agent: Agent allocating attention
            context: Simulation context

        Returns:
            list[AttentionItem]: Items to focus on
        """
        attention_items = []

        # Focus on current goals and high-importance perceptions
        if agent.state.current_goal:
            attention_items.append(
                AttentionItem(
                    topic_id="current_goal",
                    description=agent.state.current_goal,
                    urgency=0.8,
                    source="goal",
                )
            )

        # Add attention focus from state
        for topic_id in agent.state.attention_focus[: self.MAX_ATTENTION_ITEMS]:
            stance = agent.get_stance(topic_id)
            if stance:
                attention_items.append(
                    AttentionItem(
                        topic_id=topic_id,
                        description=f"Ongoing matter: {topic_id}",
                        urgency=0.6,
                        source="state",
                    )
                )

        # Limit to max items
        return attention_items[: self.MAX_ATTENTION_ITEMS]

    def _reason(
        self,
        agent: Agent,
        focus: AttentionItem,
        context: SimulationContext,
    ) -> Optional[Decision]:
        """Reason about a focus item.

        Uses CoT (Chain of Thought) reasoning.

        Args:
            agent: Agent reasoning
            focus: Focus item
            context: Simulation context

        Returns:
            Optional[Decision]: Decision made
        """
        # Retrieve relevant memories
        memory_context = agent.retrieve_memories(
            query=focus.description,
            topic_id=focus.topic_id,
            limit=5,
        )

        memory_text = "\n".join([f"- {m.content}" for m in memory_context])

        stance = agent.get_stance(focus.topic_id)
        stance_text = (
            f"Position: {stance.position} (-1反对, 1支持), Confidence: {stance.confidence}"
            if stance
            else "No current stance"
        )

        # Build reasoning prompt
        prompt = f"""你是{focus.agent_id if hasattr(focus, 'agent_id') else agent.id}，角色：{agent.persona.role}，专业：{agent.persona.expertise}。

## 当前情况
{context.current_situation_summary}

## 你关注的话题：{focus.topic_id}
{focus.description}

## 相关记忆
{memory_text or "无相关记忆"}

## 你的当前立场
{stance_text}

## 你的目标
{', '.join(agent.persona.goals) or "无明确目标"}

## 可用行动
- speak (公开发言)
- private_message (私信)
- change_belief (改变立场)
- act (采取行动)
- silent (沉默)

请推理：
1. 当前形势如何？
2. 你的最优行动是什么？为什么？
3. 这个行动可能带来什么后果？

输出JSON：
{{
  "situation_analysis": "形势分析",
  "chosen_action": "行动类型",
  "action_target": "目标agent ID（如果适用）",
  "action_content": "具体内容",
  "reasoning": "推理过程",
  "expected_impact": "预期影响"
}}
"""

        # TODO: Replace with actual LLM call
        # For MVP, return a simple silent decision
        return Decision(
            situation_analysis="Reasoning placeholder - LLM not integrated",
            chosen_action="silent",
            action_target="",
            action_content="",
            reasoning="No LLM integration yet",
            expected_impact="None",
        )

    def _generate_actions(
        self,
        agent: Agent,
        decisions: list[Decision],
        context: SimulationContext,
    ) -> list[Action]:
        """Generate validated actions from decisions.

        Args:
            agent: Agent generating actions
            decisions: Decisions made
            context: Simulation context

        Returns:
            list[Action]: Validated actions
        """
        from utopia.layer2_world.rule_engine import RuleValidator

        actions = []
        for decision in decisions:
            if decision.chosen_action == "silent":
                continue

            action = Action(
                action_type=decision.chosen_action,
                target_agent_id=decision.action_target,
                content=decision.action_content,
                topic_id=decision.chosen_action,
                timestamp=datetime.now(),
            )

            # Validate action
            result = RuleValidator.validate_action(
                agent=agent,
                action=action,
                world_rules=context.config.world_rules,
                domain=context.config.domain,
            )

            if result.allowed:
                actions.append(action)
            elif result.warning:
                # Soft constraint violation - add warning and proceed
                action.content = f"{action.content} [Warning: {result.warning}]"
                actions.append(action)

        return actions


class SimpleDecisionEngine:
    """Simplified decision engine for testing without LLM.

    Makes deterministic decisions based on simple rules.
    """

    def decide(
        self,
        agent: Agent,
        context: SimulationContext,
        perceptions: list[ReceivedMessage],
    ) -> list[Action]:
        """Make simple decisions.

        Args:
            agent: Agent
            context: Simulation context
            perceptions: Perceptions

        Returns:
            list[Action]: Actions
        """
        if not perceptions:
            return []

        # Simple rule: respond to most trusted message
        best = max(perceptions, key=lambda p: p.trust_at_reception)

        if best.trust_at_reception > 0.5:
            return [
                Action(
                    action_type="speak",
                    content=f"Received: {best.message.content[:100]}",
                    topic_id=best.message.topic_id,
                    timestamp=datetime.now(),
                )
            ]

        return []
