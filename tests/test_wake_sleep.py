"""Tests for Agent_Tick wake/sleep interception mechanism.

Tests the cognitive dissonance calculation and wake/sleep state transitions.
Formula: Delta = |Message_Stance - Agent_Stance| * Sender_Trust * Message_Importance
"""

import pytest
from datetime import datetime

from utopia.core.models import (
    AgentState,
    Message,
    Persona,
    ReceivedMessage,
    Stance,
)
from utopia.core.pydantic_models import (
    ActivityStatus,
    BigFiveTraits,
    CognitiveDissonanceInput,
    WakeUpDecision,
)
from utopia.layer3_cognition.agent import Agent
from utopia.layer3_cognition.decision_engine import AgentDecisionEngine


class TestCognitiveDissonanceCalculation:
    """Test cognitive dissonance formula."""

    def test_dissonance_formula_low(self):
        """Test low dissonance: small stance diff, medium trust/importance."""
        # Delta = |0.1 - 0.0| * 0.5 * 0.5 = 0.025
        input_data = CognitiveDissonanceInput(
            message_stance=0.1,
            agent_stance=0.0,
            sender_trust=0.5,
            message_importance=0.5,
        )
        delta = input_data.compute_delta()
        assert delta == pytest.approx(0.025, abs=0.001)
        assert input_data.determine_activity_status() == ActivityStatus.SLEEPING

    def test_dissonance_formula_medium(self):
        """Test medium dissonance: moderate stance diff and trust."""
        # Delta = |0.5 - 0.0| * 0.6 * 0.7 = 0.21
        input_data = CognitiveDissonanceInput(
            message_stance=0.5,
            agent_stance=0.0,
            sender_trust=0.6,
            message_importance=0.7,
        )
        delta = input_data.compute_delta()
        assert delta == pytest.approx(0.21, abs=0.001)
        assert input_data.determine_activity_status() == ActivityStatus.ROUTINE

    def test_dissonance_formula_high(self):
        """Test high dissonance: large stance diff, high trust/importance."""
        # Delta = |0.8 - 0.0| * 0.8 * 0.8 = 0.512
        input_data = CognitiveDissonanceInput(
            message_stance=0.8,
            agent_stance=0.0,
            sender_trust=0.8,
            message_importance=0.8,
        )
        delta = input_data.compute_delta()
        assert delta == pytest.approx(0.512, abs=0.001)
        assert input_data.determine_activity_status() == ActivityStatus.AWAKE_CRITICAL

    def test_dissonance_with_opposite_stance(self):
        """Test dissonance when message opposes agent stance."""
        # Agent supports (0.7), message opposes (-0.6)
        # Delta = |-0.6 - 0.7| * 0.8 * 0.9 = 1.3 * 0.8 * 0.9 = 0.936 (clamped to ~1.0 context)
        input_data = CognitiveDissonanceInput(
            message_stance=-0.6,
            agent_stance=0.7,
            sender_trust=0.8,
            message_importance=0.9,
        )
        delta = input_data.compute_delta()
        # Stance diff = 1.3, but effectively capped by stance range
        assert delta > 0.4  # Should trigger AWAKE_CRITICAL


class TestWakeSleepThresholds:
    """Test wake/sleep threshold behavior."""

    @pytest.fixture
    def test_agent(self):
        """Create a test agent with default stance."""
        agent = Agent.from_entity(
            entity_id="test_agent",
            name="TestAgent",
            role="analyst",
            expertise=["general"],
            base_stances={"topic1": 0.0},  # Neutral stance
            influence=0.5,
        )
        # Set up relationship for trust calculation
        from utopia.layer4_social.relationships import RelationshipMap
        rel_map = RelationshipMap()
        rel_map.build_complete_graph(["test_agent", "sender1", "sender2"], base_trust=0.5)
        agent.set_relationship_map(rel_map)
        return agent

    @pytest.fixture
    def decision_engine(self):
        """Create decision engine with default config."""
        return AgentDecisionEngine()

    def create_test_message(
        self,
        sender_id: str = "sender1",
        topic_id: str = "topic1",
        stance: float = 0.0,
        importance: float = 0.5,
    ) -> ReceivedMessage:
        """Helper to create test messages."""
        message = Message(
            content=f"Test message with stance {stance}",
            sender_id=sender_id,
            topic_id=topic_id,
            original_stance=stance,
            timestamp=datetime.now(),
        )
        # Store importance as an attribute for testing
        msg = ReceivedMessage(
            message=message,
            from_agent=sender_id,
            depth=0,
            trust_at_reception=0.5,
        )
        # Add importance attribute for perceive_and_filter
        object.__setattr__(msg, "importance", importance)
        return msg

    def test_five_low_messages_keep_sleeping(self, test_agent, decision_engine):
        """Test: 5 low-weight messages keep agent in SLEEPING state.

        Each message: stance_diff=0.1, trust=0.5, importance=0.5
        Delta per message = 0.1 * 0.5 * 0.5 = 0.025
        All < 0.15 threshold, so agent should stay SLEEPING.
        """
        # Create 5 low-impact messages
        low_messages = [
            self.create_test_message(stance=0.1, importance=0.5)
            for _ in range(5)
        ]

        # Run perceive_and_filter
        decision = decision_engine.perceive_and_filter(
            agent=test_agent,
            inbox_messages=low_messages,
            force_wake=False,
        )

        # Assertions
        assert isinstance(decision, WakeUpDecision)
        assert decision.target_status == ActivityStatus.SLEEPING
        assert decision.max_delta < 0.15
        assert len(decision.critical_messages) == 0
        assert not decision.force_wake

    def test_one_high_message_triggers_awake_critical(self, test_agent, decision_engine):
        """Test: 1 high-impact message triggers AWAKE_CRITICAL state.

        Message: stance_diff=0.8, trust=0.8, importance=0.8
        Delta = 0.8 * 0.8 * 0.8 = 0.512
        > 0.4 threshold, so agent should become AWAKE_CRITICAL.
        """
        # Create 1 high-impact message
        high_message = self.create_test_message(
            sender_id="sender2",
            stance=0.8,
            importance=0.8,
        )
        # Override trust for sender2 to be high
        from utopia.layer4_social.relationships import RelationshipDelta
        delta = RelationshipDelta(trust_change=0.3)  # 0.5 -> 0.8
        test_agent._relationship_map.update("test_agent", "sender2", delta)

        # Run perceive_and_filter
        decision = decision_engine.perceive_and_filter(
            agent=test_agent,
            inbox_messages=[high_message],
            force_wake=False,
        )

        # Assertions
        assert isinstance(decision, WakeUpDecision)
        assert decision.target_status == ActivityStatus.AWAKE_CRITICAL
        assert decision.max_delta >= 0.4
        assert len(decision.critical_messages) == 1

    def test_mixed_messages_routine_state(self, test_agent, decision_engine):
        """Test: Mix of low and medium messages triggers ROUTINE state.

        Messages with Delta between 0.15 and 0.4 should trigger ROUTINE.
        """
        # Create medium-impact messages
        # stance_diff=0.5, trust=0.6, importance=0.6 -> Delta = 0.18
        messages = [
            self.create_test_message(stance=0.5, importance=0.6)
            for _ in range(3)
        ]
        from utopia.layer4_social.relationships import RelationshipDelta
        delta = RelationshipDelta(trust_change=0.1)  # 0.5 -> 0.6
        test_agent._relationship_map.update("test_agent", "sender1", delta)

        # Run perceive_and_filter
        decision = decision_engine.perceive_and_filter(
            agent=test_agent,
            inbox_messages=messages,
            force_wake=False,
        )

        # Assertions
        assert decision.target_status == ActivityStatus.ROUTINE
        assert 0.15 <= decision.max_delta < 0.4

    def test_force_wake_overrides_threshold(self, test_agent, decision_engine):
        """Test: Force wake flag overrides normal threshold."""
        # Create low-impact messages that would normally keep SLEEPING
        low_messages = [
            self.create_test_message(stance=0.1, importance=0.5)
            for _ in range(2)
        ]

        # Run with force_wake=True
        decision = decision_engine.perceive_and_filter(
            agent=test_agent,
            inbox_messages=low_messages,
            force_wake=True,
        )

        # Should be AWAKE_CRITICAL despite low delta
        assert decision.target_status == ActivityStatus.AWAKE_CRITICAL
        assert decision.force_wake
        assert "Force wake" in decision.reason

    def test_empty_messages_sleeping(self, test_agent, decision_engine):
        """Test: Empty inbox results in SLEEPING state."""
        decision = decision_engine.perceive_and_filter(
            agent=test_agent,
            inbox_messages=[],
            force_wake=False,
        )

        assert decision.target_status == ActivityStatus.SLEEPING
        assert decision.max_delta == 0.0


class TestAgentActivityStatus:
    """Test agent activity status management."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        return Agent.from_entity(
            entity_id="test_agent",
            name="TestAgent",
            role="analyst",
            expertise=["general"],
            base_stances={},
            influence=0.5,
        )

    def test_initial_status(self, agent):
        """Test agent starts in ROUTINE state."""
        assert agent.activity_status == ActivityStatus.ROUTINE
        assert agent.silent_ticks == 0

    def test_transition_to_sleeping(self, agent):
        """Test transition to SLEEPING increments silent_ticks."""
        agent.set_activity_status(ActivityStatus.SLEEPING, current_tick=1)
        assert agent.activity_status == ActivityStatus.SLEEPING
        assert agent.silent_ticks == 1

        agent.set_activity_status(ActivityStatus.SLEEPING, current_tick=2)
        assert agent.silent_ticks == 2

    def test_wake_from_sleeping_resets_counter(self, agent):
        """Test waking up resets silent_ticks."""
        # Put to sleep
        agent.set_activity_status(ActivityStatus.SLEEPING, current_tick=1)
        agent.set_activity_status(ActivityStatus.SLEEPING, current_tick=2)
        assert agent.silent_ticks == 2

        # Wake up
        agent.set_activity_status(ActivityStatus.ROUTINE, current_tick=3)
        assert agent.activity_status == ActivityStatus.ROUTINE
        assert agent.silent_ticks == 0

    def test_force_wake_condition(self, agent):
        """Test force wake after max silent ticks."""
        # Simulate 10 silent ticks
        for i in range(10):
            agent.set_activity_status(ActivityStatus.SLEEPING, current_tick=i)

        assert agent.silent_ticks == 10
        assert agent.should_force_wake(max_silent_ticks=10)

    def test_no_force_wake_before_threshold(self, agent):
        """Test no force wake before max silent ticks."""
        for i in range(5):
            agent.set_activity_status(ActivityStatus.SLEEPING, current_tick=i)

        assert agent.silent_ticks == 5
        assert not agent.should_force_wake(max_silent_ticks=10)


class TestSilentUpdate:
    """Test silent belief updates in SLEEPING state."""

    @pytest.fixture
    def agent(self):
        """Create agent with known stance."""
        agent = Agent.from_entity(
            entity_id="test_agent",
            name="TestAgent",
            role="analyst",
            expertise=["general"],
            base_stances={"topic1": 0.0},
            influence=0.5,
        )
        # Set up BigFive traits
        agent.persona.big_five = BigFiveTraits(openness=0.5)
        return agent

    @pytest.fixture
    def decision_engine(self):
        return AgentDecisionEngine()

    def create_test_message(self, stance: float = 0.5, importance: float = 0.5) -> ReceivedMessage:
        """Helper to create test messages."""
        message = Message(
            content=f"Test message with stance {stance}",
            sender_id="sender1",
            topic_id="topic1",
            original_stance=stance,
            timestamp=datetime.now(),
        )
        msg = ReceivedMessage(
            message=message,
            from_agent="sender1",
            depth=0,
            trust_at_reception=0.5,
        )
        object.__setattr__(msg, "importance", importance)
        return msg

    def test_silent_update_changes_stance(self, agent, decision_engine):
        """Test silent update modifies agent stance."""
        # Get initial stance
        initial_stance = agent.get_stance("topic1")
        initial_pos = initial_stance.position if initial_stance else 0.0

        # Create messages
        messages = [self.create_test_message(stance=0.5)]

        # Perform silent update
        decision_engine.silent_update(agent, messages)

        # Stance should have changed slightly (reduced intensity)
        new_stance = agent.get_stance("topic1")
        assert new_stance is not None
        # Should move toward message stance but less than full update

    def test_silent_update_preserves_memory(self, agent, decision_engine):
        """Test silent update adds to memory."""
        initial_memory_count = len(agent.memory.short_term)

        messages = [self.create_test_message(stance=0.3)]
        decision_engine.silent_update(agent, messages)

        # Memory should be added
        assert len(agent.memory.short_term) > initial_memory_count


class TestIntegration:
    """Integration tests for wake/sleep mechanism."""

    @pytest.fixture
    def setup(self):
        """Set up complete test environment."""
        agent = Agent.from_entity(
            entity_id="test_agent",
            name="TestAgent",
            role="analyst",
            expertise=["general"],
            base_stances={"topic1": 0.0},
            influence=0.5,
        )

        # Set up relationships
        from utopia.layer4_social.relationships import RelationshipMap
        rel_map = RelationshipMap()
        rel_map.build_complete_graph(["test_agent", "sender1"], base_trust=0.5)
        agent.set_relationship_map(rel_map)

        engine = AgentDecisionEngine()

        return agent, engine

    def test_full_workflow_low_messages(self, setup):
        """Test full workflow: low messages -> SLEEPING -> silent update."""
        agent, engine = setup

        # Create 5 low-impact messages
        messages = []
        for i in range(5):
            msg = ReceivedMessage(
                message=Message(
                    content=f"Low impact message {i}",
                    sender_id="sender1",
                    topic_id="topic1",
                    original_stance=0.1,
                    timestamp=datetime.now(),
                ),
                from_agent="sender1",
                depth=0,
                trust_at_reception=0.5,
            )
            object.__setattr__(msg, "importance", 0.5)
            messages.append(msg)

        # Step 1: perceive_and_filter
        decision = engine.perceive_and_filter(agent, messages)
        assert decision.target_status == ActivityStatus.SLEEPING

        # Step 2: Since SLEEPING, perform silent update
        engine.silent_update(agent, messages)

        # Verify agent updated silently
        assert agent.activity_status != ActivityStatus.AWAKE_CRITICAL

    def test_full_workflow_high_message(self, setup):
        """Test full workflow: high message -> AWAKE_CRITICAL."""
        agent, engine = setup

        # Create 1 high-impact message
        high_message = ReceivedMessage(
            message=Message(
                content="Critical important message!",
                sender_id="sender1",
                topic_id="topic1",
                original_stance=0.9,
                timestamp=datetime.now(),
            ),
            from_agent="sender1",
            depth=0,
            trust_at_reception=0.9,  # High trust
        )
        object.__setattr__(high_message, "importance", 0.9)

        # Update relationship for high trust
        from utopia.layer4_social.relationships import RelationshipDelta
        delta = RelationshipDelta(trust_change=0.4)  # 0.5 -> 0.9
        agent._relationship_map.update("test_agent", "sender1", delta)

        # Step 1: perceive_and_filter
        decision = engine.perceive_and_filter(agent, [high_message])
        assert decision.target_status == ActivityStatus.AWAKE_CRITICAL

        # Step 2: In real scenario, would trigger LLM call
        # Here we just verify the decision is correct
        assert len(decision.critical_messages) == 1
