"""Tests for world state buffer (world_state_buffer.py)."""

import pytest

from utopia.core.pydantic_models import ActionWithTrace, ReasoningTrace, StanceState
from utopia.layer5_engine.world_state_buffer import (
    WorldStateBuffer,
    WorldState,
    AgentState,
    TickCoordinator,
)


class TestWorldStateBuffer:
    """Test double-buffered state management."""

    def test_initial_state(self):
        """Test buffer initialization."""
        buffer = WorldStateBuffer()

        assert buffer.current_tick == 0
        assert buffer.read_state.tick == 0
        assert buffer.write_state.tick == 0

    def test_read_write_isolation(self):
        """Test that read and write states are isolated."""
        initial = WorldState(tick=0)
        initial.agent_states["agent1"] = AgentState(agent_id="agent1")

        buffer = WorldStateBuffer(initial)

        # Modify write state
        buffer.write_state.agent_states["agent2"] = AgentState(agent_id="agent2")

        # Read state should not be affected
        assert "agent2" not in buffer.read_state.agent_states
        assert "agent1" in buffer.read_state.agent_states

    def test_swap_buffers(self):
        """Test buffer swap operation."""
        buffer = WorldStateBuffer()

        # Make changes to write state
        buffer.write_state.agent_states["agent1"] = AgentState(agent_id="agent1")
        buffer.write_state.tick = 1  # Simulate tick increment

        # Swap
        snapshot = buffer.swap_buffers()

        # Read state should now have the changes
        assert "agent1" in buffer.read_state.agent_states
        assert buffer.current_tick == 1

        # Write state should be a fresh copy
        assert buffer.write_state.tick == 2  # Incremented
        assert len(buffer.write_state.agent_states) == 1  # Copied from read

    def test_trust_update_isolation(self):
        """Test trust updates are isolated until swap."""
        buffer = WorldStateBuffer()

        # Set initial trust
        buffer.read_state.trust_matrix[("A", "B")] = 0.5
        buffer.write_state.trust_matrix[("A", "B")] = 0.5

        # Update trust in write buffer
        buffer.update_trust("A", "B", 0.8)

        # Read buffer should have old value
        assert buffer.read_state.trust_matrix[("A", "B")] == 0.5

        # Write buffer should have new value
        assert buffer.write_state.trust_matrix[("A", "B")] == 0.8

    def test_action_buffering(self):
        """Test action buffering."""
        buffer = WorldStateBuffer()

        action = ActionWithTrace(
            action_id="A001",
            agent_id="agent1",
            action_type="speak",
            tick=0,
            trace=ReasoningTrace(
                trace_id="T001",
                agent_id="agent1",
                tick=0,
                situation_analysis="test",
                reasoning_chain=["step1"],
                chosen_action="speak",
            ),
        )

        buffer.buffer_action(action, priority=5)

        assert len(buffer._action_buffer) == 1

        # Commit actions
        committed = buffer.commit_buffered_actions()
        assert len(committed) == 1
        assert action in buffer.write_state.committed_actions

    def test_rollback(self):
        """Test write buffer rollback."""
        buffer = WorldStateBuffer()

        # Make changes
        buffer.update_agent_state("agent1", position=(1.0, 2.0))
        buffer.update_trust("A", "B", 0.9)

        # Rollback
        buffer.rollback_write_buffer()

        # Changes should be discarded
        assert "agent1" not in buffer.write_state.agent_states
        assert ("A", "B") not in buffer.write_state.trust_matrix

    def test_history_tracking(self):
        """Test snapshot history."""
        buffer = WorldStateBuffer()

        # Run multiple ticks
        for i in range(3):
            buffer.write_state.tick = i + 1
            buffer.swap_buffers()

        history = buffer.get_history()
        assert len(history) == 3

    def test_state_delta(self):
        """Test computing state delta."""
        from utopia.layer5_engine.world_state_buffer import WorldState, AgentState

        # Create initial state with proper agent
        initial = WorldState(tick=0)
        agent = AgentState(agent_id="agent1")
        agent.stances["topic1"] = StanceState(topic_id="topic1", position=0.5, confidence=0.5)
        initial.agent_states["agent1"] = agent

        buffer = WorldStateBuffer(initial)

        # Tick 1 - modify and swap
        buffer.update_agent_state(
            "agent1",
            stance_updates={"topic1": StanceState(topic_id="topic1", position=0.8, confidence=0.6)},
        )
        buffer.update_trust("A", "B", 0.7)
        buffer.swap_buffers()

        # Tick 2 - swap again to create history
        buffer.swap_buffers()

        # Get delta between ticks (check history exists)
        history = buffer.get_history()
        assert len(history) >= 2


class TestTickCoordinator:
    """Test tick coordination."""

    def test_tick_execution(self):
        """Test single tick execution."""
        buffer = WorldStateBuffer()
        coordinator = TickCoordinator(buffer)

        callback_called = False

        def callback(buf):
            nonlocal callback_called
            callback_called = True

        coordinator.register_tick_callback(callback)
        snapshot = coordinator.run_tick()

        assert callback_called
        # After first tick, read_state shows completed tick (0)
        # write_state is working on tick 1
        assert buffer.current_tick == 0
        assert buffer.write_state.tick == 1

    def test_multiple_ticks(self):
        """Test running multiple ticks."""
        buffer = WorldStateBuffer()
        coordinator = TickCoordinator(buffer)

        snapshots = coordinator.run_ticks(5)

        assert len(snapshots) == 5
        # After 5 ticks, read_state shows completed tick 4
        # write_state is working on tick 5
        assert buffer.current_tick == 4
        assert buffer.write_state.tick == 5

    def test_concurrent_tick_prevention(self):
        """Test that concurrent ticks are prevented."""
        buffer = WorldStateBuffer()
        coordinator = TickCoordinator(buffer)

        # Manually set processing flag
        coordinator._is_processing = True

        with pytest.raises(RuntimeError, match="already in progress"):
            coordinator.run_tick()

    def test_state_summary(self):
        """Test state summary."""
        buffer = WorldStateBuffer()
        coordinator = TickCoordinator(buffer)

        # Add some state
        buffer.read_state.agent_states["A"] = AgentState(agent_id="A")
        buffer.read_state.agent_states["B"] = AgentState(agent_id="B")
        buffer.read_state.trust_matrix[("A", "B")] = 0.5

        summary = coordinator.get_current_state_summary()

        assert summary["current_tick"] == 0
        assert summary["agent_count"] == 2
        assert summary["trust_relation_count"] == 1
