"""Double-buffered world state management for causal consistency.

This module implements the double-buffering pattern to prevent
"causal confusion" in the simulation:
- Read buffer: Immutable snapshot of state at tick N (for all reads)
- Write buffer: Mutable working state for tick N+1 (for all writes)

At tick end, buffers are swapped atomically.

Mathematical guarantee: All agents read the same world state when
making decisions, eliminating order-dependent artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from utopia.core.pydantic_models import (
    ActionBufferEntry,
    ActionWithTrace,
    StanceState,
    WorldStateSnapshot,
)


@dataclass
class AgentState:
    """Complete state of an agent at a point in time.

    This is a mutable working copy used during simulation ticks.
    """

    agent_id: str
    stances: dict[str, StanceState] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)  # x, y coordinates
    current_action: Optional[str] = None
    last_action_tick: int = 0

    def copy(self) -> "AgentState":
        """Create deep copy of agent state."""
        return AgentState(
            agent_id=self.agent_id,
            stances={k: v.copy() for k, v in self.stances.items()},
            position=self.position,
            current_action=self.current_action,
            last_action_tick=self.last_action_tick,
        )


@dataclass
class WorldState:
    """Mutable world state for a simulation tick.

    This is the working state that gets modified during tick processing.
    """

    tick: int
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    trust_matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    pending_actions: list[ActionWithTrace] = field(default_factory=list)
    committed_actions: list[ActionWithTrace] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "WorldState":
        """Create deep copy of world state."""
        return WorldState(
            tick=self.tick,
            agent_states={k: v.copy() for k, v in self.agent_states.items()},
            trust_matrix=self.trust_matrix.copy(),
            pending_actions=self.pending_actions.copy(),
            committed_actions=self.committed_actions.copy(),
            event_log=self.event_log.copy(),
        )

    def to_snapshot(self) -> WorldStateSnapshot:
        """Convert to immutable snapshot."""
        from utopia.core.pydantic_models import StanceState

        agent_stances = {}
        for aid, state in self.agent_states.items():
            # Get first stance or create default
            if state.stances:
                first_stance = next(iter(state.stances.values()))
                agent_stances[aid] = first_stance
            else:
                agent_stances[aid] = StanceState(
                    topic_id="default",
                    position=0.0,
                    confidence=0.0,
                )

        return WorldStateSnapshot(
            tick=self.tick,
            agent_states=agent_stances,
            relationship_matrix=self.trust_matrix.copy(),
            event_log=self.event_log.copy(),
        )


class WorldStateBuffer:
    """Double-buffered world state manager.

    Implements the classic double-buffering pattern:
    - read_state: Immutable snapshot for current tick (all reads go here)
    - write_state: Mutable working state for next tick (all writes go here)

    At end of tick: swap buffers atomically

    This ensures causal consistency: all agents read the SAME state
    when making decisions, regardless of execution order.
    """

    def __init__(self, initial_state: Optional[WorldState] = None):
        """Initialize double buffer.

        Args:
            initial_state: Initial world state (creates empty if None)
        """
        if initial_state is None:
            initial_state = WorldState(tick=0)

        # Read buffer: immutable snapshot for current tick
        self._read_state: WorldState = initial_state.copy()

        # Write buffer: mutable working state for next tick
        self._write_state: WorldState = initial_state.copy()

        # Snapshot history for debugging/analysis
        self._history: list[WorldStateSnapshot] = []
        self._max_history: int = 100

        # Pending action buffer
        self._action_buffer: list[ActionBufferEntry] = []

    @property
    def read_state(self) -> WorldState:
        """Get read-only state for current tick.

        All agent decision reads should use this state to ensure
        consistency across the tick.
        """
        return self._read_state

    @property
    def write_state(self) -> WorldState:
        """Get writeable state for next tick.

        All state modifications should target this buffer.
        """
        return self._write_state

    @property
    def current_tick(self) -> int:
        """Get current simulation tick."""
        return self._read_state.tick

    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state from read buffer.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent state or None
        """
        return self._read_state.agent_states.get(agent_id)

    def get_trust(self, agent_a: str, agent_b: str) -> float:
        """Get trust level from read buffer.

        Args:
            agent_a: First agent
            agent_b: Second agent

        Returns:
            Trust level (-1 to 1, 0 if not found)
        """
        return self._read_state.trust_matrix.get((agent_a, agent_b), 0.0)

    def update_agent_state(
        self,
        agent_id: str,
        stance_updates: Optional[dict[str, StanceState]] = None,
        position: Optional[tuple[float, float]] = None,
    ) -> None:
        """Update agent state in write buffer.

        Args:
            agent_id: Agent to update
            stance_updates: New stance states
            position: New position
        """
        if agent_id not in self._write_state.agent_states:
            self._write_state.agent_states[agent_id] = AgentState(agent_id=agent_id)

        agent = self._write_state.agent_states[agent_id]

        if stance_updates:
            agent.stances.update(stance_updates)

        if position:
            agent.position = position

    def update_trust(
        self,
        agent_a: str,
        agent_b: str,
        new_trust: float,
    ) -> None:
        """Update trust level in write buffer.

        Args:
            agent_a: First agent
            agent_b: Second agent
            new_trust: New trust value
        """
        self._write_state.trust_matrix[(agent_a, agent_b)] = float(
            np.clip(new_trust, -1.0, 1.0)
        )

    def buffer_action(
        self,
        action: ActionWithTrace,
        priority: int = 0,
    ) -> None:
        """Buffer an action for next tick.

        Args:
            action: Action to buffer
            priority: Execution priority (higher = earlier)
        """
        entry = ActionBufferEntry(
            action=action,
            source_tick=self.current_tick,
            priority=priority,
        )
        self._action_buffer.append(entry)

        # Keep sorted by priority
        self._action_buffer.sort(key=lambda e: e.priority, reverse=True)

    def commit_buffered_actions(self) -> list[ActionWithTrace]:
        """Commit all buffered actions to write state.

        Returns:
            List of committed actions
        """
        committed = []

        for entry in self._action_buffer:
            self._write_state.committed_actions.append(entry.action)
            committed.append(entry.action)

        self._action_buffer.clear()
        return committed

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event to write buffer.

        Args:
            event_type: Type of event
            data: Event data
        """
        self._write_state.event_log.append({
            "tick": self.current_tick,
            "type": event_type,
            **data,
        })

    def swap_buffers(self) -> WorldStateSnapshot:
        """Swap read and write buffers atomically.

        This is the critical operation that ensures causal consistency:
        1. Archive current read state to history
        2. Swap read/write pointers
        3. Reset write state as copy of new read state
        4. Increment tick in write state

        Returns:
            Snapshot of committed state
        """
        # Archive current read state
        snapshot = self._read_state.to_snapshot()
        self._history.append(snapshot)

        # Trim history if needed
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Swap buffers
        self._read_state = self._write_state

        # Create new write state as copy with incremented tick
        self._write_state = self._read_state.copy()
        self._write_state.tick += 1
        self._write_state.pending_actions.clear()
        self._write_state.committed_actions.clear()
        self._write_state.event_log.clear()

        return snapshot

    def rollback_write_buffer(self) -> None:
        """Rollback write buffer to match read buffer.

        Use this if tick processing fails and you need to
        discard all pending changes.
        """
        self._write_state = self._read_state.copy()
        self._action_buffer.clear()

    def get_history(self, ticks: Optional[int] = None) -> list[WorldStateSnapshot]:
        """Get historical snapshots.

        Args:
            ticks: Number of recent ticks (None = all)

        Returns:
            List of historical snapshots
        """
        if ticks is None:
            return self._history.copy()
        return self._history[-ticks:].copy()

    def get_state_at_tick(self, tick: int) -> Optional[WorldStateSnapshot]:
        """Get state snapshot at specific tick.

        Args:
            tick: Target tick

        Returns:
            Snapshot or None if not in history
        """
        for snapshot in reversed(self._history):
            if snapshot.tick == tick:
                return snapshot
        return None

    def compute_state_delta(self, tick_a: int, tick_b: int) -> dict[str, Any]:
        """Compute difference between two ticks.

        Args:
            tick_a: First tick
            tick_b: Second tick

        Returns:
            Delta dictionary
        """
        state_a = self.get_state_at_tick(tick_a)
        state_b = self.get_state_at_tick(tick_b)

        if state_a is None or state_b is None:
            return {"error": "One or both ticks not in history"}

        # Compute stance changes
        stance_changes = {}
        for agent_id in set(state_a.agent_states.keys()) | set(state_b.agent_states.keys()):
            a_stance = state_a.agent_states.get(agent_id)
            b_stance = state_b.agent_states.get(agent_id)

            if a_stance != b_stance:
                stance_changes[agent_id] = {
                    "from": a_stance.position if a_stance else None,
                    "to": b_stance.position if b_stance else None,
                }

        # Compute trust changes
        trust_changes = {}
        all_pairs = set(state_a.relationship_matrix.keys()) | set(state_b.relationship_matrix.keys())
        for pair in all_pairs:
            a_trust = state_a.relationship_matrix.get(pair, 0)
            b_trust = state_b.relationship_matrix.get(pair, 0)
            if abs(a_trust - b_trust) > 0.01:
                trust_changes[f"{pair[0]}-{pair[1]}"] = {
                    "from": a_trust,
                    "to": b_trust,
                }

        return {
            "tick_a": tick_a,
            "tick_b": tick_b,
            "tick_delta": tick_b - tick_a,
            "stance_changes": stance_changes,
            "trust_changes": trust_changes,
            "event_count_a": len(state_a.event_log),
            "event_count_b": len(state_b.event_log),
        }


class TickCoordinator:
    """Coordinates tick-based simulation execution with double buffering.

    Provides high-level API for running simulation ticks with proper
    isolation and state management.
    """

    def __init__(self, buffer: WorldStateBuffer):
        """Initialize tick coordinator.

        Args:
            buffer: Double-buffered state manager
        """
        self.buffer = buffer
        self._tick_callbacks: list[callable] = []
        self._is_processing = False

    def register_tick_callback(self, callback: callable) -> None:
        """Register callback to run each tick.

        Args:
            callback: Function(buffer) to call each tick
        """
        self._tick_callbacks.append(callback)

    def run_tick(self) -> WorldStateSnapshot:
        """Execute single simulation tick.

        Flow:
        1. All agents read from read_state (consistent view)
        2. All agents write to write_state (isolated changes)
        3. Run callbacks
        4. Commit buffered actions
        5. Swap buffers atomically

        Returns:
            Snapshot of committed state
        """
        if self._is_processing:
            raise RuntimeError("Tick already in progress")

        self._is_processing = True

        try:
            # Execute callbacks (agents make decisions)
            for callback in self._tick_callbacks:
                callback(self.buffer)

            # Commit all buffered actions
            self.buffer.commit_buffered_actions()

            # Swap buffers to complete tick
            snapshot = self.buffer.swap_buffers()

            return snapshot

        finally:
            self._is_processing = False

    def run_ticks(self, n: int) -> list[WorldStateSnapshot]:
        """Run multiple ticks.

        Args:
            n: Number of ticks to run

        Returns:
            List of snapshots
        """
        snapshots = []
        for _ in range(n):
            snapshot = self.run_tick()
            snapshots.append(snapshot)
        return snapshots

    def get_current_state_summary(self) -> dict[str, Any]:
        """Get summary of current state.

        Returns:
            State summary dictionary
        """
        read_state = self.buffer.read_state

        return {
            "current_tick": self.buffer.current_tick,
            "agent_count": len(read_state.agent_states),
            "trust_relation_count": len(read_state.trust_matrix),
            "pending_actions": len(self.buffer._action_buffer),
            "committed_actions_this_tick": len(read_state.committed_actions),
            "event_log_size": len(read_state.event_log),
            "history_size": len(self.buffer._history),
        }
