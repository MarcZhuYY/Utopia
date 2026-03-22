"""Action buffer for deferred execution.

Buffers actions generated during a tick for ordered execution,
preventing race conditions and enabling prioritization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utopia.core.pydantic_models import ActionWithTrace


@dataclass
class BufferedAction:
    """An action waiting in the buffer."""

    action: ActionWithTrace
    priority: int = 0
    tick: int = 0
    dependencies: set[str] = field(default_factory=set)
    resolved: bool = False


class ActionBuffer:
    """Buffer for deferred action execution.

    Provides:
    - Priority ordering (higher priority = earlier execution)
    - Dependency resolution (actions wait for prerequisites)
    - Conflict detection (preventing contradictory actions)
    - Batch processing (execute all buffered actions)
    """

    def __init__(self):
        """Initialize action buffer."""
        self._buffer: list[BufferedAction] = []
        self._executed: list[ActionWithTrace] = []
        self._conflicts_detected = 0

    def add(
        self,
        action: ActionWithTrace,
        priority: int = 0,
        dependencies: Optional[set[str]] = None,
    ) -> None:
        """Add action to buffer.

        Args:
            action: Action to buffer
            priority: Execution priority
            dependencies: IDs of actions that must execute first
        """
        buffered = BufferedAction(
            action=action,
            priority=priority,
            tick=action.tick,
            dependencies=dependencies or set(),
        )
        self._buffer.append(buffered)

        # Sort by priority (descending)
        self._buffer.sort(key=lambda b: (-b.priority, b.tick))

    def add_batch(
        self,
        actions: list[tuple[ActionWithTrace, int]],  # (action, priority)
    ) -> None:
        """Add multiple actions to buffer.

        Args:
            actions: List of (action, priority) tuples
        """
        for action, priority in actions:
            self.add(action, priority)

    def execute_all(
        self,
        executor: Callable[[ActionWithTrace], Any],
        check_conflicts: bool = True,
    ) -> list[Any]:
        """Execute all buffered actions in priority order.

        Args:
            executor: Function to execute each action
            check_conflicts: Whether to detect conflicts

        Returns:
            List of execution results
        """
        results = []
        executed_ids: set[str] = set()

        while self._buffer:
            # Find next executable action
            executable = None
            executable_idx = -1

            for i, buffered in enumerate(self._buffer):
                # Check if dependencies are resolved
                deps_satisfied = all(
                    dep in executed_ids for dep in buffered.dependencies
                )

                if deps_satisfied:
                    executable = buffered
                    executable_idx = i
                    break

            if executable is None:
                # Deadlock - remaining actions have unsatisfied dependencies
                break

            # Remove from buffer
            self._buffer.pop(executable_idx)

            # Check for conflicts
            if check_conflicts and self._has_conflict(executable, executed_ids):
                self._conflicts_detected += 1
                continue

            # Execute
            result = executor(executable.action)
            results.append(result)
            executed_ids.add(executable.action.action_id)
            self._executed.append(executable.action)

        return results

    def _has_conflict(
        self,
        buffered: BufferedAction,
        executed_ids: set[str],
    ) -> bool:
        """Check if action conflicts with already executed actions.

        Args:
            buffered: Action to check
            executed_ids: IDs of already executed actions

        Returns:
            True if conflict detected
        """
        action = buffered.action

        # Check for contradictory stance changes
        for executed in self._executed:
            if executed.action_id in executed_ids:
                # Same agent trying to do contradictory things
                if (
                    executed.agent_id == action.agent_id
                    and executed.action_type == action.action_type
                    and executed.topic_id == action.topic_id
                    and executed.action_type in ["speak", "change_belief"]
                ):
                    # Allow if content is similar (reinforcement)
                    if executed.content != action.content:
                        return True

        return False

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def get_pending(self) -> list[BufferedAction]:
        """Get all pending actions.

        Returns:
            List of pending actions
        """
        return self._buffer.copy()

    def get_executed(self) -> list[ActionWithTrace]:
        """Get executed actions.

        Returns:
            List of executed actions
        """
        return self._executed.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "pending": len(self._buffer),
            "executed": len(self._executed),
            "conflicts_detected": self._conflicts_detected,
        }
