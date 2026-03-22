"""Convergence detection for simulation.

Determines when simulation has reached stable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from utopia.core.config import WorldRules


@dataclass
class ConvergenceResult:
    """Result of convergence check.

    Attributes:
        converged: Whether simulation has converged
        reason: Reason for convergence (or why not)
        confidence: Confidence in convergence判定 (0-1)
    """

    converged: bool = False
    reason: str = ""
    confidence: float = 0.0


class ConvergenceDetector:
    """Detects when simulation has converged.

    Convergence conditions:
    1. Stance convergence: Group stances std < threshold
    2. Behavior saturation: No new action patterns
    3. Max ticks reached
    """

    def __init__(self, rules: Optional[WorldRules] = None):
        """Initialize detector.

        Args:
            rules: World rules
        """
        self.rules = rules or WorldRules()
        self._stance_history: list[float] = []

    def check(
        self,
        dynamics_history: list[dict[str, Any]],
    ) -> ConvergenceResult:
        """Check if simulation has converged.

        Args:
            dynamics_history: History of dynamics measurements

        Returns:
            ConvergenceResult: Convergence result
        """
        if len(dynamics_history) < 3:
            return ConvergenceResult(converged=False, reason="Insufficient history")

        # Check 1: Stance convergence (check if std is decreasing)
        stance_stds = []
        for dyn in dynamics_history[-5:]:
            if "polarization" in dyn:
                pol = dyn["polarization"]
                if "std" in pol:
                    stance_stds.append(pol["std"])

        if len(stance_stds) >= 3:
            if self._is_decreasing(stance_stds) and stance_stds[-1] < self.rules.convergence_threshold:
                return ConvergenceResult(
                    converged=True,
                    reason="Group stances converged",
                    confidence=0.9,
                )

        # Check 2: Behavior saturation
        recent_actions = []
        for dyn in dynamics_history[-3:]:
            if "action_count" in dyn:
                recent_actions.append(dyn["action_count"])

        if len(recent_actions) >= 3:
            if all(abs(recent_actions[i] - recent_actions[i + 1]) < 1 for i in range(len(recent_actions) - 1)):
                return ConvergenceResult(
                    converged=True,
                    reason="Behavior saturated - no new dynamics",
                    confidence=0.7,
                )

        # Not converged
        return ConvergenceResult(converged=False, reason="Still evolving")

    def _is_decreasing(self, values: list[float], tolerance: float = 0.01) -> bool:
        """Check if values are consistently decreasing.

        Args:
            values: List of values
            tolerance: Tolerance for decrease

        Returns:
            bool: True if decreasing
        """
        if len(values) < 2:
            return False

        for i in range(len(values) - 1):
            if values[i] < values[i + 1] - tolerance:
                return False
        return True

    def _compute_stance_std(self, agents: list[Any]) -> float:
        """Compute standard deviation of agent stances.

        Args:
            agents: List of agents

        Returns:
            float: Standard deviation
        """
        import numpy as np

        stances = []
        for agent in agents:
            for stance in agent.beliefs.stances.values():
                stances.append(stance.position)

        if not stances:
            return 0.0

        return float(np.std(stances))
