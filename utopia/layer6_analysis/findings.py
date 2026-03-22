"""Key findings extraction from simulation results.

Identifies trends, anomalies, and predictions from simulation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from utopia.layer5_engine.engine import SimulationResult


class FindingType(str, Enum):
    """Type of finding."""

    TREND = "trend"
    ANOMALY = "anomaly"
    PREDICTION = "prediction"
    RELATIONSHIP = "relationship"


@dataclass
class Finding:
    """Represents a key finding from simulation.

    Attributes:
        type: Finding type
        title: Short title
        description: Detailed description
        confidence: Confidence level (0-1)
        evidence: List of supporting evidence
    """

    type: FindingType
    title: str
    description: str
    confidence: float = 0.5
    evidence: list[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


def extract_findings(result: SimulationResult) -> list[Finding]:
    """Extract key findings from simulation result.

    This is a simplified extraction for MVP.
    Phase 2: Use LLM to analyze and generate findings.

    Args:
        result: Simulation result

    Returns:
        list[Finding]: Extracted findings
    """
    findings = []

    # Analyze polarization
    findings.extend(_analyze_polarization(result))

    # Analyze trends
    findings.extend(_analyze_trends(result))

    # Detect anomalies
    findings.extend(_detect_anomalies(result))

    # Generate predictions
    findings.extend(_generate_predictions(result))

    return findings


def _analyze_polarization(result: SimulationResult) -> list[Finding]:
    """Analyze polarization from dynamics history.

    Args:
        result: Simulation result

    Returns:
        list[Finding]: Polarization findings
    """
    findings = []

    for dyn in result.dynamics_history:
        if "polarization" in dyn:
            pol = dyn["polarization"]
            score = pol.get("score", 0.0)

            if score > 0.5:
                findings.append(Finding(
                    type=FindingType.TREND,
                    title="High Group Polarization Detected",
                    description=f"The group showed significant polarization with a score of {score:.2f}. "
                               f"Agents have formed distinct clusters with opposing views.",
                    confidence=score,
                    evidence=[
                        f"Polarization score: {score:.2f}",
                        f"Number of clusters: {len(pol.get('clusters', []))}",
                    ],
                ))
            elif score > 0.3:
                findings.append(Finding(
                    type=FindingType.TREND,
                    title="Moderate Group Polarization",
                    description=f"Moderate polarization (score: {score:.2f}) indicates some division "
                               f"among agents on key topics.",
                    confidence=score * 0.8,
                ))

    return findings


def _analyze_trends(result: SimulationResult) -> list[Finding]:
    """Analyze trends from stance history.

    Args:
        result: Simulation result

    Returns:
        list[Finding]: Trend findings
    """
    findings = []

    # Analyze stance evolution
    for topic_id, history in result.stance_history.items():
        if len(history) < 2:
            continue

        positions = [h.get("position", 0) for h in history if "position" in h]
        if len(positions) < 2:
            continue

        # Calculate trend direction
        change = positions[-1] - positions[0]
        if abs(change) > 0.3:
            direction = "positive" if change > 0 else "negative"
            findings.append(Finding(
                type=FindingType.TREND,
                title=f"Stance Shift on {topic_id}",
                description=f"The group's stance on {topic_id} shifted {direction} by {abs(change):.2f} "
                           f"during the simulation.",
                confidence=min(0.9, abs(change) * 2),
                evidence=[
                    f"Initial position: {positions[0]:.2f}",
                    f"Final position: {positions[-1]:.2f}",
                    f"Total change: {change:.2f}",
                ],
            ))

    return findings


def _detect_anomalies(result: SimulationResult) -> list[Finding]:
    """Detect anomalies in simulation results.

    Args:
        result: Simulation result

    Returns:
        list[Finding]: Anomaly findings
    """
    findings = []

    # Check for rapid convergence
    if result.converged and result.final_tick < result.final_tick * 0.3:
        findings.append(Finding(
            type=FindingType.ANOMALY,
            title="Rapid Convergence",
            description=f"Simulation converged very quickly at tick {result.final_tick}. "
                       f"This may indicate strong initial consensus or overly homogeneous agents.",
            confidence=0.7,
            evidence=[
                f"Converged at tick: {result.final_tick}",
                f"Total ticks run: {result.final_tick}",
            ],
        ))

    # Check for erratic behavior
    if len(result.dynamics_history) >= 3:
        scores = []
        for dyn in result.dynamics_history:
            if "polarization" in dyn:
                scores.append(dyn["polarization"].get("score", 0.0))

        if len(scores) >= 3:
            import numpy as np
            variance = float(np.var(scores))
            if variance > 0.1:
                findings.append(Finding(
                    type=FindingType.ANOMALY,
                    title="Erratic Group Dynamics",
                    description=f"High variance in group dynamics indicates unstable or erratic behavior. "
                               f"Variance: {variance:.3f}",
                    confidence=min(0.9, variance * 5),
                ))

    return findings


def _generate_predictions(result: SimulationResult) -> list[Finding]:
    """Generate predictions based on simulation results.

    Args:
        result: Simulation result

    Returns:
        list[Finding]: Prediction findings
    """
    findings = []

    # Simple prediction based on current trend
    if result.stance_history:
        for topic_id, history in result.stance_history.items():
            if len(history) < 3:
                continue

            recent = history[-3:]
            positions = [h.get("position", 0) for h in recent]

            # Extrapolate
            import numpy as np
            if len(positions) >= 2:
                slope = float(np.polyfit(range(len(positions)), positions, 1)[0])

                if abs(slope) > 0.05:
                    direction = "increasing" if slope > 0 else "decreasing"
                    findings.append(Finding(
                        type=FindingType.PREDICTION,
                        title=f"Stance Likely to Continue {direction.capitalize()}",
                        description=f"The stance on {topic_id} shows a consistent {direction} trend. "
                                   f"This trajectory is likely to continue unless external events intervene.",
                        confidence=0.6,
                        evidence=[
                            f"Recent slope: {slope:.3f}",
                            f"Based on last {len(recent)} observations",
                        ],
                    ))

    return findings
