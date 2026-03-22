"""Simulation report generator.

Generates comprehensive reports from simulation results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from utopia.layer5_engine.engine import SimulationResult
from utopia.layer6_analysis.findings import Finding, FindingType, extract_findings


@dataclass
class SimulationReport:
    """Complete simulation report.

    Attributes:
        summary: Executive summary
        key_findings: List of key findings
        agent_behavior_traces: Per-agent behavior traces
        group_dynamics: Group dynamics report
        opinion_evolution: How opinions changed over time
        predictions: Future predictions
        metadata: Simulation metadata
    """

    summary: str = ""
    key_findings: list[Finding] = field(default_factory=list)
    agent_behavior_traces: dict[str, list[dict]] = field(default_factory=dict)
    group_dynamics: dict[str, Any] = field(default_factory=dict)
    opinion_evolution: dict[str, list[dict]] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_findings": [f.to_dict() for f in self.key_findings],
            "agent_behavior_traces": self.agent_behavior_traces,
            "group_dynamics": self.group_dynamics,
            "opinion_evolution": self.opinion_evolution,
            "predictions": self.predictions,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate Markdown report.

        Returns:
            str: Markdown formatted report
        """
        lines = [
            "# Simulation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            self.summary,
            "",
            "## Key Findings",
        ]

        for i, finding in enumerate(self.key_findings, 1):
            lines.append(f"\n### {i}. {finding.title} ({finding.type.value})")
            lines.append(f"\n{finding.description}")
            lines.append(f"\n**Confidence**: {finding.confidence:.0%}")
            if finding.evidence:
                lines.append("\n**Evidence:**")
                for e in finding.evidence:
                    lines.append(f"- {e}")

        lines.extend([
            "",
            "## Group Dynamics",
            f"- Polarization Score: {self.group_dynamics.get('polarization_score', 'N/A')}",
            f"- Major Factions: {len(self.group_dynamics.get('clusters', []))}",
            "",
            "## Metadata",
            f"- Simulation Duration: {self.metadata.get('duration_seconds', 0):.2f}s",
            f"- Agents: {self.metadata.get('agent_count', 0)}",
            f"- Ticks: {self.metadata.get('final_tick', 0)}",
        ])

        return "\n".join(lines)


class ReportGenerator:
    """Generates simulation reports.

    Analyzes simulation results and generates:
    - Executive summary
    - Key findings (trends, anomalies, predictions)
    - Behavior traces
    - Group dynamics
    """

    def __init__(self, result: SimulationResult):
        """Initialize report generator.

        Args:
            result: Simulation result
        """
        self.result = result

    def generate(self) -> SimulationReport:
        """Generate complete report.

        Returns:
            SimulationReport: Generated report
        """
        # Extract key findings
        findings = self._extract_findings()

        # Generate summary
        summary = self._write_summary(findings)

        # Compile group dynamics
        dynamics = self._compile_dynamics()

        # Generate predictions
        predictions = self._generate_predictions()

        return SimulationReport(
            summary=summary,
            key_findings=findings,
            agent_behavior_traces=self.result.agent_traces,
            group_dynamics=dynamics,
            opinion_evolution=self.result.stance_history,
            predictions=predictions,
            metadata={
                "tick_count": self.result.final_tick,
                "agent_count": self.result.agent_count,
                "duration_seconds": self.result.duration_seconds,
                "llm_calls": self.result.total_llm_calls,
                "cost_estimate": self.result.estimated_cost,
                "converged": self.result.converged,
                "convergence_reason": self.result.convergence_reason,
                "domain": self.result.domain,
            },
        )

    def _extract_findings(self) -> list[Finding]:
        """Extract key findings from simulation.

        Returns:
            list[Finding]: Key findings
        """
        # Use the standalone extraction function
        return extract_findings(self.result)

    def _write_summary(self, findings: list[Finding]) -> str:
        """Write executive summary.

        Args:
            findings: Key findings

        Returns:
            str: Summary text
        """
        # Count findings by type
        trends = [f for f in findings if f.type == FindingType.TREND]
        anomalies = [f for f in findings if f.type == FindingType.ANOMALY]
        predictions = [f for f in findings if f.type == FindingType.PREDICTION]

        summary_parts = [
            f"This simulation ran for {self.result.final_tick} ticks with {self.result.agent_count} agents.",
        ]

        if self.result.converged:
            summary_parts.append(f"The simulation converged: {self.result.convergence_reason}.")
        else:
            summary_parts.append("The simulation reached maximum ticks without converging.")

        if trends:
            summary_parts.append(f"Identified {len(trends)} key trend(s).")
        if anomalies:
            summary_parts.append(f"Detected {len(anomalies)} anomaly(ies).")
        if predictions:
            summary_parts.append(f"Generated {len(predictions)} prediction(s).")

        return " ".join(summary_parts)

    def _compile_dynamics(self) -> dict[str, Any]:
        """Compile group dynamics report.

        Returns:
            dict: Group dynamics summary
        """
        if not self.result.dynamics_history:
            return {}

        latest = self.result.dynamics_history[-1]

        return {
            "polarization_score": latest.get("polarization", {}).get("score", 0.0),
            "clusters": latest.get("polarization", {}).get("clusters", []),
            "final_tick": self.result.final_tick,
        }

    def _generate_predictions(self) -> dict[str, Any]:
        """Generate future predictions.

        Returns:
            dict: Predictions
        """
        # TODO: Replace with LLM-based prediction generation
        # For MVP, use simple extrapolation

        predictions = {
            "timeframe": "simulation_end",
            "confidence": 0.5,
            "summary": "Predictions require LLM integration for detailed forecasts.",
        }

        return predictions

    def save_report(self, output_dir: str, format: str = "json") -> Path:
        """Save report to file.

        Args:
            output_dir: Output directory
            format: Output format (json/markdown)

        Returns:
            Path: Output file path
        """
        report = self.generate()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            filepath = output_path / "simulation_report.md"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
        else:
            filepath = output_path / "simulation_report.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        return filepath
