"""L6: Result Analysis Layer.

This layer generates simulation reports and collects metrics.
"""

from utopia.layer6_analysis.report_generator import ReportGenerator, SimulationReport
from utopia.layer6_analysis.findings import Finding, FindingType, extract_findings
from utopia.layer6_analysis.metrics_collector import (
    SimulationMetricsCollector,
    TickMetrics,
    TimeSeriesFactors,
)

__all__ = [
    "ReportGenerator",
    "SimulationReport",
    "Finding",
    "FindingType",
    "extract_findings",
    "SimulationMetricsCollector",
    "TickMetrics",
    "TimeSeriesFactors",
]
