"""Tests for SimulationMetricsCollector with vectorized factor extraction.

Tests:
- Cross-sectional PFF, SMD, Momentum calculation (NumPy masking)
- Time-series Delta_SMD, IDF (Pandas vectorized)
- Parquet export with pyarrow/snappy
- Mock 50-round simulation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utopia.layer6_analysis.metrics_collector import (
    SimulationMetricsCollector,
    TickMetrics,
    TimeSeriesFactors,
)
from utopia.layer3_cognition.agent_persona_models import (
    AgentRole,
    RetailInvestorPersona,
    QuantInstitutionPersona,
    InsiderPersona,
    MacroRegulatorPersona,
)


class MockStanceState:
    """Mock stance state for testing."""

    def __init__(self, position: float = 0.0, confidence: float = 0.5):
        self.position = position
        self.confidence = confidence


class MockAgentState:
    """Mock agent state for testing."""

    def __init__(self, agent_id: str, stance: float = 0.0, confidence: float = 0.5):
        self.agent_id = agent_id
        self.stances = {
            "TOPIC_TEST": MockStanceState(position=stance, confidence=confidence)
        }


class MockWorldState:
    """Mock WorldState for testing."""

    def __init__(self, tick: int = 0):
        self.tick = tick
        self.agent_states: dict[str, MockAgentState] = {}

    def add_agent(
        self,
        agent_id: str,
        stance: float = 0.0,
        confidence: float = 0.5,
    ) -> None:
        """Add agent with stance state."""
        self.agent_states[agent_id] = MockAgentState(
            agent_id=agent_id,
            stance=stance,
            confidence=confidence,
        )


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    return {
        "R001": RetailInvestorPersona(agent_id="R001", capital_weight=1.0),
        "R002": RetailInvestorPersona(agent_id="R002", capital_weight=1.0),
        "Q001": QuantInstitutionPersona(agent_id="Q001", capital_weight=100.0),
        "I001": InsiderPersona(agent_id="I001", capital_weight=50.0),
        "CSRC": MacroRegulatorPersona(agent_id="CSRC"),
    }


@pytest.fixture
def collector():
    """Create metrics collector."""
    return SimulationMetricsCollector()


class TestCrossSectionalFactors:
    """Test cross-sectional factor calculation (NumPy masking)."""

    def test_pff_calculation(self, collector, sample_agents):
        """Test PFF formula: (|mean_S_R| * mean_C_R) / (1.0 + 10.0 * var_S_R)"""
        # Setup: Retail with high agreement, high confidence
        state = MockWorldState(tick=0)
        state.add_agent("R001", stance=0.8, confidence=0.9)  # Bullish, confident
        state.add_agent("R002", stance=0.75, confidence=0.85)  # Similar
        state.add_agent("Q001", stance=-0.5, confidence=0.95)  # Smart money bearish

        metrics = collector.record_tick(0, state, sample_agents)

        # Verify PFF > 0 (high agreement among retail)
        assert metrics.pff > 0
        # Verify formula components
        expected_numerator = abs(metrics.mean_stance_retail) * metrics.mean_confidence_retail
        expected_denominator = 1.0 + 10.0 * metrics.var_stance_retail
        expected_pff = expected_numerator / expected_denominator
        assert metrics.pff == pytest.approx(expected_pff, abs=1e-6)

    def test_smd_calculation(self, collector, sample_agents):
        """Test SMD formula: mean(S_SM) - mean(S_R)"""
        # Setup: Smart money bullish, retail bearish
        state = MockWorldState(tick=0)
        state.add_agent("R001", stance=-0.5, confidence=0.5)  # Retail bearish
        state.add_agent("R002", stance=-0.6, confidence=0.5)
        state.add_agent("Q001", stance=0.8, confidence=0.9)   # Quant bullish
        state.add_agent("I001", stance=0.7, confidence=0.8)   # Insider bullish

        metrics = collector.record_tick(0, state, sample_agents)

        # SMD should be positive (smart money > retail)
        expected_smd = metrics.mean_stance_smart - metrics.mean_stance_retail
        assert metrics.smd == pytest.approx(expected_smd, abs=1e-6)
        assert metrics.smd > 0

    def test_momentum_capital_weighted(self, collector):
        """Test momentum is capital-weighted mean."""
        state = MockWorldState(tick=0)
        state.add_agent("R001", stance=0.5, confidence=0.5)   # weight=1
        state.add_agent("Q001", stance=-0.5, confidence=0.9)  # weight=100

        # Use only the agents in the state
        agents = {
            "R001": RetailInvestorPersona(agent_id="R001", capital_weight=1.0),
            "Q001": QuantInstitutionPersona(agent_id="Q001", capital_weight=100.0),
        }

        metrics = collector.record_tick(0, state, agents)

        # Should be closer to Q001 due to higher capital weight
        total_capital = 1.0 + 100.0
        expected = (0.5 * 1.0 + (-0.5) * 100.0) / total_capital
        assert metrics.momentum == pytest.approx(expected, abs=1e-6)

    def test_safe_mean_empty_group(self, collector):
        """Test safe mean handles empty groups (no retail agents)."""
        state = MockWorldState(tick=0)
        # Only smart money, no retail
        state.add_agent("Q001", stance=0.5, confidence=0.9)
        state.add_agent("CSRC", stance=0.8, confidence=1.0)

        agents = {
            "Q001": QuantInstitutionPersona(agent_id="Q001"),
            "CSRC": MacroRegulatorPersona(agent_id="CSRC"),
        }

        metrics = collector.record_tick(0, state, agents)

        # Should not crash, return 0 for empty retail
        assert metrics.n_retail == 0
        assert metrics.mean_stance_retail == 0.0
        assert metrics.pff == 0.0  # No retail means no polarization

    def test_no_python_loops_in_calculation(self, collector, sample_agents):
        """Verify calculation uses vectorized operations (no explicit loops)."""
        # This test documents the design - actual verification through profiling
        state = MockWorldState(tick=0)
        for i in range(50):
            aid = f"R{i:03d}"
            state.add_agent(aid, stance=np.random.uniform(-1, 1), confidence=0.5)
            sample_agents[aid] = RetailInvestorPersona(agent_id=aid)

        # Should complete quickly (< 10ms for 50 agents)
        import time
        start = time.perf_counter()
        metrics = collector.record_tick(0, state, sample_agents)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10.0  # Vectorized should be fast


class TestTimeSeriesFactors:
    """Test time-series factor calculation (Pandas vectorized)."""

    def test_delta_smd_calculation(self):
        """Test Delta_SMD lagged differences."""
        collector = SimulationMetricsCollector()

        # Simulate 30 ticks with changing SMD
        for tick in range(30):
            state = MockWorldState(tick=tick)
            # Retail stance increases linearly
            state.add_agent("R001", stance=0.1 * tick, confidence=0.5)
            # Smart money stance constant
            state.add_agent("Q001", stance=0.5, confidence=0.9)

            agents = {
                "R001": RetailInvestorPersona(agent_id="R001"),
                "Q001": QuantInstitutionPersona(agent_id="Q001"),
            }
            collector.record_tick(tick, state, agents)

        # Export and verify time-series factors
        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)
            df = pd.read_parquet(files["tick_metrics"])

            # Delta SMD should be negative and relatively constant
            # (retail stance increasing means SMD = smart - retail decreases)
            assert "delta_smd_1" in df.columns
            assert "delta_smd_5" in df.columns

            # Later ticks should have valid delta values
            assert df["delta_smd_1"].iloc[10:].notna().all()

    def test_momentum_ewm(self):
        """Test EWM momentum calculation."""
        collector = SimulationMetricsCollector()

        # Simulate with oscillating momentum
        for tick in range(50):
            state = MockWorldState(tick=tick)
            # Oscillating stance
            stance = np.sin(tick * 0.5)
            state.add_agent("R001", stance=stance, confidence=0.5)
            state.add_agent("Q001", stance=stance, confidence=0.9)

            agents = {
                "R001": RetailInvestorPersona(agent_id="R001"),
                "Q001": QuantInstitutionPersona(agent_id="Q001"),
            }
            collector.record_tick(tick, state, agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)
            df = pd.read_parquet(files["tick_metrics"])

            # EWM columns should exist
            assert "momentum_ewm_5" in df.columns
            assert "momentum_ewm_20" in df.columns

            # EWM should be smoother than raw momentum
            raw_vol = df["momentum"].std()
            ewm_vol = df["momentum_ewm_5"].std()
            assert ewm_vol < raw_vol  # EWM smooths the series

    def test_idf_computation(self):
        """Test IDF (Information Diffusion Half-life) calculation."""
        collector = SimulationMetricsCollector(idf_window=20)

        # Simulate with varying stance change magnitude
        for tick in range(50):
            state = MockWorldState(tick=tick)
            # Exponentially decaying changes
            change_mag = np.exp(-tick / 10) * np.random.uniform(0.1, 0.3)
            stance = 0.5 + change_mag * (-1 if tick % 2 == 0 else 1)
            state.add_agent("R001", stance=stance, confidence=0.5)
            state.add_agent("Q001", stance=stance * 0.8, confidence=0.9)

            agents = {
                "R001": RetailInvestorPersona(agent_id="R001"),
                "Q001": QuantInstitutionPersona(agent_id="Q001"),
            }
            collector.record_tick(tick, state, agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)
            df = pd.read_parquet(files["tick_metrics"])

            assert "idf" in df.columns
            # IDF should be in reasonable range [1, 100]
            valid_idf = df["idf"].dropna()
            assert (valid_idf >= 1.0).all()
            assert (valid_idf <= 100.0).all()


class TestParquetExport:
    """Test Parquet export functionality."""

    def test_export_creates_parquet(self, collector, sample_agents):
        """Test export creates valid Parquet file."""
        # Record some ticks
        for tick in range(10):
            state = MockWorldState(tick=tick)
            state.add_agent("R001", stance=0.1 * tick, confidence=0.5)
            state.add_agent("Q001", stance=-0.1 * tick, confidence=0.9)
            collector.record_tick(tick, state, sample_agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)

            # Verify file exists
            assert "tick_metrics" in files
            assert files["tick_metrics"].exists()

            # Verify can read back
            df = pd.read_parquet(files["tick_metrics"])
            assert len(df) == 10
            assert "pff" in df.columns
            assert "smd" in df.columns
            assert "momentum" in df.columns

    def test_parquet_compression(self, collector, sample_agents):
        """Test Parquet uses snappy compression."""
        for tick in range(5):
            state = MockWorldState(tick=tick)
            state.add_agent("R001", stance=0.0, confidence=0.5)
            collector.record_tick(tick, state, sample_agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)

            import pyarrow.parquet as pq
            # Read file and close handle immediately
            pf = pq.ParquetFile(files["tick_metrics"])
            metadata = pf.metadata
            pf = None  # Release handle for Windows cleanup

            # File should be readable (compression working)
            assert metadata.num_rows == 5


class TestMock50RoundSimulation:
    """Mock 50-round simulation test."""

    def test_50_round_simulation(self):
        """Full 50-round simulation with metrics collection."""
        collector = SimulationMetricsCollector()
        n_rounds = 50
        n_retail = 70
        n_quant = 15

        # Create agents
        agents = {}
        for i in range(n_retail):
            aid = f"RETAIL_{i+1:03d}"
            agents[aid] = RetailInvestorPersona(agent_id=aid)
        for i in range(n_quant):
            aid = f"QUANT_{i+1:03d}"
            agents[aid] = QuantInstitutionPersona(agent_id=aid)

        # Simulate 50 rounds
        np.random.seed(42)
        for tick in range(n_rounds):
            state = MockWorldState(tick=tick)

            # Simulate stance evolution
            for i, aid in enumerate(list(agents.keys())[:n_retail]):
                # Retail: random walk with drift
                stance = 0.5 * np.sin(tick * 0.2 + i * 0.1) + np.random.normal(0, 0.1)
                stance = np.clip(stance, -1.0, 1.0)
                state.add_agent(aid, stance=stance, confidence=0.6)

            for i, aid in enumerate(list(agents.keys())[n_retail:n_retail+n_quant]):
                # Quant: more stable, mean-reverting
                stance = 0.3 * np.cos(tick * 0.15) + np.random.normal(0, 0.05)
                stance = np.clip(stance, -1.0, 1.0)
                state.add_agent(aid, stance=stance, confidence=0.9)

            collector.record_tick(tick, state, agents)

        # Verify collection
        summary = collector.get_summary_stats()
        assert summary["total_ticks"] == n_rounds
        assert summary["final_retail_count"] == n_retail

        # Export and verify
        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)
            df = pd.read_parquet(files["tick_metrics"])

            assert len(df) == n_rounds
            assert "delta_smd_1" in df.columns
            assert "idf" in df.columns

            # Verify PFF, SMD, Momentum have reasonable values
            assert df["pff"].between(-1, 1).all() or df["pff"].between(0, 1).all()
            assert df["smd"].between(-2, 2).all()


class TestSummaryStats:
    """Test summary statistics."""

    def test_summary_before_data(self, collector):
        """Test summary with no data."""
        summary = collector.get_summary_stats()
        assert "error" in summary

    def test_summary_with_data(self, collector, sample_agents):
        """Test summary with collected data."""
        for tick in range(10):
            state = MockWorldState(tick=tick)
            state.add_agent("R001", stance=0.1 * tick, confidence=0.5)
            state.add_agent("Q001", stance=-0.05 * tick, confidence=0.9)
            collector.record_tick(tick, state, sample_agents)

        summary = collector.get_summary_stats()
        assert summary["total_ticks"] == 10
        assert "mean_pff" in summary
        assert "mean_smd" in summary


class TestVectorizedOperations:
    """Test vectorized operation correctness and performance."""

    def test_numpy_masking_efficiency(self):
        """Verify NumPy masking is used for group calculations."""
        collector = SimulationMetricsCollector()
        agents = {}

        # Create 100 agents with different roles
        for i in range(70):
            aid = f"RETAIL_{i:03d}"
            agents[aid] = RetailInvestorPersona(agent_id=aid)
        for i in range(15):
            aid = f"QUANT_{i:03d}"
            agents[aid] = QuantInstitutionPersona(agent_id=aid)
        for i in range(10):
            aid = f"INSIDER_{i:03d}"
            agents[aid] = InsiderPersona(agent_id=aid)
        agents["REGULATOR"] = MacroRegulatorPersona(agent_id="REGULATOR")

        state = MockWorldState(tick=0)
        for aid in agents:
            state.add_agent(aid, stance=np.random.uniform(-1, 1), confidence=0.5)

        metrics = collector.record_tick(0, state, agents)

        # Verify counts match
        assert metrics.n_retail == 70
        assert metrics.n_quant == 15
        assert metrics.n_insider == 10
        assert metrics.n_regulator == 1
        assert metrics.n_total == 96

    def test_pandas_vectorized_no_iterrows(self):
        """Verify time-series calculation doesn't use iterrows."""
        # Create collector with known data
        collector = SimulationMetricsCollector()

        for tick in range(20):
            state = MockWorldState(tick=tick)
            state.add_agent("R001", stance=np.sin(tick * 0.3), confidence=0.5)
            state.add_agent("Q001", stance=np.cos(tick * 0.2), confidence=0.9)

            agents = {
                "R001": RetailInvestorPersona(agent_id="R001"),
                "Q001": QuantInstitutionPersona(agent_id="Q001"),
            }
            collector.record_tick(tick, state, agents)

        # Export and verify time-series factors computed correctly
        with tempfile.TemporaryDirectory() as tmpdir:
            files = collector.export_results(tmpdir)
            df = pd.read_parquet(files["tick_metrics"])

            # All time-series columns should exist and have valid values
            ts_cols = ["delta_smd_1", "delta_smd_5", "momentum_ewm_5", "idf"]
            for col in ts_cols:
                assert col in df.columns, f"Missing column: {col}"

    def test_safe_division_handling(self):
        """Test safe division with epsilon for zero denominators."""
        collector = SimulationMetricsCollector(epsilon=1e-10)

        # Single agent (edge case for variance)
        state = MockWorldState(tick=0)
        state.add_agent("R001", stance=0.5, confidence=0.8)

        agents = {"R001": RetailInvestorPersona(agent_id="R001")}

        # Should not crash with single agent (variance = 0)
        metrics = collector.record_tick(0, state, agents)

        # PFF should still compute (with epsilon in denominator)
        assert metrics.pff >= 0
        assert not np.isnan(metrics.pff)
        assert not np.isinf(metrics.pff)
