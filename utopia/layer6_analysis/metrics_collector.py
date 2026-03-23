"""Metrics Collector - Cross-sectional and time-series Alpha factor extraction.

Uses vectorized NumPy/Pandas operations only - no Python for loops.

Alpha Factors:
- PFF (Polarization Fragility Factor): Measures system fragility from retail polarization
- SMD (Smart Money Divergence): Difference between smart money and retail stances
- Momentum: Capital-weighted mean stance
- IDF (Information Diffusion Half-life): Rate of information decay
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from utopia.layer5_engine.world_state_buffer import WorldState, AgentState
    from utopia.layer3_cognition.agent_persona_models import BaseAgentPersona


class TickMetrics(BaseModel):
    """Cross-sectional metrics for a single tick."""

    model_config = {"frozen": True}

    tick: int = Field(..., description="Simulation tick")
    timestamp: float = Field(..., description="Unix timestamp")

    # Population counts
    n_retail: int = Field(..., ge=0, description="Number of retail investors")
    n_quant: int = Field(..., ge=0, description="Number of quant institutions")
    n_insider: int = Field(..., ge=0, description="Number of insiders")
    n_regulator: int = Field(..., ge=0, description="Number of regulators")
    n_total: int = Field(..., ge=0, description="Total agents")

    # Stance statistics (population means)
    mean_stance_retail: float = Field(..., description="Mean retail stance")
    mean_stance_quant: float = Field(..., description="Mean quant stance")
    mean_stance_insider: float = Field(..., description="Mean insider stance")
    mean_stance_regulator: float = Field(..., description="Mean regulator stance")
    mean_stance_smart: float = Field(..., description="Mean smart money stance")
    mean_stance_overall: float = Field(..., description="Capital-weighted mean stance")

    # Confidence statistics
    mean_confidence_retail: float = Field(..., description="Mean retail confidence")
    mean_confidence_smart: float = Field(..., description="Mean smart money confidence")

    # Variance (for polarization)
    var_stance_retail: float = Field(..., description="Retail stance variance")
    var_stance_overall: float = Field(..., description="Overall stance variance")

    # Alpha Factors (cross-sectional)
    pff: float = Field(..., description="Polarization Fragility Factor")
    smd: float = Field(..., description="Smart Money Divergence")
    momentum: float = Field(..., description="Capital-weighted momentum")

    # Trust network metrics
    mean_trust: float = Field(default=0.0, description="Mean trust level")
    trust_density: float = Field(default=0.0, description="Non-zero trust ratio")


class TimeSeriesFactors(BaseModel):
    """Time-series factors computed from tick history."""

    model_config = {"frozen": True}

    tick: int = Field(..., description="Current tick")

    # SMD changes
    delta_smd_1: float = Field(..., description="SMD change from t-1")
    delta_smd_5: float = Field(..., description="SMD change from t-5")
    delta_smd_20: float = Field(..., description="SMD change from t-20")

    # EWM momentum
    momentum_ewm_5: float = Field(..., description="5-tick EWM momentum")
    momentum_ewm_20: float = Field(..., description="20-tick EWM momentum")

    # Information Diffusion Factor (half-life)
    idf: float = Field(..., description="Information Diffusion Half-life (ticks)")

    # Volatility
    realized_vol_20: float = Field(..., description="20-tick realized volatility")


@dataclass
class SimulationMetricsCollector:
    """Collect and extract Alpha factors from simulation ticks.

    Design principles:
    1. Cross-sectional factors per tick: NumPy masking (no loops)
    2. Time-series factors at export: Vectorized Pandas operations
    3. Safe division: replace 0 with NaN, ffill, add epsilon
    4. Parquet output: pyarrow/snappy compression

    Example:
        >>> collector = SimulationMetricsCollector()
        >>> for tick in range(100):
        ...     state = run_simulation_tick()
        ...     collector.record_tick(tick, state, agents)
        >>> collector.export_results("output/")
    """

    # Raw data storage (list of dicts for memory efficiency)
    _tick_records: list[dict[str, Any]] = field(default_factory=list)
    _agent_snapshots: list[dict[str, Any]] = field(default_factory=list)

    # Configuration
    epsilon: float = 1e-10
    idf_window: int = 50  # Ticks for IDF calculation

    def record_tick(
        self,
        tick: int,
        world_state: WorldState,
        agents: dict[str, BaseAgentPersona],
    ) -> TickMetrics:
        """Record cross-sectional metrics for one tick.

        Args:
            tick: Current tick number
            world_state: World state buffer
            agents: Dictionary of agent personas

        Returns:
            TickMetrics for this tick
        """
        timestamp = pd.Timestamp.now().timestamp()

        # Build arrays using vectorized operations (no loops)
        agent_data = self._extract_agent_arrays(world_state, agents)

        # Cross-sectional factor calculation
        metrics = self._compute_cross_sectional(tick, timestamp, agent_data)

        # Store raw data for time-series analysis
        self._tick_records.append(metrics.model_dump())
        self._agent_snapshots.append({
            "tick": tick,
            "timestamp": timestamp,
            "agent_data": agent_data,
        })

        return metrics

    def _extract_agent_arrays(
        self,
        world_state: WorldState,
        agents: dict[str, BaseAgentPersona],
    ) -> dict[str, np.ndarray]:
        """Extract agent data into NumPy arrays using masking.

        Returns dict with arrays:
        - agent_ids: str array
        - roles: str array
        - stances: float array
        - confidences: float array
        - capital_weights: float array
        - is_retail: bool mask
        - is_smart: bool mask (quant + insider + regulator)
        """
        n = len(agents)

        # Pre-allocate arrays
        agent_ids = np.empty(n, dtype=object)
        roles = np.empty(n, dtype=object)
        stances = np.zeros(n, dtype=np.float64)
        confidences = np.zeros(n, dtype=np.float64)
        capital_weights = np.zeros(n, dtype=np.float64)

        # Fill arrays (single loop for data extraction)
        for i, (aid, persona) in enumerate(agents.items()):
            agent_ids[i] = aid
            roles[i] = persona.role.value
            capital_weights[i] = persona.capital_weight

            # Get stance from world state
            agent_state = world_state.agent_states.get(aid)
            if agent_state and agent_state.stances:
                # Use first stance or aggregate
                first_stance = next(iter(agent_state.stances.values()))
                stances[i] = first_stance.position
                confidences[i] = first_stance.confidence

        # Create masks for vectorized filtering
        is_retail = roles == "retail"
        is_quant = roles == "quant"
        is_insider = roles == "insider"
        is_regulator = roles == "regulator"
        is_smart = is_quant | is_insider | is_regulator

        return {
            "agent_ids": agent_ids,
            "roles": roles,
            "stances": stances,
            "confidences": confidences,
            "capital_weights": capital_weights,
            "is_retail": is_retail,
            "is_smart": is_smart,
            "is_quant": is_quant,
            "is_insider": is_insider,
            "is_regulator": is_regulator,
        }

    def _compute_cross_sectional(
        self,
        tick: int,
        timestamp: float,
        data: dict[str, np.ndarray],
    ) -> TickMetrics:
        """Compute cross-sectional factors using NumPy masking."""
        stances = data["stances"]
        confidences = data["confidences"]
        capital_weights = data["capital_weights"]
        is_retail = data["is_retail"]
        is_smart = data["is_smart"]
        is_quant = data["is_quant"]
        is_insider = data["is_insider"]
        is_regulator = data["is_regulator"]

        # Safe mean calculation (handle empty groups)
        def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
            """Compute mean with safe handling of empty groups."""
            if not np.any(mask):
                return 0.0
            return float(np.mean(arr[mask]))

        def safe_var(arr: np.ndarray, mask: np.ndarray) -> float:
            """Compute variance with safe handling."""
            if np.sum(mask) < 2:
                return 0.0
            return float(np.var(arr[mask], ddof=1))

        # Population statistics
        n_retail = int(np.sum(is_retail))
        n_quant = int(np.sum(is_quant))
        n_insider = int(np.sum(is_insider))
        n_regulator = int(np.sum(is_regulator))
        n_total = len(stances)

        # Stance means by group (vectorized masking)
        mean_stance_retail = safe_mean(stances, is_retail)
        mean_stance_quant = safe_mean(stances, is_quant)
        mean_stance_insider = safe_mean(stances, is_insider)
        mean_stance_regulator = safe_mean(stances, is_regulator)
        mean_stance_smart = safe_mean(stances, is_smart)

        # Capital-weighted overall mean
        total_capital = np.sum(capital_weights)
        if total_capital > 0:
            mean_stance_overall = float(
                np.sum(stances * capital_weights) / total_capital
            )
        else:
            mean_stance_overall = float(np.mean(stances))

        # Confidence means
        mean_confidence_retail = safe_mean(confidences, is_retail)
        mean_confidence_smart = safe_mean(confidences, is_smart)

        # Variance
        var_stance_retail = safe_var(stances, is_retail)
        var_stance_overall = safe_var(stances, np.ones(len(stances), dtype=bool))

        # ===== Alpha Factor Calculation =====

        # PFF: Polarization Fragility Factor
        # PFF = (|mean_S_R| * mean_C_R) / (1.0 + 10.0 * var_S_R)
        pff_numerator = abs(mean_stance_retail) * mean_confidence_retail
        pff_denominator = 1.0 + 10.0 * var_stance_retail
        pff = pff_numerator / (pff_denominator + self.epsilon)

        # SMD: Smart Money Divergence
        # SMD = mean(S_SM) - mean(S_R)
        smd = mean_stance_smart - mean_stance_retail

        # Momentum: Capital-weighted stance
        momentum = mean_stance_overall

        return TickMetrics(
            tick=tick,
            timestamp=timestamp,
            n_retail=n_retail,
            n_quant=n_quant,
            n_insider=n_insider,
            n_regulator=n_regulator,
            n_total=n_total,
            mean_stance_retail=mean_stance_retail,
            mean_stance_quant=mean_stance_quant,
            mean_stance_insider=mean_stance_insider,
            mean_stance_regulator=mean_stance_regulator,
            mean_stance_smart=mean_stance_smart,
            mean_stance_overall=mean_stance_overall,
            mean_confidence_retail=mean_confidence_retail,
            mean_confidence_smart=mean_confidence_smart,
            var_stance_retail=var_stance_retail,
            var_stance_overall=var_stance_overall,
            pff=pff,
            smd=smd,
            momentum=momentum,
            mean_trust=0.0,
            trust_density=0.0,
        )

    def export_results(
        self,
        output_dir: str | Path,
        include_timeseries: bool = True,
    ) -> dict[str, Path]:
        """Export metrics to Parquet files.

        Args:
            output_dir: Directory for output files
            include_timeseries: Whether to compute and export time-series factors

        Returns:
            Dictionary of output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result_files = {}

        # Export tick-level cross-sectional metrics
        if self._tick_records:
            df_ticks = pd.DataFrame(self._tick_records)

            # Compute time-series factors if requested
            if include_timeseries and len(df_ticks) >= 2:
                df_ticks = self._compute_time_series(df_ticks)

            # Write to Parquet with compression
            tick_file = output_path / "tick_metrics.parquet"
            pq.write_table(
                pa.Table.from_pandas(df_ticks),
                tick_file,
                compression="snappy",
            )
            result_files["tick_metrics"] = tick_file

        return result_files

    def _compute_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute time-series factors using vectorized Pandas operations.

        Args:
            df: DataFrame with tick-level metrics

        Returns:
            DataFrame with additional time-series columns
        """
        # Sort by tick to ensure correct order
        df = df.sort_values("tick").reset_index(drop=True)

        # Delta SMD (lagged differences)
        df["delta_smd_1"] = df["smd"].diff(1)
        df["delta_smd_5"] = df["smd"].diff(5)
        df["delta_smd_20"] = df["smd"].diff(20)

        # EWM momentum
        df["momentum_ewm_5"] = df["momentum"].ewm(span=5, min_periods=1).mean()
        df["momentum_ewm_20"] = df["momentum"].ewm(span=20, min_periods=1).mean()

        # Realized volatility (20-tick rolling std of stance changes)
        stance_changes = df["momentum"].diff().fillna(0)
        df["realized_vol_20"] = stance_changes.rolling(window=20, min_periods=5).std()

        # IDF: Information Diffusion Half-life Factor
        # Fit exponential decay to absolute stance changes
        df["idf"] = self._compute_idf(stance_changes)

        return df

    def _compute_idf(self, stance_changes: pd.Series) -> pd.Series:
        """Compute Information Diffusion Half-life Factor.

        Uses exponential decay model: |ΔS(t)| ~ exp(-λt)
        IDF = log(2) / λ

        Args:
            stance_changes: Series of stance changes per tick

        Returns:
            Series of IDF values (half-life in ticks)
        """
        # Use absolute changes as proxy for information impact
        abs_changes = stance_changes.abs()

        # EWM to estimate decay rate (inverse of half-life)
        # Lower decay rate = longer half-life = slower diffusion
        ewm_mean = abs_changes.ewm(span=self.idf_window, min_periods=10).mean()

        # Estimate half-life from decay rate
        # Heuristic: use log ratio of current to past EWM
        past_ewm = ewm_mean.shift(self.idf_window // 2)

        # Safe division: replace zeros with NaN, ffill
        ratio = ewm_mean / (past_ewm.replace(0, np.nan).fillna(self.epsilon))

        # Compute half-life: log(2) / decay_rate
        # decay_rate approximated from ratio
        decay_rate = -np.log(ratio + self.epsilon) / (self.idf_window // 2)
        decay_rate = decay_rate.replace([np.inf, -np.inf], np.nan).fillna(0.1)

        # IDF = ln(2) / lambda
        idf = np.log(2) / (decay_rate + self.epsilon)

        # Clip to reasonable range [1, 100] ticks
        return idf.clip(1.0, 100.0)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics of collected metrics."""
        if not self._tick_records:
            return {"error": "No data collected"}

        df = pd.DataFrame(self._tick_records)

        return {
            "total_ticks": len(df),
            "mean_pff": float(df["pff"].mean()),
            "mean_smd": float(df["smd"].mean()),
            "smd_volatility": float(df["smd"].std()),
            "mean_momentum": float(df["momentum"].mean()),
            "final_retail_count": int(df["n_retail"].iloc[-1]),
            "final_smart_count": int(
                (df["n_quant"] + df["n_insider"] + df["n_regulator"]).iloc[-1]
            ),
        }
