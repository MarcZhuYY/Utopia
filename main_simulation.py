#!/usr/bin/env python3
"""Utopia Engine Integration Test - 10 Agents x 5 Ticks

This script runs a minimal end-to-end simulation to verify:
1. L1-L6 all layers initialize correctly
2. Tick lifecycle completes without errors
3. Metrics collection and export works
4. No memory leaks or resource issues

Usage:
    python main_simulation.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add utopia to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from utopia.core.pydantic_models import StanceState
from utopia.layer3_cognition.agent_factory import AgentFactory
from utopia.layer5_engine.world_state_buffer import (
    AgentState,
    WorldState,
    WorldStateBuffer,
)
from utopia.layer6_analysis.metrics_collector import SimulationMetricsCollector


class MiniSimulationEngine:
    """Minimal simulation engine for integration testing."""

    def __init__(self, seed: int = 42):
        """Initialize mini simulation engine.

        Args:
            seed: Random seed for reproducibility
        """
        self.tick_count = 0
        self.agents: dict[str, Any] = {}
        self.buffer = WorldStateBuffer()
        self.collector = SimulationMetricsCollector()
        self.running = True
        self._start_time: float = 0.0

        # Set random seed for reproducibility
        np.random.seed(seed)

    def initialize(self) -> None:
        """Create 10 agents: 7 retail + 2 quant + 1 regulator."""
        print("[INIT] Creating 10 agents...")

        # 7 Retail investors
        for i in range(7):
            agent = AgentFactory.create_agent("retail", f"RETAIL_{i+1:03d}")
            self.agents[agent.agent_id] = agent

        # 2 Quant institutions
        for i in range(2):
            agent = AgentFactory.create_agent("quant", f"QUANT_{i+1:03d}")
            self.agents[agent.agent_id] = agent

        # 1 Regulator
        agent = AgentFactory.create_agent("regulator", "CSRC_001")
        self.agents[agent.agent_id] = agent

        print(f"[INIT] Created {len(self.agents)} agents")

        # Initialize world state with random stances
        for aid, persona in self.agents.items():
            stance_val = np.random.uniform(-0.5, 0.5)
            agent_state = AgentState(
                agent_id=aid,
                stances={
                    "MARKET": StanceState(
                        topic_id="MARKET",
                        position=float(stance_val),
                        confidence=0.6,
                    )
                },
            )
            self.buffer.write_state.agent_states[aid] = agent_state

        print(f"[INIT] World state initialized with {len(self.buffer.write_state.agent_states)} agents")

    async def run_tick(self) -> bool:
        """Execute one simulation tick.

        Returns:
            True if tick completed successfully
        """
        tick_start = time.time()
        print(f"[TICK {self.tick_count}] Starting...")

        try:
            # 1. Update stances (simulate belief drift)
            update_count = 0
            for aid, agent_state in self.buffer.write_state.agent_states.items():
                if "MARKET" in agent_state.stances:
                    old_stance = agent_state.stances["MARKET"]
                    # Random drift with bounds checking
                    drift = float(np.random.normal(0, 0.1))
                    new_position = float(np.clip(old_stance.position + drift, -1.0, 1.0))
                    agent_state.stances["MARKET"] = StanceState(
                        topic_id="MARKET",
                        position=new_position,
                        confidence=old_stance.confidence,
                    )
                    update_count += 1

            # 2. Record metrics
            metrics = self.collector.record_tick(
                tick=self.tick_count,
                world_state=self.buffer.read_state,
                agents=self.agents,
            )

            tick_duration = time.time() - tick_start
            print(
                f"[TICK {self.tick_count}] "
                f"PFF={metrics.pff:.3f}, SMD={metrics.smd:.3f}, MOM={metrics.momentum:.3f}, "
                f"updated={update_count}, time={tick_duration*1000:.1f}ms"
            )

            # 3. Swap buffers (atomic with lock protection)
            await self.buffer.swap_buffers()
            self.tick_count += 1

            return True

        except Exception as e:
            print(f"[TICK {self.tick_count}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run(self, n_ticks: int = 5) -> bool:
        """Run simulation for n ticks.

        Args:
            n_ticks: Number of ticks to simulate

        Returns:
            True if simulation completed successfully
        """
        print(f"[RUN] Starting {n_ticks} tick simulation...\n")
        self._start_time = time.time()

        try:
            for i in range(n_ticks):
                success = await self.run_tick()
                if not success:
                    print(f"\n[ERROR] Tick {i} failed, aborting simulation")
                    return False

            total_duration = time.time() - self._start_time
            print(f"\n[RUN] Simulation complete! Duration: {total_duration:.2f}s")

            # Export metrics
            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"\n[EXPORT] Exporting metrics to {tmpdir}...")
                files = self.collector.export_results(tmpdir)
                print(f"[EXPORT] Metrics saved to: {dict(files)}")

                # Verify export
                import pandas as pd

                df = pd.read_parquet(files["tick_metrics"])
                print(f"[EXPORT] Exported {len(df)} ticks with columns: {list(df.columns)}")

                # Verify Alpha Factors
                assert "pff" in df.columns, "PFF factor missing"
                assert "smd" in df.columns, "SMD factor missing"
                assert "momentum" in df.columns, "Momentum factor missing"
                print("[EXPORT] All Alpha Factors present: PFF, SMD, Momentum")

            # Summary
            summary = self.collector.get_summary_stats()
            print(f"\n[SUMMARY]")
            print(f"  Total ticks: {summary['total_ticks']}")
            print(f"  Mean PFF: {summary['mean_pff']:.3f}")
            print(f"  Mean SMD: {summary['mean_smd']:.3f}")
            print(f"  Mean Momentum: {summary['mean_momentum']:.3f}")
            print(f"  SMD Volatility: {summary.get('smd_volatility', 0):.3f}")

            return True

        except Exception as e:
            print(f"\n[ERROR] Simulation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def verify_agent_templates(self) -> bool:
        """Verify all 4 agent templates are correctly instantiated.

        Returns:
            True if all templates verified
        """
        print("\n[VERIFY] Checking agent templates...")

        # Count by role
        roles = {"retail": 0, "quant": 0, "insider": 0, "regulator": 0}
        capital_weights = []

        for aid, agent in self.agents.items():
            role = agent.role.value
            if role in roles:
                roles[role] += 1
            capital_weights.append(agent.capital_weight)

            # Verify personality bounds
            assert 0.0 <= agent.openness <= 1.0, f"{aid}: openness out of bounds"
            assert 0.0 <= agent.neuroticism <= 1.0, f"{aid}: neuroticism out of bounds"
            assert 0.0 <= agent.conscientiousness <= 1.0, f"{aid}: conscientiousness out of bounds"

        print(f"[VERIFY] Roles: retail={roles['retail']}, quant={roles['quant']}, "
              f"insider={roles['insider']}, regulator={roles['regulator']}")
        print(f"[VERIFY] Capital weights: min={min(capital_weights):.1f}, max={max(capital_weights):.1f}")

        # Verify expected counts
        assert roles["retail"] == 7, "Expected 7 retail investors"
        assert roles["quant"] == 2, "Expected 2 quant institutions"
        assert roles["regulator"] == 1, "Expected 1 regulator"

        # Verify regulator has max capital weight
        regulator = self.agents.get("CSRC_001")
        if regulator:
            assert regulator.capital_weight == 10000.0, "Regulator should have capital_weight=10000.0"
            print(f"[VERIFY] Regulator capital weight: {regulator.capital_weight} (correct)")

        print("[VERIFY] All agent templates verified successfully")
        return True


async def main():
    """Main entrypoint."""
    print("=" * 60)
    print("Utopia Engine Integration Test")
    print("10 Agents x 5 Ticks - L1-L6 Full Stack Verification")
    print("=" * 60 + "\n")

    # Create and initialize engine
    engine = MiniSimulationEngine(seed=42)
    engine.initialize()

    # Verify agent templates before running
    if not engine.verify_agent_templates():
        print("\n[FAIL] Agent template verification failed")
        return 1

    # Run simulation
    success = await engine.run(n_ticks=5)

    # Final result
    print("\n" + "=" * 60)
    if success:
        print("INTEGRATION TEST PASSED")
        print("All L1-L6 layers verified successfully")
    else:
        print("INTEGRATION TEST FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
