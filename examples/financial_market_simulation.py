#!/usr/bin/env python3
"""Financial Market Simulation Example - Utopia v0.2.0.

This example demonstrates how to use Utopia for simulating financial market sentiment
dynamics with different agent types: retail investors, quant institutions, insiders,
and macro regulators.

Features demonstrated:
- Market-specific entity extraction
- Multi-type agent population
- Stance evolution tracking
- Market sentiment analysis
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np

from utopia.core.models import (
    Entity,
    EntityType,
    Event,
    Intent,
    MaterialType,
    SeedMaterial,
    Stakeholder,
    StakeholderRole,
)
from utopia.core.pydantic_models import StanceState
from utopia.layer3_cognition.agent_factory import AgentFactory
from utopia.layer3_cognition.beliefs_v2 import BayesianBeliefSystem
from utopia.layer5_engine.world_state_buffer import AgentState, WorldState, WorldStateBuffer
from utopia.layer6_analysis.metrics_collector import SimulationMetricsCollector


class MarketSimulationEngine:
    """Simplified market simulation engine."""

    def __init__(
        self,
        num_retail: int = 70,
        num_quant: int = 15,
        num_insider: int = 10,
        num_regulator: int = 1,
        max_ticks: int = 10,
    ):
        self.num_retail = num_retail
        self.num_quant = num_quant
        self.num_insider = num_insider
        self.num_regulator = num_regulator
        self.max_ticks = max_ticks

        self.agents: dict = {}
        self.buffer = WorldStateBuffer()
        self.collector = SimulationMetricsCollector()
        self.tick_count = 0

    def initialize(self) -> None:
        """Initialize agents with market seed."""
        print("[INIT] Creating market agent population...")

        # Create population
        population_config = {
            "retail": self.num_retail,
            "quant": self.num_quant,
            "insider": self.num_insider,
            "regulator": self.num_regulator,
        }

        agents_list = AgentFactory.create_batch(population_config)
        for agent in agents_list:
            self.agents[agent.agent_id] = agent

        print(f"  Created {len(self.agents)} agents:")
        role_counts = {}
        for agent in self.agents.values():
            role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
        for role, count in sorted(role_counts.items()):
            print(f"    - {role}: {count}")

        # Initialize stances with market bias
        np.random.seed(42)
        for aid, persona in self.agents.items():
            # Different starting stances based on role
            if persona.role.value == "retail":
                stance_val = np.random.uniform(-0.3, 0.5)  # Slightly bullish
            elif persona.role.value == "quant":
                stance_val = np.random.uniform(-0.1, 0.1)  # Neutral
            elif persona.role.value == "insider":
                stance_val = 0.6  # Bullish (information advantage)
            else:  # regulator
                stance_val = 0.0  # Neutral/stabilizing

            agent_state = AgentState(
                agent_id=aid,
                stances={
                    "MARKET": StanceState(
                        topic_id="MARKET",
                        position=float(stance_val),
                        confidence=0.5 if persona.role.value != "insider" else 0.9,
                    ),
                    "TECH_SECTOR": StanceState(
                        topic_id="TECH_SECTOR",
                        position=float(stance_val * 0.8),
                        confidence=0.4,
                    ),
                },
            )
            self.buffer.write_state.agent_states[aid] = agent_state

        print(f"\n[INIT] Market initialized with {len(self.agents)} agents")

    async def run_tick(self, market_event: dict | None = None) -> dict:
        """Run one market tick."""
        tick_start = self.tick_count
        self.tick_count += 1

        # Simulate market event impact
        if market_event:
            print(f"\n[TICK {tick_start}] Event: {market_event.get('description', 'None')}")

        # Agent updates (parallel in real implementation)
        updates = {}
        for aid, agent in self.agents.items():
            current_state = self.buffer.read_state.agent_states.get(aid)
            if not current_state:
                continue

            # Get market stance
            market_stance = current_state.stances.get("MARKET")
            if not market_stance:
                continue

            # Apply belief update based on agent type
            belief_system = BayesianBeliefSystem(agent_openness=agent.openness)

            # Simulate information exposure
            if market_event:
                evidence_pos = market_event.get("impact", 0.0)
                evidence_conf = 0.7 if agent.role.value == "insider" else 0.5
            else:
                # Random noise for retail, stable for quant
                noise = np.random.normal(0, 0.1) if agent.role.value == "retail" else 0
                evidence_pos = noise
                evidence_conf = 0.3

            # Update stance
            new_stance = belief_system.update_stance(
                current=market_stance,
                evidence_position=float(evidence_pos),
                evidence_confidence=float(evidence_conf),
            )

            # Apply confidence penalty for retail during volatility
            if agent.role.value == "retail" and abs(evidence_pos) > 0.3:
                penalty = agent.get_confidence_penalty(dissonance_level=abs(evidence_pos))
                new_stance.confidence = max(0.1, new_stance.confidence - penalty)

            updates[aid] = new_stance

        # Commit updates
        for aid, new_stance in updates.items():
            if aid in self.buffer.write_state.agent_states:
                self.buffer.write_state.agent_states[aid].stances["MARKET"] = new_stance

        # Calculate market sentiment
        sentiments = []
        capital_weights = []
        for aid, agent in self.agents.items():
            state = self.buffer.read_state.agent_states.get(aid)
            if state and "MARKET" in state.stances:
                stance = state.stances["MARKET"]
                # Weight by confidence and capital
                weighted_sentiment = stance.position * stance.confidence
                sentiments.append(weighted_sentiment)
                capital_weights.append(agent.capital_weight)

        if sentiments:
            # Simple average
            avg_sentiment = np.mean(sentiments)
            # Capital-weighted average
            cap_weighted = np.average(sentiments, weights=capital_weights)
        else:
            avg_sentiment = 0.0
            cap_weighted = 0.0

        return {
            "tick": tick_start,
            "avg_sentiment": float(avg_sentiment),
            "cap_weighted_sentiment": float(cap_weighted),
            "active_agents": len(updates),
        }

    async def run(self, events: list[dict] | None = None) -> dict:
        """Run full simulation."""
        print("\n" + "=" * 60)
        print("Starting Market Simulation")
        print("=" * 60)

        self.initialize()

        results = []
        for tick in range(self.max_ticks):
            # Get event for this tick if any
            event = events[tick] if events and tick < len(events) else None

            tick_result = await self.run_tick(event)
            results.append(tick_result)

            print(f"  Sentiment: {tick_result['avg_sentiment']:+.3f} "
                  f"(cap-weighted: {tick_result['cap_weighted_sentiment']:+.3f})")

        # Final analysis
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)

        final_sentiments = [r["avg_sentiment"] for r in results]
        print(f"\nMarket Sentiment Evolution:")
        print(f"  Initial: {final_sentiments[0]:+.3f}")
        print(f"  Final: {final_sentiments[-1]:+.3f}")
        print(f"  Change: {final_sentiments[-1] - final_sentiments[0]:+.3f}")
        print(f"  Volatility: {np.std(final_sentiments):.3f}")

        # Role analysis
        print(f"\nFinal Stance by Role:")
        role_stances = {}
        for aid, agent in self.agents.items():
            state = self.buffer.read_state.agent_states.get(aid)
            if state and "MARKET" in state.stances:
                stance = state.stances["MARKET"]
                role = agent.role.value
                if role not in role_stances:
                    role_stances[role] = []
                role_stances[role].append(stance.position)

        for role, stances in sorted(role_stances.items()):
            avg = np.mean(stances)
            std = np.std(stances)
            print(f"  {role}: {avg:+.3f} (±{std:.3f})")

        return {
            "ticks": self.tick_count,
            "results": results,
            "final_role_stances": {k: float(np.mean(v)) for k, v in role_stances.items()},
        }


def create_market_events() -> list[dict]:
    """Create sample market events."""
    return [
        {"description": "Positive earnings report", "impact": 0.3},
        {"description": "Fed signals rate cut", "impact": 0.5},
        {"description": "Trade war concerns", "impact": -0.4},
        {"description": "Tech breakthrough announced", "impact": 0.2},
        {"description": "Regulatory crackdown", "impact": -0.3},
    ]


async def main():
    """Run market simulation."""
    # Create simulation with default population
    engine = MarketSimulationEngine(
        num_retail=50,
        num_quant=10,
        num_insider=5,
        num_regulator=1,
        max_ticks=10,
    )

    # Create market events
    events = create_market_events()

    # Run simulation
    results = await engine.run(events)

    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("""
1. Retail investors show high volatility in sentiment (high openness + neuroticism)
2. Quant institutions maintain stable positions (low openness, high conscientiousness)
3. Insiders have high confidence and early advantage
4. Regulators provide stabilizing influence through capital weight

To extend this simulation:
- Add L4 social layer for information propagation
- Implement actual price dynamics
- Connect to real market data feed
- Add LLM for natural language sentiment analysis
    """)


if __name__ == "__main__":
    asyncio.run(main())
