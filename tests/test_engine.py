"""Integration tests for Utopia simulation."""

import pytest

from utopia.core.config import SimulationConfig
from utopia.core.models import (
    Entity,
    EntityType,
    MaterialType,
    SeedMaterial,
)
from utopia.layer5_engine.engine import SimulationEngine, WorldState


class TestSimulationEngine:
    """Test simulation engine."""

    def test_engine_creation(self):
        """Test engine creation."""
        config = SimulationConfig(agent_count=5, max_ticks=3)
        engine = SimulationEngine(config)
        assert engine.config.agent_count == 5
        assert engine.tick_count == 0

    def test_engine_initialization(self):
        """Test engine initialization."""
        seed = SeedMaterial(
            raw_text="Test simulation with agents.",
            material_type=MaterialType.NEWS,
        )
        seed.entities = [
            Entity(id="E1", name="Alice", type=EntityType.PERSON),
            Entity(id="E2", name="Bob", type=EntityType.PERSON),
        ]

        config = SimulationConfig(agent_count=2, max_ticks=2)
        engine = SimulationEngine(config)
        engine.initialize(seed)

        assert engine.world_state is not None
        assert len(engine.world_state.agents) == 2

    def test_run_simulation(self):
        """Test running simulation."""
        seed = SeedMaterial(
            raw_text="Market event: Tech stocks rise on AI news.",
            material_type=MaterialType.FINANCIAL,
        )
        seed.entities = [
            Entity(id="E1", name="Investor A", type=EntityType.PERSON, influence_score=0.6),
            Entity(id="E2", name="Investor B", type=EntityType.PERSON, influence_score=0.5),
        ]

        config = SimulationConfig(
            agent_count=2,
            max_ticks=3,
            domain="financial",
            enable_distortion=False,
        )

        engine = SimulationEngine(config)
        engine.initialize(seed)
        result = engine.run()

        assert result.final_tick <= 3
        assert result.agent_count == 2
        assert result.duration_seconds >= 0

    def test_world_state_serialization(self):
        """Test world state to_dict."""
        state = WorldState()
        data = state.to_dict()

        assert data["tick"] == 0
        assert data["agent_count"] == 0
