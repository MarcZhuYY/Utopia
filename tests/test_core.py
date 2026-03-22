"""Tests for core models."""

import pytest

from utopia.core.models import (
    Entity,
    EntityType,
    Event,
    SeedMaterial,
    MaterialType,
    Persona,
)
from utopia.core.config import SimulationConfig, WorldRules


class TestCoreModels:
    """Test core data models."""

    def test_entity_creation(self):
        """Test entity creation with defaults."""
        entity = Entity(name="Test Entity", type=EntityType.PERSON)
        assert entity.name == "Test Entity"
        assert entity.type == EntityType.PERSON
        assert entity.influence_score == 0.5
        assert entity.id.startswith("E")

    def test_entity_serialization(self):
        """Test entity to_dict."""
        entity = Entity(
            id="E1",
            name="Test",
            type=EntityType.ORG,
            influence_score=0.8,
            initial_stance={"topic1": 0.5},
        )
        data = entity.to_dict()
        assert data["id"] == "E1"
        assert data["name"] == "Test"
        assert data["type"] == "org"
        assert data["influence_score"] == 0.8

    def test_seed_material_creation(self):
        """Test seed material creation."""
        seed = SeedMaterial(
            raw_text="Test text",
            material_type=MaterialType.NEWS,
        )
        assert seed.raw_text == "Test text"
        assert seed.material_type == MaterialType.NEWS

    def test_seed_material_from_dict(self):
        """Test seed material deserialization."""
        data = {
            "raw_text": "Test",
            "material_type": "financial",
            "entities": [],
            "events": [],
            "relationships": [],
        }
        seed = SeedMaterial.from_dict(data)
        assert seed.material_type == MaterialType.FINANCIAL


class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig()
        assert config.agent_count == 100
        assert config.max_ticks == 50
        assert config.domain == "general"

    def test_world_rules_defaults(self):
        """Test world rules defaults."""
        rules = WorldRules()
        assert rules.max_propagation_depth == 5
        assert rules.propagation_decay == 0.7
        assert rules.max_stance_change_per_tick == 0.15


class TestPersona:
    """Test Persona model."""

    def test_persona_creation(self):
        """Test persona creation."""
        persona = Persona(
            name="Test Agent",
            role="analyst",
            expertise=["finance", "tech"],
            influence_base=0.7,
        )
        assert persona.name == "Test Agent"
        assert persona.role == "analyst"
        assert persona.influence_base == 0.7
        assert "finance" in persona.expertise

    def test_persona_default_traits(self):
        """Test default BigFive traits."""
        persona = Persona()
        assert "openness" in persona.traits
        assert "conscientiousness" in persona.traits
        assert all(0 <= v <= 1 for v in persona.traits.values())
