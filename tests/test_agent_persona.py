"""Tests for Agent Persona system with Pydantic v2 validation.

Test scenarios:
- 70 Retail Investors + 15 Quant Institutions
- ValidationError on illegal parameters (e.g., neuroticism=1.2)
- Regulator capital_weight must be exactly 10000.0
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from utopia.layer3_cognition.agent_factory import AgentFactory
from utopia.layer3_cognition.agent_persona_models import (
    AgentRole,
    BaseAgentPersona,
    InsiderPersona,
    MacroRegulatorPersona,
    QuantInstitutionPersona,
    RetailInvestorPersona,
)


class TestPersonaValidation:
    """Test Pydantic boundary validation."""

    def test_valid_persona_creation(self):
        """Normal Persona creation should succeed."""
        persona = RetailInvestorPersona(agent_id="RETAIL_001", capital_weight=1.0)
        assert persona.agent_id == "RETAIL_001"
        assert persona.role == AgentRole.RETAIL
        assert persona.openness == 0.8  # Default value
        assert persona.neuroticism == 0.9
        assert persona.conscientiousness == 0.2
        assert persona.influence_weight == 0.1

    def test_invalid_neuroticism_too_high(self):
        """neuroticism > 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RetailInvestorPersona(
                agent_id="RETAIL_001", capital_weight=1.0, neuroticism=1.2  # Illegal
            )
        assert "neuroticism" in str(exc_info.value)

    def test_invalid_openness_negative(self):
        """openness < 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            QuantInstitutionPersona(
                agent_id="QUANT_001", capital_weight=100.0, openness=-0.1
            )

    def test_invalid_conscientiousness_above_one(self):
        """conscientiousness > 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            RetailInvestorPersona(
                agent_id="R001", capital_weight=1.0, conscientiousness=1.5
            )

    def test_invalid_influence_weight_negative(self):
        """influence_weight < 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            InsiderPersona(agent_id="I001", capital_weight=50.0, influence_weight=-0.5)

    def test_invalid_capital_weight_zero(self):
        """capital_weight <= 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BaseAgentPersona(
                agent_id="TEST_001", role=AgentRole.RETAIL, capital_weight=0
            )

    def test_invalid_capital_weight_negative(self):
        """capital_weight < 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BaseAgentPersona(
                agent_id="TEST_001", role=AgentRole.RETAIL, capital_weight=-100.0
            )


class TestRegulatorValidation:
    """Test national team special validation."""

    def test_regulator_valid_capital(self):
        """Regulator with capital_weight=10000.0 should pass."""
        regulator = MacroRegulatorPersona(agent_id="CSRC_001")
        assert regulator.capital_weight == 10000.0
        assert regulator.role == AgentRole.REGULATOR

    def test_regulator_invalid_capital_direct(self):
        """Regulator with capital_weight != 10000.0 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BaseAgentPersona(
                agent_id="CSRC_001",
                role=AgentRole.REGULATOR,
                capital_weight=5000.0,  # Wrong value
            )
        assert "must be exactly 10000.0" in str(exc_info.value)

    def test_factory_creates_regulator_with_correct_capital(self):
        """Factory method should auto-set Regulator capital_weight."""
        regulator = AgentFactory.create_agent("regulator", "CSRC_002")
        assert regulator.capital_weight == 10000.0


class TestAgentFactory:
    """Test factory methods."""

    def test_create_retail(self):
        """Create retail investor."""
        retail = AgentFactory.create_agent("retail", "RETAIL_001")
        assert isinstance(retail, RetailInvestorPersona)
        assert retail.openness == 0.8
        assert retail.neuroticism == 0.9

    def test_create_quant(self):
        """Create quant institution."""
        quant = AgentFactory.create_agent("quant", "QUANT_001")
        assert isinstance(quant, QuantInstitutionPersona)
        assert quant.neuroticism == 0.0  # Absolutely rational
        assert quant.conscientiousness == 1.0  # Perfect memory

    def test_create_insider(self):
        """Create insider."""
        insider = AgentFactory.create_agent("insider", "INSIDER_001")
        assert isinstance(insider, InsiderPersona)
        assert insider.influence_weight == 0.8  # High influence

    def test_create_regulator(self):
        """Create regulator."""
        regulator = AgentFactory.create_agent("regulator", "CSRC_001")
        assert isinstance(regulator, MacroRegulatorPersona)
        assert regulator.influence_weight == 1.0  # Maximum influence

    def test_create_with_override(self):
        """Allow overriding default parameters."""
        # Create a slightly calmer retail investor
        calm_retail = AgentFactory.create_agent(
            "retail", "RETAIL_002", neuroticism=0.7  # Override default 0.9
        )
        assert calm_retail.neuroticism == 0.7
        # Other defaults remain
        assert calm_retail.openness == 0.8

    def test_create_with_capital_override(self):
        """Allow overriding capital_weight (except Regulator)."""
        big_retail = AgentFactory.create_agent(
            "retail", "RETAIL_BIG", capital_weight=10.0  # Bigger retail
        )
        assert big_retail.capital_weight == 10.0

    def test_create_with_invalid_override(self):
        """Invalid override should raise ValidationError."""
        with pytest.raises(ValidationError):
            AgentFactory.create_agent(
                "retail", "RETAIL_003", neuroticism=1.2  # Illegal value
            )

    def test_create_batch(self):
        """Batch create agents."""
        retailers = AgentFactory.create_batch("retail", 5, "RETAIL")
        assert len(retailers) == 5
        assert retailers[0].agent_id == "RETAIL_001"
        assert retailers[4].agent_id == "RETAIL_005"
        # All should be RetailInvestorPersona
        for r in retailers:
            assert isinstance(r, RetailInvestorPersona)

    def test_invalid_role(self):
        """Invalid role should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AgentFactory.create_agent("invalid_role", "TEST_001")
        assert "Invalid role" in str(exc_info.value)

    def test_get_default_params(self):
        """Get default parameters for a role."""
        defaults = AgentFactory.get_default_params("retail")
        assert defaults["openness"] == 0.8
        assert defaults["neuroticism"] == 0.9
        assert defaults["conscientiousness"] == 0.2
        assert defaults["influence_weight"] == 0.1

    def test_get_default_params_quant(self):
        """Get default parameters for quant."""
        defaults = AgentFactory.get_default_params("quant")
        assert defaults["openness"] == 0.1
        assert defaults["neuroticism"] == 0.0
        assert defaults["conscientiousness"] == 1.0
        assert defaults["influence_weight"] == 0.5


class TestPersonaBehavior:
    """Test Persona behavior calculations."""

    def test_memory_decay_rate_retail(self):
        """Test memory decay rate calculation for retail."""
        # Retail: conscientiousness = 0.2, decay = 0.1 * (1 - 0.2) = 0.08
        retail = RetailInvestorPersona(agent_id="R001", capital_weight=1.0)
        assert retail.get_memory_decay_rate() == pytest.approx(0.08)

    def test_memory_decay_rate_quant(self):
        """Test memory decay rate calculation for quant (perfect memory)."""
        # Quant: conscientiousness = 1.0, decay = 0.1 * (1 - 1.0) = 0.0
        quant = QuantInstitutionPersona(agent_id="Q001", capital_weight=100.0)
        assert quant.get_memory_decay_rate() == pytest.approx(0.0)

    def test_bayesian_update_rate_retail(self):
        """Test Bayesian update rate for retail (high openness)."""
        retail = RetailInvestorPersona(agent_id="R001", capital_weight=1.0)
        assert retail.get_bayesian_update_rate() == 0.8  # High openness

    def test_bayesian_update_rate_quant(self):
        """Test Bayesian update rate for quant (low openness)."""
        quant = QuantInstitutionPersona(agent_id="Q001", capital_weight=100.0)
        assert quant.get_bayesian_update_rate() == 0.1  # Low openness

    def test_confidence_penalty_retail(self):
        """Test confidence penalty for retail (high neuroticism)."""
        retail = RetailInvestorPersona(agent_id="R001", capital_weight=1.0)
        # neuroticism = 0.9, dissonance = 0.5 -> penalty = 0.45
        assert retail.get_confidence_penalty(0.5) == pytest.approx(0.45)

    def test_confidence_penalty_quant(self):
        """Test confidence penalty for quant (zero neuroticism)."""
        quant = QuantInstitutionPersona(agent_id="Q001", capital_weight=100.0)
        # neuroticism = 0.0, any dissonance -> penalty = 0
        assert quant.get_confidence_penalty(0.5) == pytest.approx(0.0)

    def test_to_system_prompt(self):
        """Test system prompt generation."""
        retail = RetailInvestorPersona(agent_id="R001", capital_weight=1.0)
        prompt = retail.to_system_prompt()
        assert "R001" in prompt
        assert "retail" in prompt
        assert "Openness: 0.8" in prompt
        assert "Memory Decay Rate: 0.080" in prompt


class TestSeventyRetailersFifteenQuants:
    """
    Test scenario: Generate 70 retail investors + 15 quant institutions.
    """

    def test_create_70_retailers(self):
        """Generate 70 retail investors."""
        retailers = AgentFactory.create_batch("retail", 70, "RETAIL")
        assert len(retailers) == 70

        # Verify all are RetailInvestorPersona
        for r in retailers:
            assert isinstance(r, RetailInvestorPersona)
            assert r.role == AgentRole.RETAIL
            assert r.neuroticism == 0.9  # Default

    def test_create_15_quants(self):
        """Generate 15 quant institutions."""
        quants = AgentFactory.create_batch("quant", 15, "QUANT")
        assert len(quants) == 15

        # Verify all are QuantInstitutionPersona
        for q in quants:
            assert isinstance(q, QuantInstitutionPersona)
            assert q.role == AgentRole.QUANT
            assert q.neuroticism == 0.0  # Absolutely rational

    def test_mixed_population(self):
        """Create mixed population and verify totals."""
        retailers = AgentFactory.create_batch("retail", 70, "RETAIL")
        quants = AgentFactory.create_batch("quant", 15, "QUANT")
        insiders = AgentFactory.create_batch("insider", 10, "INSIDER")
        regulators = [AgentFactory.create_agent("regulator", "CSRC_001")]

        all_agents = retailers + quants + insiders + regulators
        assert len(all_agents) == 96

        # Verify total capital weight
        # (retail 70*1 + quant 15*100 + insider 10*50 + regulator 10000)
        total_capital = sum(a.capital_weight for a in all_agents)
        expected = 70 * 1 + 15 * 100 + 10 * 50 + 10000
        assert total_capital == expected

    def test_mixed_with_override(self):
        """Create mixed population with some overrides."""
        # 70 standard retailers
        retailers = AgentFactory.create_batch("retail", 70, "RETAIL")

        # 15 quants, but some with higher capital
        quants = AgentFactory.create_batch(
            "quant", 15, "QUANT", capital_weight=200.0
        )

        # Verify override worked
        for q in quants:
            assert q.capital_weight == 200.0
            assert q.neuroticism == 0.0  # Default still applies
