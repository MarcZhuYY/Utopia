"""Agent Persona models using Pydantic v2 with strict validation.

Implements four standard Agent templates:
- RetailInvestorPersona: 散户投资者 (high openness, high neuroticism)
- QuantInstitutionPersona: 量化机构 (data-driven, rational)
- InsiderPersona: 内幕知情者 (high influence, conservative)
- MacroRegulatorPersona: 宏观调控者 (national team, forced capital=10000)

All personas use Pydantic v2 Field validation with ge/le constraints.
"""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class AgentRole(str, Enum):
    """Agent role type enumeration."""

    RETAIL = "retail"  # Retail investor
    QUANT = "quant"  # Quant institution
    INSIDER = "insider"  # Insider with information advantage
    REGULATOR = "regulator"  # Macro regulator (national team)


class BaseAgentPersona(BaseModel):
    """
    Base Agent Persona class - Pydantic v2 strict validation.

    Four core psychological parameters directly affect cognition layer:
    - openness: Affects Bayesian belief update absorption rate (L3 beliefs_v2)
    - neuroticism: Affects confidence penalty during cognitive dissonance (L3 decision_engine)
    - conscientiousness: Affects L3 memory decay rate (memory_consolidation)
    - influence_weight: Affects L4 information propagation tensor (social_tensor)
    """

    model_config = {"frozen": False, "validate_assignment": True}

    agent_id: str = Field(..., description="Unique agent identifier")
    role: AgentRole = Field(..., description="Agent role type")

    # Capital weight must be > 0
    capital_weight: float = Field(
        ..., gt=0, description="Capital weight (affects market impact calculation)"
    )

    # Four core psychological parameters [0.0, 1.0]
    openness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Openness - Bayesian update absorption rate coefficient",
    )
    neuroticism: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Neuroticism - confidence penalty coefficient during cognitive dissonance",
    )
    conscientiousness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Conscientiousness - memory decay rate adjustment (high=slow decay)",
    )
    influence_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Social influence - L4 information propagation weight",
    )

    @model_validator(mode="after")
    def validate_regulator_capital(self) -> Self:
        """
        Composite validation: Regulator capital weight must be exactly 10000.0.
        """
        if self.role == AgentRole.REGULATOR and self.capital_weight != 10000.0:
            raise ValueError(
                f"Regulator capital_weight must be exactly 10000.0, "
                f"got {self.capital_weight}"
            )
        return self

    def get_memory_decay_rate(self) -> float:
        """
        Calculate memory decay rate (negatively correlated with conscientiousness).
        Formula: lambda = 0.1 * (1 - conscientiousness)
        """
        return 0.1 * (1.0 - self.conscientiousness)

    def get_bayesian_update_rate(self) -> float:
        """
        Get Bayesian belief update rate (positively correlated with openness).
        """
        return self.openness

    def get_confidence_penalty(self, dissonance_level: float) -> float:
        """
        Calculate confidence penalty during cognitive dissonance.
        Penalty = neuroticism * dissonance_level
        """
        return self.neuroticism * dissonance_level

    def to_system_prompt(self) -> str:
        """Generate system prompt for LLM persona anchoring."""
        lines = [
            f"【Agent Persona: {self.agent_id}】",
            f"Role: {self.role.value}",
            f"Capital Weight: {self.capital_weight}",
            "",
            "【Psychological Profile】",
            f"- Openness: {self.openness:.1f} ({'Innovative' if self.openness > 0.7 else 'Conservative' if self.openness < 0.4 else 'Moderate'})",
            f"- Neuroticism: {self.neuroticism:.1f} ({'Sensitive' if self.neuroticism > 0.7 else 'Stable' if self.neuroticism < 0.4 else 'Moderate'})",
            f"- Conscientiousness: {self.conscientiousness:.1f} ({'Disciplined' if self.conscientiousness > 0.7 else 'Spontaneous' if self.conscientiousness < 0.4 else 'Moderate'})",
            f"- Influence: {self.influence_weight:.1f} ({'High' if self.influence_weight > 0.7 else 'Low' if self.influence_weight < 0.4 else 'Moderate'})",
            "",
            "【Behavioral Parameters】",
            f"- Memory Decay Rate: {self.get_memory_decay_rate():.3f}",
            f"- Bayesian Update Rate: {self.get_bayesian_update_rate():.1f}",
        ]
        return "\n".join(lines)


class RetailInvestorPersona(BaseAgentPersona):
    """
    Retail Investor Persona.

    Characteristics: Easily panicked, short-term memory, low influence, high openness.
    """

    role: AgentRole = Field(default=AgentRole.RETAIL, frozen=True)

    # Retail default parameters
    openness: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Highly susceptible to influence"
    )
    neuroticism: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Easily panicked (panic selling)"
    )
    conscientiousness: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Short-term memory"
    )
    influence_weight: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Low social influence"
    )

    def __init__(self, agent_id: str, capital_weight: float = 1.0, **kwargs):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RETAIL,
            capital_weight=capital_weight,
            **kwargs,
        )


class QuantInstitutionPersona(BaseAgentPersona):
    """
    Quant Institution Persona - Digital twin for Alpha-A quant script testing.

    Characteristics: Data-driven, perfectly rational, perfect memory, medium influence.
    """

    role: AgentRole = Field(default=AgentRole.QUANT, frozen=True)

    # Quant institution default parameters
    openness: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Only trusts data"
    )
    neuroticism: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Absolutely rational"
    )
    conscientiousness: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Perfect memory"
    )
    influence_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Medium influence"
    )

    def __init__(self, agent_id: str, capital_weight: float = 100.0, **kwargs):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.QUANT,
            capital_weight=capital_weight,
            **kwargs,
        )


class InsiderPersona(BaseAgentPersona):
    """
    Insider Persona - Agent with information advantage.

    Characteristics: Information advantage, high influence, moderately conservative.
    """

    role: AgentRole = Field(default=AgentRole.INSIDER, frozen=True)

    # Insider default parameters
    openness: float = Field(default=0.3, ge=0.0, le=1.0)
    neuroticism: float = Field(default=0.2, ge=0.0, le=1.0)
    conscientiousness: float = Field(default=0.8, ge=0.0, le=1.0)
    influence_weight: float = Field(
        default=0.8, ge=0.0, le=1.0, description="High influence"
    )

    def __init__(self, agent_id: str, capital_weight: float = 50.0, **kwargs):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.INSIDER,
            capital_weight=capital_weight,
            **kwargs,
        )


class MacroRegulatorPersona(BaseAgentPersona):
    """
    Macro Regulator Persona - "National Team".

    Characteristics: Absolute authority, highest influence, market stabilization mission.
    Forced constraint: capital_weight must be exactly 10000.0
    """

    role: AgentRole = Field(default=AgentRole.REGULATOR, frozen=True)

    # National team default parameters
    openness: float = Field(default=0.1, ge=0.0, le=1.0)
    neuroticism: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Absolutely stable"
    )
    conscientiousness: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Policy memory"
    )
    influence_weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Maximum influence"
    )

    def __init__(self, agent_id: str, **kwargs):
        # Force capital_weight = 10000.0
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.REGULATOR,
            capital_weight=10000.0,
            **kwargs,
        )
