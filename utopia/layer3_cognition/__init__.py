"""L3: Individual Cognition Layer - Agent Brain.

This layer implements the cognitive system for each agent:
- Memory system (3-tier: Hot/Warm/Cold)
- Belief system (Bayesian updating)
- Decision engine (CoT/ToT reasoning)
- Persona templates (Retail/Quant/Insider/Regulator)
"""

from utopia.layer3_cognition.agent import Agent
from utopia.layer3_cognition.memory import MemorySystem, MemoryItem, MemorySystem3Tier
from utopia.layer3_cognition.beliefs import BayesianBeliefSystem, BayesianBeliefDelta
from utopia.layer3_cognition.warm_memory_models import (
    ColdMemory,
    HotMemoryItem,
    WarmMemoryItem,
    RetrievedMemory,
)
from utopia.layer3_cognition.agent_persona_models import (
    AgentRole,
    BaseAgentPersona,
    RetailInvestorPersona,
    QuantInstitutionPersona,
    InsiderPersona,
    MacroRegulatorPersona,
)
from utopia.layer3_cognition.agent_factory import AgentFactory

__all__ = [
    "Agent",
    # Memory system
    "MemorySystem",
    "MemorySystem3Tier",
    "MemoryItem",
    "ColdMemory",
    "HotMemoryItem",
    "WarmMemoryItem",
    "RetrievedMemory",
    # Belief system
    "BayesianBeliefSystem",
    "BayesianBeliefDelta",
    # Agent Persona templates
    "AgentRole",
    "BaseAgentPersona",
    "RetailInvestorPersona",
    "QuantInstitutionPersona",
    "InsiderPersona",
    "MacroRegulatorPersona",
    "AgentFactory",
]
