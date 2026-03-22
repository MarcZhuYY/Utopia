"""L3: Individual Cognition Layer - Agent Brain.

This layer implements the cognitive system for each agent:
- Memory system (short-term + long-term + consolidation)
- Belief system (stance + Bayesian updating)
- Decision engine (CoT/ToT reasoning)
- Persona anchor (anti-collapse mechanism)
"""

from utopia.layer3_cognition.agent import Agent
from utopia.layer3_cognition.memory import MemorySystem, MemoryItem
from utopia.layer3_cognition.beliefs import BeliefSystem, Stance, BeliefDelta
from utopia.layer3_cognition.beliefs_v2 import BayesianBeliefSystem, BayesianBeliefDelta
from utopia.layer3_cognition.memory_consolidation import (
    MemoryConsolidationSystem,
    MemoryCluster,
)
from utopia.layer3_cognition.persona_anchor import (
    PersonaAnchorSystem,
    PersonaState,
    CoreConflict,
)
from utopia.layer3_cognition.decision_engine import AgentDecisionEngine, Decision, AttentionItem

__all__ = [
    "Agent",
    "MemorySystem",
    "MemoryItem",
    "BeliefSystem",
    "Stance",
    "BeliefDelta",
    "BayesianBeliefSystem",
    "BayesianBeliefDelta",
    "MemoryConsolidationSystem",
    "MemoryCluster",
    "PersonaAnchorSystem",
    "PersonaState",
    "CoreConflict",
    "AgentDecisionEngine",
    "Decision",
    "AttentionItem",
]
