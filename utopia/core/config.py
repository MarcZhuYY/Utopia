"""Configuration dataclasses for Utopia simulation.

Implements SimulationConfig and WorldRules from the design document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# World Rules (Physical and Social Constraints)
# ============================================================================


@dataclass
class WorldRules:
    """World rules that govern simulation behavior.

    These are the fundamental constraints that all agents must follow.
    Divided into physical rules (hard constraints) and social rules (soft).
    """

    # Information propagation
    max_propagation_depth: int = 5
    propagation_decay: float = 0.7

    # Agent capabilities
    max_memory_per_tick: int = 10
    min_confidence_threshold: float = 0.3
    max_actions_per_tick: int = 3

    # Stance change
    max_stance_change_per_tick: float = 0.15
    stance_change_momentum: float = 0.3
    confidence_boost_rate: float = 0.1

    # Relationship change
    max_trust_change_per_tick: float = 0.1
    trust_recovery_rate: float = 0.02

    # Simulation control
    max_ticks: int = 50
    convergence_threshold: float = 0.1
    parallel_workers: int = 8

    # Cognitive distortion
    distortion_coefficient: float = 0.3

    # Memory
    memory_half_life_hours: float = 48.0
    short_term_capacity: int = 20
    long_term_promotion_threshold: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_propagation_depth": self.max_propagation_depth,
            "propagation_decay": self.propagation_decay,
            "max_memory_per_tick": self.max_memory_per_tick,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_actions_per_tick": self.max_actions_per_tick,
            "max_stance_change_per_tick": self.max_stance_change_per_tick,
            "stance_change_momentum": self.stance_change_momentum,
            "confidence_boost_rate": self.confidence_boost_rate,
            "max_trust_change_per_tick": self.max_trust_change_per_tick,
            "trust_recovery_rate": self.trust_recovery_rate,
            "max_ticks": self.max_ticks,
            "convergence_threshold": self.convergence_threshold,
            "parallel_workers": self.parallel_workers,
            "distortion_coefficient": self.distortion_coefficient,
            "memory_half_life_hours": self.memory_half_life_hours,
            "short_term_capacity": self.short_term_capacity,
            "long_term_promotion_threshold": self.long_term_promotion_threshold,
        }


# ============================================================================
# Role Constraints (Social Rules per Role)
# ============================================================================


ROLE_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "politician": {
        "can_speak_publicly": True,
        "can_negotiate": True,
        "typical_topics": ["policy", "economy", "social"],
        "constraint": "Must maintain party/constituency interests",
    },
    "journalist": {
        "can_speak_publicly": True,
        "can_investigate": True,
        "typical_topics": ["all"],
        "constraint": "Claims objective reporting but may have bias",
    },
    "celebrity": {
        "can_influence_public_opinion": True,
        "typical_topics": ["social", "lifestyle", "politics"],
        "constraint": "Speech constrained by public image",
    },
    "analyst": {
        "can_publish_analysis": True,
        "typical_topics": ["financial", "market"],
        "constraint": "Analysis should be data-supported but reasonable inference allowed",
    },
    "investor": {
        "can_trade": True,
        "typical_topics": ["financial", "market"],
        "constraint": "Must maximize returns within legal bounds",
    },
    "trader": {
        "can_trade": True,
        "typical_topics": ["financial", "market"],
        "constraint": "Subject to market rules and position limits",
    },
    "regulator": {
        "can_regulate": True,
        "typical_topics": ["policy", "financial"],
        "constraint": "Must maintain market fairness and stability",
    },
}


# ============================================================================
# Domain-Specific Rules
# ============================================================================


DOMAIN_RULES: dict[str, dict[str, str]] = {
    "financial": {
        "no_insider_trading": "Even with inside information, simulation should reflect this as violation",
        "market_impact": "Large trades affect prices, price changes feed back to market sentiment",
        "short-selling": "Short selling allowed but requires margin collateral",
    },
    "political": {
        "electoral_cycles": "Policy positions become more extreme near elections",
        "coalition_building": "Minorities need alliances to pass legislation",
        "public_opinion_matters": "Low support policies are harder to execute",
    },
    "general": {
        "free_speech": "Agents can express opinions unless in constrained roles",
        "causality": "Actions have consequences that propagate through network",
    },
}


# ============================================================================
# Simulation Configuration
# ============================================================================


@dataclass
class SimulationConfig:
    """Main configuration for simulation execution.

    This is the user-facing configuration that controls how the simulation runs.
    """

    # Agent configuration
    agent_count: int = 100
    agent_generation_mode: str = "from_seed"  # "from_seed" | "manual"

    # LLM configuration
    primary_model: str = "minimax-m2.7"
    fallback_model: str = "minimax-m2.5"
    small_model: str = "qwen3.5"

    # Execution configuration
    max_ticks: int = 50
    parallel_workers: int = 8
    tick_delay_ms: int = 0

    # Output configuration
    log_level: str = "INFO"  # DEBUG | INFO | WARN
    save_traces: bool = True
    save_graph: bool = True
    output_dir: str = "outputs"

    # Quality control
    enable_distortion: bool = True
    enable_polarization_detection: bool = True
    enable_opinion_leader_detection: bool = True

    # Domain configuration
    domain: str = "general"  # general | financial | political
    domain_rules: dict[str, str] = field(default_factory=dict)

    # World rules
    world_rules: WorldRules = field(default_factory=WorldRules)

    def __post_init__(self):
        """Apply domain-specific rules if not already set."""
        if not self.domain_rules and self.domain in DOMAIN_RULES:
            self.domain_rules = DOMAIN_RULES[self.domain].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_count": self.agent_count,
            "agent_generation_mode": self.agent_generation_mode,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "small_model": self.small_model,
            "max_ticks": self.max_ticks,
            "parallel_workers": self.parallel_workers,
            "tick_delay_ms": self.tick_delay_ms,
            "log_level": self.log_level,
            "save_traces": self.save_traces,
            "save_graph": self.save_graph,
            "output_dir": self.output_dir,
            "enable_distortion": self.enable_distortion,
            "enable_polarization_detection": self.enable_polarization_detection,
            "enable_opinion_leader_detection": self.enable_opinion_leader_detection,
            "domain": self.domain,
            "domain_rules": self.domain_rules,
            "world_rules": self.world_rules.to_dict(),
        }


# ============================================================================
# Default Configuration
# ============================================================================


DEFAULT_CONFIG = SimulationConfig()
