#!/usr/bin/env python3
"""Agent Persona System Demo - Utopia v0.2.0.

This example demonstrates the four standard Agent personas and their behavioral differences.

Features demonstrated:
- Creating different agent types using AgentFactory
- Comparing psychological parameters across roles
- Understanding how personas affect cognition layer
- Batch agent creation for population simulation
"""

from __future__ import annotations

from utopia.layer3_cognition.agent_factory import AgentFactory
from utopia.layer3_cognition.agent_persona_models import (
    AgentRole,
    RetailInvestorPersona,
    QuantInstitutionPersona,
    InsiderPersona,
    MacroRegulatorPersona,
)


def demo_single_agent_creation():
    """Demonstrate creating individual agents."""
    print("=" * 60)
    print("Demo 1: Single Agent Creation")
    print("=" * 60)

    # Create different types of agents
    retail = AgentFactory.create_agent("retail", "RETAIL_001")
    quant = AgentFactory.create_agent("quant", "QUANT_001")
    insider = AgentFactory.create_agent("insider", "INSIDER_001")
    regulator = AgentFactory.create_agent("regulator", "CSRC_001")

    agents = [
        ("Retail Investor", retail),
        ("Quant Institution", quant),
        ("Insider", insider),
        ("Macro Regulator", regulator),
    ]

    for name, agent in agents:
        print(f"\n【{name}】")
        print(f"  ID: {agent.agent_id}")
        print(f"  Role: {agent.role.value}")
        print(f"  Capital Weight: {agent.capital_weight}")
        print(f"  Psychological Profile:")
        print(f"    - Openness: {agent.openness:.1f} (Bayesian update rate)")
        print(f"    - Neuroticism: {agent.neuroticism:.1f} (Panic tendency)")
        print(f"    - Conscientiousness: {agent.conscientiousness:.1f} (Memory retention)")
        print(f"    - Influence: {agent.influence_weight:.1f} (Social impact)")
        print(f"  Behavioral Parameters:")
        print(f"    - Memory Decay Rate: {agent.get_memory_decay_rate():.3f}")
        print(f"    - Bayesian Update Rate: {agent.get_bayesian_update_rate():.1f}")


def demo_behavioral_differences():
    """Demonstrate how personas affect behavior calculations."""
    print("\n" + "=" * 60)
    print("Demo 2: Behavioral Differences")
    print("=" * 60)

    retail = RetailInvestorPersona(agent_id="R", capital_weight=1.0)
    quant = QuantInstitutionPersona(agent_id="Q", capital_weight=100.0)

    # Simulate cognitive dissonance scenario
    dissonance_level = 0.5

    print(f"\nScenario: Cognitive Dissonance (level={dissonance_level})")
    print(f"  Retail confidence penalty: {retail.get_confidence_penalty(dissonance_level):.2f}")
    print(f"  Quant confidence penalty: {quant.get_confidence_penalty(dissonance_level):.2f}")
    print(f"  → Retail suffers more from conflicting information (high neuroticism)")

    print(f"\nScenario: Information Absorption")
    print(f"  Retail Bayesian update rate: {retail.get_bayesian_update_rate():.1f}")
    print(f"  Quant Bayesian update rate: {quant.get_bayesian_update_rate():.1f}")
    print(f"  → Retail accepts new information faster (high openness)")

    print(f"\nScenario: Memory Retention")
    print(f"  Retail memory decay rate: {retail.get_memory_decay_rate():.3f}")
    print(f"  Quant memory decay rate: {quant.get_memory_decay_rate():.3f}")
    print(f"  → Quant has perfect memory (conscientiousness=1.0)")


def demo_batch_creation():
    """Demonstrate batch agent creation for population simulation."""
    print("\n" + "=" * 60)
    print("Demo 3: Batch Agent Creation")
    print("=" * 60)

    # Create a market population: 70 retail + 15 quant + 10 insider + 1 regulator
    retailers = AgentFactory.create_batch("retail", 70, "RETAIL")
    quants = AgentFactory.create_batch("quant", 15, "QUANT")
    insiders = AgentFactory.create_batch("insider", 10, "INSIDER")
    regulators = AgentFactory.create_batch("regulator", 1, "CSRC")
    population = retailers + quants + insiders + regulators

    print(f"\nCreated population of {len(population)} agents:")

    # Count by role
    role_counts = {}
    total_capital = 0.0
    for agent in population:
        role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
        total_capital += agent.capital_weight

    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")

    print(f"\nTotal capital weight: {total_capital:,.0f}")
    print("  → Regulator (CSRC) has 10000 weight, dominates capital")


def demo_parameter_override():
    """Demonstrate customizing agent parameters."""
    print("\n" + "=" * 60)
    print("Demo 4: Parameter Override")
    print("=" * 60)

    # Create custom agents with parameter overrides
    panic_retail = AgentFactory.create_agent(
        "retail",
        "PANIC_RETAIL",
        neuroticism=0.99,  # Extremely prone to panic
        openness=0.95,     # Very easily influenced
    )

    stubborn_quant = AgentFactory.create_agent(
        "quant",
        "STUBBORN_QUANT",
        openness=0.01,     # Almost no openness to new info
        conscientiousness=1.0,
    )

    print(f"\nCustom Retail (Panic Mode):")
    print(f"  Neuroticism: {panic_retail.neuroticism:.2f} (default: 0.9)")
    print(f"  Panic penalty at 0.5 dissonance: {panic_retail.get_confidence_penalty(0.5):.2f}")

    print(f"\nCustom Quant (Ultra-Stubborn):")
    print(f"  Openness: {stubborn_quant.openness:.2f} (default: 0.1)")
    print(f"  Information absorption rate: {stubborn_quant.get_bayesian_update_rate():.2f}")


def demo_system_prompt():
    """Demonstrate LLM system prompt generation."""
    print("\n" + "=" * 60)
    print("Demo 5: LLM System Prompt")
    print("=" * 60)

    retail = AgentFactory.create_agent("retail", "RETAIL_042")
    print("\n" + retail.to_system_prompt())

    print("\n" + "-" * 40)
    quant = AgentFactory.create_agent("quant", "QUANT_007")
    print("\n" + quant.to_system_prompt())


def demo_validation():
    """Demonstrate Pydantic validation."""
    print("\n" + "=" * 60)
    print("Demo 6: Pydantic Validation")
    print("=" * 60)

    # Valid creation
    try:
        valid = AgentFactory.create_agent("retail", "VALID", openness=0.5)
        print(f"✓ Valid agent created: openness={valid.openness}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # Invalid: neuroticism > 1.0
    try:
        AgentFactory.create_agent("retail", "INVALID", neuroticism=1.5)
    except Exception as e:
        print(f"✓ Caught invalid neuroticism: {e}")

    # Invalid: regulator with wrong capital
    try:
        from pydantic import ValidationError
        MacroRegulatorPersona(agent_id="BAD", capital_weight=100.0)
    except ValidationError as e:
        print(f"✓ Caught invalid regulator capital: {e.errors()[0]['msg']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Utopia Agent Persona System Demo v0.2.0")
    print("=" * 60)

    demo_single_agent_creation()
    demo_behavioral_differences()
    demo_batch_creation()
    demo_parameter_override()
    demo_system_prompt()
    demo_validation()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run main_simulation.py for full simulation")
    print("  - See simple_simulation.py for basic usage")
    print("  - Check tests/ for more examples")


if __name__ == "__main__":
    main()
