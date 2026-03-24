"""Simple simulation example - Utopia v0.2.0.

This example demonstrates the Agent Persona system.
Shows how different agent roles (retail, quant, regulator) are created.
"""

from utopia.layer3_cognition.agent_factory import AgentFactory


def run_simple_simulation():
    """Run a simple agent creation demo."""
    print("=" * 60)
    print("Utopia Simple Simulation Demo")
    print("=" * 60)

    # Create different types of agents
    print("\n1. Creating individual agents:")
    retail = AgentFactory.create_agent("retail", "RETAIL_001")
    quant = AgentFactory.create_agent("quant", "QUANT_001")
    insider = AgentFactory.create_agent("insider", "INSIDER_001")
    regulator = AgentFactory.create_agent("regulator", "CSRC_001")

    agents = [retail, quant, insider, regulator]
    for agent in agents:
        print(f"  - {agent.agent_id}: {agent.role.value} (capital: {agent.capital_weight})")

    # Batch creation
    print("\n2. Batch creating agents:")
    retailers = AgentFactory.create_batch("retail", 10, "RETAIL")
    quants = AgentFactory.create_batch("quant", 5, "QUANT")
    print(f"  - Created {len(retailers)} retail agents")
    print(f"  - Created {len(quants)} quant agents")

    # Display psychological profiles
    print("\n3. Psychological Profiles:")
    for agent in agents:
        print(f"\n  {agent.agent_id} ({agent.role.value}):")
        print(f"    Openness: {agent.openness:.1f}")
        print(f"    Neuroticism: {agent.neuroticism:.1f}")
        print(f"    Conscientiousness: {agent.conscientiousness:.1f}")
        print(f"    Influence: {agent.influence_weight:.1f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return agents


if __name__ == "__main__":
    run_simple_simulation()
