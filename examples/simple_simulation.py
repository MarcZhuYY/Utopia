"""Simple simulation example.

This example demonstrates running a minimal simulation without LLM integration.
"""

from utopia.core.config import SimulationConfig
from utopia.core.models import (
    MaterialType,
    SeedMaterial,
    Entity,
    EntityType,
)
from utopia.layer1_seed.parser import MaterialParser
from utopia.layer5_engine.engine import SimulationEngine
from utopia.layer6_analysis.report_generator import ReportGenerator


def run_simple_simulation():
    """Run a simple simulation."""
    # Create seed material directly
    seed = SeedMaterial(
        raw_text="TechCorp announces new AI product. Investor Alice is excited about the potential. "
                 "Analyst Bob remains skeptical due to competition. The market response is mixed.",
        material_type=MaterialType.NEWS,
    )

    # Add entities
    seed.entities = [
        Entity(
            id="E1",
            name="TechCorp",
            type=EntityType.ORG,
            attributes={"role": "company", "sector": "tech"},
            influence_score=0.8,
            initial_stance={"AI_adoption": 0.5},
        ),
        Entity(
            id="E2",
            name="Alice",
            type=EntityType.PERSON,
            attributes={"role": "investor"},
            influence_score=0.6,
            initial_stance={"AI_adoption": 0.8},
        ),
        Entity(
            id="E3",
            name="Bob",
            type=EntityType.PERSON,
            attributes={"role": "analyst"},
            influence_score=0.5,
            initial_stance={"AI_adoption": -0.3},
        ),
    ]

    # Configure simulation
    config = SimulationConfig(
        agent_count=3,
        max_ticks=5,
        domain="general",
        log_level="INFO",
        enable_distortion=False,  # Disable for deterministic results
        enable_polarization_detection=True,
    )

    # Create and run engine
    engine = SimulationEngine(config)
    engine.initialize(seed)

    print("Running simulation...")
    result = engine.run()

    print(f"\nSimulation complete!")
    print(f"Ticks: {result.final_tick}")
    print(f"Agents: {result.agent_count}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Converged: {result.converged}")
    if result.convergence_reason:
        print(f"Reason: {result.convergence_reason}")

    # Generate report
    report_gen = ReportGenerator(result)
    report = report_gen.generate()

    print("\n--- Key Findings ---")
    if report.key_findings:
        for f in report.key_findings:
            print(f"- [{f.type.value}] {f.title}")
    else:
        print("No significant findings.")

    print("\n--- Report ---")
    print(report.summary)

    # Save results
    output_path = report_gen.save_report("outputs", format="markdown")
    print(f"\nReport saved to: {output_path}")

    return result


if __name__ == "__main__":
    run_simple_simulation()
