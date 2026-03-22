# Utopia

**High-Fidelity Multi-Agent Simulation Engine**

Utopia is an open-source multi-agent simulation system designed for modeling collective intelligence through realistic agent behaviors. It simulates how individuals perceive information, form beliefs, interact socially, and how group dynamics emerge.

## Key Features

- **6-Layer Architecture**: Modular design from seed material processing to result analysis
- **Cognitive Agents**: Memory systems, belief networks, and decision engines
- **Social Dynamics**: Information propagation with cognitive distortion, polarization detection
- **Extensible**: Easy to add new domains (finance, politics, etc.)
- **Predictive Engine**: Can be integrated as a prediction engine for quantitative trading

## Architecture

```
┌─────────────────────────────────────────┐
│  L6: Result Analysis                    │
│      Findings / Predictions / Reports    │
├─────────────────────────────────────────┤
│  L5: Simulation Engine                  │
│      Tick Loop / Scheduling / Events     │
├─────────────────────────────────────────┤
│  L4: Social Interaction                 │
│      Relationships / Propagation         │
├─────────────────────────────────────────┤
│  L3: Individual Cognition               │
│      Memory / Beliefs / Decisions        │
├─────────────────────────────────────────┤
│  L2: World Model                        │
│      Knowledge Graph / Rules            │
├─────────────────────────────────────────┤
│  L1: Seed Material Processing           │
│      Entity Extraction / Parsing        │
└─────────────────────────────────────────┘
```

## Installation

```bash
pip install utopia-sim
```

Or install from source:

```bash
git clone https://github.com/anthropics-agents/utopia.git
cd utopia
pip install -e .
```

## Quick Start

```python
from utopia.core.config import SimulationConfig
from utopia.core.models import Entity, EntityType, SeedMaterial
from utopia.layer5_engine.engine import SimulationEngine

# Create seed material
seed = SeedMaterial(
    raw_text="Your seed text here...",
)

# Add entities (optional - will auto-generate if not provided)
seed.entities = [
    Entity(id="E1", name="Agent 1", type=EntityType.PERSON),
]

# Configure simulation
config = SimulationConfig(
    agent_count=10,
    max_ticks=50,
    domain="general",
)

# Run simulation
engine = SimulationEngine(config)
engine.initialize(seed)
result = engine.run()

print(f"Completed in {result.final_tick} ticks")
```

## Domain-Specific Extensions

### Financial Markets

Configure for market simulation:

```python
config = SimulationConfig(
    agent_count=100,
    max_ticks=100,
    domain="financial",
)
```

### Political Analysis

Configure for political simulation:

```python
config = SimulationConfig(
    agent_count=50,
    max_ticks=50,
    domain="political",
)
```

## Design Principles

1. **真实性优先于规模** - Quality over quantity; 100 realistic agents > 10,000 simple ones
2. **认知层次分离** - Decoupled world model, individual cognition, and social interaction
3. **可验证性** - Every agent's reasoning is traceable

## Roadmap

- [ ] Phase 2: Neo4j graph database integration
- [ ] Phase 2: Vector store integration (Milvus/Qdrant) for memory
- [ ] Phase 3: React + D3.js visualization dashboard
- [ ] Phase 3: REST API for trading system integration
- [ ] Phase 4: Financial market-specific agents and rules

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see issues for planned features and bugs.
