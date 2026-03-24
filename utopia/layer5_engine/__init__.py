"""L5: Simulation Engine Layer.

This layer orchestrates the tick-by-tick simulation:
- SimulationEngine: Main loop and state management (CQRS architecture)
- WorldStateBuffer: Double-buffered state for causal consistency
- LLMRouter: Capability-based model routing
- BatchEmbeddingProcessor: Lazy batch embedding processing
"""

from utopia.layer5_engine.engine import (
    SimulationEngine,
    SimulationEngineCQRS,
    SimulationConfig,
    TickResult,
    WorldState,
    SimulationResult,
)
from utopia.layer5_engine.world_state_buffer import (
    WorldStateBuffer,
    WorldState,
    AgentState,
    TickCoordinator,
)
from utopia.layer5_engine.llm_router import (
    LLMRouter,
    ExponentialBackoff,
    LONG_CONTEXT_THRESHOLD,
    MODEL_CONCURRENCY,
    MODEL_CAPABILITIES,
    FALLBACK_CHAIN,
)
from utopia.layer5_engine.batch_embedding_processor import (
    BatchEmbeddingProcessor,
    MockBatchEmbeddingProcessor,
)

__all__ = [
    "SimulationEngine",
    "SimulationEngineCQRS",
    "SimulationConfig",
    "TickResult",
    "WorldState",
    "SimulationResult",
    "WorldStateBuffer",
    "AgentState",
    "TickCoordinator",
    "LLMRouter",
    "ExponentialBackoff",
    "LONG_CONTEXT_THRESHOLD",
    "MODEL_CONCURRENCY",
    "MODEL_CAPABILITIES",
    "FALLBACK_CHAIN",
    "BatchEmbeddingProcessor",
    "MockBatchEmbeddingProcessor",
]
