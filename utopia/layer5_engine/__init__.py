"""L5: Simulation Engine Layer.

This layer orchestrates the tick-by-tick simulation:
- SimulationEngine: Main loop and state management
- WorldStateBuffer: Double-buffered state for causal consistency
- LLMRouter: Capability-based model routing
- AsyncLLMScheduler: Rate-limited LLM API calls
- ActionBuffer: Deferred action execution
- EventInjector: External event injection
- ConvergenceDetector: Simulation convergence checking
"""

from utopia.layer5_engine.engine import SimulationEngine, WorldState, SimulationResult
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
from utopia.layer5_engine.async_llm_scheduler import (
    AsyncLLMScheduler,
    LLMResult,
    RateLimiter,
)
from utopia.layer5_engine.action_buffer import ActionBuffer, BufferedAction
from utopia.layer5_engine.event_injector import ExternalEventInjector, ExternalEvent
from utopia.layer5_engine.convergence import ConvergenceDetector, ConvergenceResult
from utopia.layer5_engine.mailbox import (
    Mailbox,
    MessageBroker,
    MessagePriority,
    ActiveTaskPool,
    TickProcessor,
)

__all__ = [
    "SimulationEngine",
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
    "AsyncLLMScheduler",
    "LLMResult",
    "RateLimiter",
    "ActionBuffer",
    "BufferedAction",
    "ExternalEventInjector",
    "ExternalEvent",
    "ConvergenceDetector",
    "ConvergenceResult",
    "Mailbox",
    "MessageBroker",
    "MessagePriority",
    "ActiveTaskPool",
    "TickProcessor",
]
