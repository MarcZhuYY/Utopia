"""L5: Simulation Engine Layer.

This layer orchestrates the tick-by-tick simulation:
- SimulationEngine: Main loop and state management
- EventInjector: External event injection
- ConvergenceDetector: Simulation convergence checking
"""

from utopia.layer5_engine.engine import SimulationEngine, WorldState, SimulationResult
from utopia.layer5_engine.event_injector import ExternalEventInjector, ExternalEvent
from utopia.layer5_engine.convergence import ConvergenceDetector, ConvergenceResult

__all__ = [
    "SimulationEngine",
    "WorldState",
    "SimulationResult",
    "ExternalEventInjector",
    "ExternalEvent",
    "ConvergenceDetector",
    "ConvergenceResult",
]
