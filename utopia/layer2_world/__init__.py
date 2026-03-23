"""L2: World Model Layer - Knowledge Graph and Rule Engine.

This layer builds the knowledge graph and enforces world rules.

Phase 9: CQRS + Event Sourcing components for 100+ agent concurrency:
- WorldEvent: Immutable domain events
- WorldEventBuffer: Lock-free event collection
- Neo4jGraphMutator: UNWIND batch processing
- KnowledgeGraphQueryService: Read-only queries
"""

from utopia.layer2_world.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder
from utopia.layer2_world.rule_engine import RuleEngine, RuleValidator, ValidationResult

# Phase 9: CQRS + Event Sourcing
from utopia.layer2_world.world_events import (
    AgentActionEvent,
    EventType,
    NodePropertyUpdateEvent,
    OpinionCreateEvent,
    RelationshipCreateEvent,
    RelationshipUpdateEvent,
    StanceChangeEvent,
    WorldEvent,
    WorldEventUnion,
)
from utopia.layer2_world.world_event_buffer import WorldEventBuffer
from utopia.layer2_world.neo4j_graph_mutator import (
    Neo4jBatchError,
    Neo4jGraphMutator,
)
from utopia.layer2_world.query_service import (
    AgentStance,
    KnowledgeGraphQueryService,
    ReadOnlyContext,
    RecentEvent,
    TrustRelationship,
)

__all__ = [
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "RuleEngine",
    "RuleValidator",
    "ValidationResult",
    # Phase 9: Event Sourcing
    "WorldEvent",
    "WorldEventUnion",
    "EventType",
    "StanceChangeEvent",
    "RelationshipCreateEvent",
    "RelationshipUpdateEvent",
    "NodePropertyUpdateEvent",
    "OpinionCreateEvent",
    "AgentActionEvent",
    # Phase 9: CQRS Infrastructure
    "WorldEventBuffer",
    "Neo4jGraphMutator",
    "Neo4jBatchError",
    "KnowledgeGraphQueryService",
    "ReadOnlyContext",
    "AgentStance",
    "TrustRelationship",
    "RecentEvent",
]
