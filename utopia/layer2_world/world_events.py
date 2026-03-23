"""Event sourcing models for CQRS architecture.

Defines WorldEvent domain models using Pydantic v2 with frozen=True
for immutability. All world state changes are recorded as immutable events.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event type enumeration."""

    NODE_PROPERTY_UPDATE = "node_property_update"
    RELATIONSHIP_CREATE = "relationship_create"
    RELATIONSHIP_UPDATE = "relationship_update"
    STANCE_CHANGE = "stance_change"
    OPINION_CREATE = "opinion_create"
    AGENT_ACTION = "agent_action"


class WorldEvent(BaseModel):
    """
    Event sourcing base class - immutable domain events.

    All world state changes must be recorded as WorldEvent,
    enabling complete change history tracking and auditing.

    Note:
        frozen=True ensures events are immutable after creation,
        which is essential for event sourcing integrity.
    """

    model_config = {"frozen": True}

    event_id: str = Field(..., description="Global unique event ID (UUID)")
    event_type: EventType = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tick_number: int = Field(..., ge=0, description="Tick number when event was generated")
    source_agent_id: str = Field(..., description="Agent ID that generated the event")

    # Causality tracking
    causation_event_id: Optional[str] = Field(
        None, description="Parent event ID that triggered this event"
    )
    correlation_id: Optional[str] = Field(
        None, description="Associated business process ID"
    )


class NodePropertyUpdateEvent(WorldEvent):
    """
    Node property update event.

    Used for updating properties of Agent/Topic/Event/Opinion nodes.
    """

    model_config = {"frozen": True}

    event_type: Literal[EventType.NODE_PROPERTY_UPDATE] = EventType.NODE_PROPERTY_UPDATE
    node_id: str = Field(..., description="Target node ID")
    node_type: str = Field(..., description="Node type (Agent/Topic/Event/Opinion)")
    property_name: str = Field(..., description="Property name")
    old_value: Optional[Any] = Field(None, description="Old value (for audit)")
    new_value: Any = Field(..., description="New value")
    update_reason: str = Field("", description="Update reason/context")


class RelationshipCreateEvent(WorldEvent):
    """
    Relationship/edge creation event.

    Used for creating new graph relationships.
    """

    model_config = {"frozen": True}

    event_type: Literal[EventType.RELATIONSHIP_CREATE] = EventType.RELATIONSHIP_CREATE
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(
        ..., description="Relationship type (SUPPORTS/OPPOSES/TRUSTS/...)"
    )
    weight: float = Field(default=1.0, ge=-1.0, le=1.0, description="Relationship weight")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class RelationshipUpdateEvent(WorldEvent):
    """Relationship weight update event."""

    model_config = {"frozen": True}

    event_type: Literal[EventType.RELATIONSHIP_UPDATE] = EventType.RELATIONSHIP_UPDATE
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Relationship type")
    old_weight: float = Field(..., description="Old weight")
    new_weight: float = Field(..., description="New weight")
    delta: float = Field(..., description="Weight change amount")


class StanceChangeEvent(WorldEvent):
    """
    Agent stance change event.

    Records changes in an Agent's stance on a specific Topic.
    """

    model_config = {"frozen": True}

    event_type: Literal[EventType.STANCE_CHANGE] = EventType.STANCE_CHANGE
    agent_id: str = Field(..., description="Agent ID")
    topic_id: str = Field(..., description="Topic ID")
    old_position: float = Field(
        ..., ge=-1.0, le=1.0, description="Original stance (-1 oppose ~ 1 support)"
    )
    new_position: float = Field(
        ..., ge=-1.0, le=1.0, description="New stance"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence level")
    trigger_message_id: Optional[str] = Field(
        None, description="Message ID that triggered the change"
    )

    @property
    def stance_delta(self) -> float:
        """Stance change magnitude."""
        return abs(self.new_position - self.old_position)


class OpinionCreateEvent(WorldEvent):
    """
    Opinion creation event.

    Agent expresses an opinion on a specific Topic.
    """

    model_config = {"frozen": True}

    event_type: Literal[EventType.OPINION_CREATE] = EventType.OPINION_CREATE
    opinion_id: str = Field(..., description="Opinion node ID")
    agent_id: str = Field(..., description="Agent ID")
    topic_id: str = Field(..., description="Topic ID")
    content: str = Field(..., max_length=2000, description="Opinion content")
    stance_position: float = Field(
        ..., ge=-1.0, le=1.0, description="Stance position"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence level")


class AgentActionEvent(WorldEvent):
    """
    Agent action event.

    Records Agent behaviors like speak/act/change_belief.
    """

    model_config = {"frozen": True}

    event_type: Literal[EventType.AGENT_ACTION] = EventType.AGENT_ACTION
    action_type: str = Field(
        ..., description="Action type (speak/private_message/act)"
    )
    target_agent_id: Optional[str] = Field(
        None, description="Target Agent ID (if applicable)"
    )
    content: str = Field(..., description="Action content")
    topic_id: Optional[str] = Field(None, description="Related Topic ID")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance")


# Union type for type hints
WorldEventUnion = (
    NodePropertyUpdateEvent
    | RelationshipCreateEvent
    | RelationshipUpdateEvent
    | StanceChangeEvent
    | OpinionCreateEvent
    | AgentActionEvent
)
