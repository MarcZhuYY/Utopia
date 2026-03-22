"""Core data models for Utopia simulation system.

These models implement the data structures defined in the Multi-Agent Simulation
System design document (v1.0, 2026-03-22).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MaterialType(str, Enum):
    """Type of seed material."""

    NEWS = "news"
    POLICY = "policy"
    FINANCIAL = "financial"
    FICTION = "fiction"


class Intent(str, Enum):
    """Intent of the seed material."""

    PERSUADE = "persuade"
    INFORM = "inform"
    ENTERTAIN = "entertain"


class EntityType(str, Enum):
    """Type of entity."""

    PERSON = "person"
    ORG = "org"
    CONCEPT = "concept"
    LOCATION = "location"


class RelationType(str, Enum):
    """Type of relationship between entities."""

    WORKS_FOR = "works_for"
    OPPOSES = "opposes"
    ALLIES = "allies"
    COMPETES = "competes"
    BELONGS_TO = "belongs_to"
    PART_OF = "part_of"
    KNOWS = "knows"
    TRUSTS = "trusts"
    DISTRUSTS = "distrusts"


class StakeholderRole(str, Enum):
    """Role of a stakeholder in an event."""

    WINNER = "winner"
    LOSER = "loser"
    BYSTANDER = "bystander"


@dataclass
class Entity:
    """Represents an entity extracted from seed material.

    Attributes:
        id: Unique identifier (e.g., "E1")
        name: Human-readable name
        type: Entity type (person/org/concept/location)
        attributes: Additional attributes (role, sector, etc.)
        influence_score: Influence in current topic (0.0-1.0)
        initial_stance: Initial stance on topics {topic_id: stance (-1 to 1)}
    """

    id: str = field(default_factory=lambda: f"E{uuid.uuid4().hex[:8]}")
    name: str = ""
    type: EntityType = EntityType.PERSON
    attributes: dict[str, Any] = field(default_factory=dict)
    influence_score: float = 0.5
    initial_stance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "attributes": self.attributes,
            "influence_score": self.influence_score,
            "initial_stance": self.initial_stance,
        }


@dataclass
class Event:
    """Represents an event from seed material.

    Attributes:
        id: Unique identifier
        description: Human-readable description
        participants: List of entity IDs involved
        timestamp: ISO format timestamp
        causality: Description of cause and effect
        importance: Importance score (0.0-1.0)
    """

    id: str = field(default_factory=lambda: f"EV{uuid.uuid4().hex[:8]}")
    description: str = ""
    participants: list[str] = field(default_factory=list)
    timestamp: str = ""
    causality: str = ""
    importance: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "participants": self.participants,
            "timestamp": self.timestamp,
            "causality": self.causality,
            "importance": self.importance,
        }


@dataclass
class Relation:
    """Represents a relationship between entities.

    Attributes:
        from_entity: Source entity ID
        to_entity: Target entity ID
        type: Relationship type
        strength: Relationship strength (0.0-1.0)
    """

    from_entity: str = ""
    to_entity: str = ""
    type: RelationType = RelationType.KNOWS
    strength: float = 0.5

    def to_dict(self) -> dict:
        return {
            "from": self.from_entity,
            "to": self.to_entity,
            "type": self.type.value,
            "strength": self.strength,
        }


@dataclass
class Stakeholder:
    """Represents a stakeholder in an event.

    Attributes:
        entity_id: ID of the entity
        role: Role in this event (winner/loser/bystander)
        interest: Core interest description
        capacity: Impact degree (0.0-1.0)
        sentiment_toward: Sentiment toward topics
    """

    entity_id: str = ""
    role: StakeholderRole = StakeholderRole.BYSTANDER
    interest: str = ""
    capacity: float = 0.5
    sentiment_toward: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "role": self.role.value,
            "interest": self.interest,
            "capacity": self.capacity,
            "sentiment_toward": self.sentiment_toward,
        }


@dataclass
class TimelineNode:
    """Represents a node in the material timeline.

    Attributes:
        timestamp: ISO timestamp
        event_id: Associated event ID
        description: What happened
    """

    timestamp: str = ""
    event_id: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "description": self.description,
        }


@dataclass
class SeedMaterial:
    """Complete structured representation of seed input material.

    Attributes:
        raw_text: Original text
        material_type: Type (news/policy/financial/fiction)
        entities: Extracted entities
        events: Extracted events
        relationships: Extracted relationships
        sentiment_map: Sentiment per entity (entity_id -> sentiment -1 to 1)
        timeline: Chronological timeline
        stakeholders: List of stakeholders
        intent: Material intent (persuade/inform/entertain)
        credibility: Credibility score (0.0-1.0)
        target_audience: Target audience tags
    """

    raw_text: str = ""
    material_type: MaterialType = MaterialType.NEWS
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    relationships: list[Relation] = field(default_factory=list)
    sentiment_map: dict[str, float] = field(default_factory=dict)
    timeline: list[TimelineNode] = field(default_factory=list)
    stakeholders: list[Stakeholder] = field(default_factory=list)
    intent: Intent = Intent.INFORM
    credibility: float = 0.8
    target_audience: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "material_type": self.material_type.value,
            "entities": [e.to_dict() for e in self.entities],
            "events": [e.to_dict() for e in self.events],
            "relationships": [r.to_dict() for r in self.relationships],
            "sentiment_map": self.sentiment_map,
            "timeline": [t.to_dict() for t in self.timeline],
            "stakeholders": [s.to_dict() for s in self.stakeholders],
            "intent": self.intent.value,
            "credibility": self.credibility,
            "target_audience": self.target_audience,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SeedMaterial:
        """Create SeedMaterial from dictionary."""
        return cls(
            raw_text=data.get("raw_text", ""),
            material_type=MaterialType(data.get("material_type", "news")),
            entities=[Entity(**e) if isinstance(e, dict) else e for e in data.get("entities", [])],
            events=[Event(**e) if isinstance(e, dict) else e for e in data.get("events", [])],
            relationships=[
                Relation(**r) if isinstance(r, dict) else r for r in data.get("relationships", [])
            ],
            sentiment_map=data.get("sentiment_map", {}),
            timeline=[TimelineNode(**t) if isinstance(t, dict) else t for t in data.get("timeline", [])],
            stakeholders=[Stakeholder(**s) if isinstance(s, dict) else s for s in data.get("stakeholders", [])],
            intent=Intent(data.get("intent", "inform")),
            credibility=data.get("credibility", 0.8),
            target_audience=data.get("target_audience", []),
        )


# ============================================================================
# Agent-Related Models
# ============================================================================


@dataclass
class Persona:
    """Agent persona/human identity.

    Attributes:
        name: Agent name
        role: Role type (politician/journalist/investor/etc.)
        traits: BigFive personality traits
        expertise: List of expertise areas
        communication_style: formal/casual/aggressive/etc.
        goals: Long-term goals
        constraints: Role constraint description
        influence_base: Base influence score (0.0-1.0)
    """

    name: str = ""
    role: str = "analyst"
    traits: dict[str, float] = field(
        default_factory=lambda: {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
    )
    expertise: list[str] = field(default_factory=list)
    communication_style: str = "formal"
    goals: list[str] = field(default_factory=list)
    constraints: str = ""
    influence_base: float = 0.5

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "traits": self.traits,
            "expertise": self.expertise,
            "communication_style": self.communication_style,
            "goals": self.goals,
            "constraints": self.constraints,
            "influence_base": self.influence_base,
        }


@dataclass
class MemoryItem:
    """Single memory entry.

    Attributes:
        content: Memory content text
        timestamp: When this was experienced
        importance: Importance score (0.0-1.0)
        emotional_valence: Emotional tone (-1.0 to 1.0)
        source_agent: Who told me this
        verified: Whether this has been verified
    """

    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    emotional_valence: float = 0.0
    source_agent: str = ""
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "source_agent": self.source_agent,
            "verified": self.verified,
        }


@dataclass
class Stance:
    """Stance on a specific topic.

    Attributes:
        topic_id: Topic identifier
        position: Position (-1.0反对 to 1.0支持)
        confidence: Confidence in this stance (0.0-1.0)
        evidence: List of supporting evidence
        counter_arguments: List of counter-arguments
        last_updated: Last update timestamp
    """

    topic_id: str = ""
    position: float = 0.0
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    counter_arguments: list[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "position": self.position,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "counter_arguments": self.counter_arguments,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class AgentState:
    """Current state of an agent.

    Attributes:
        current_goal: Current goal description
        attention_focus: List of topics/events currently focused on
        emotional_state: Emotional dimensions (anger, fear, joy, sadness, surprise)
        energy: Energy/motivation level (0.0-1.0)
        resources: Domain-specific resources
    """

    current_goal: str = ""
    attention_focus: list[str] = field(default_factory=list)
    emotional_state: dict[str, float] = field(
        default_factory=lambda: {
            "anger": 0.0,
            "fear": 0.0,
            "joy": 0.0,
            "sadness": 0.0,
            "surprise": 0.0,
        }
    )
    energy: float = 1.0
    resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "current_goal": self.current_goal,
            "attention_focus": self.attention_focus,
            "emotional_state": self.emotional_state,
            "energy": self.energy,
            "resources": self.resources,
        }


@dataclass
class Action:
    """Represents an action taken by an agent.

    Attributes:
        action_type: Type of action (speak/privately_change_belief/act/silent)
        target_agent_id: Target agent ID (if applicable)
        content: Action content
        topic_id: Related topic ID
        timestamp: When action was taken
    """

    action_type: str = "speak"
    target_agent_id: str = ""
    content: str = ""
    topic_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target_agent_id": self.target_agent_id,
            "content": self.content,
            "topic_id": self.topic_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Message:
    """Represents a message between agents.

    Attributes:
        content: Message text
        sender_id: Sender agent ID
        receiver_id: Receiver agent ID
        topic_id: Related topic ID
        original_stance: Sender's stance when sending
        timestamp: When message was sent
    """

    content: str = ""
    sender_id: str = ""
    receiver_id: str = ""
    topic_id: str = ""
    original_stance: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "topic_id": self.topic_id,
            "original_stance": self.original_stance,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReceivedMessage:
    """Message as received by an agent (may have distortion applied).

    Attributes:
        message: Original message
        from_agent: Sender ID
        to_agent: Receiver ID
        depth: How many hops from original sender
        distortion_applied: Whether cognitive distortion was applied
        trust_at_reception: Trust level when received
    """

    message: Message
    from_agent: str = ""
    to_agent: str = ""
    depth: int = 0
    distortion_applied: bool = False
    trust_at_reception: float = 0.5

    def to_dict(self) -> dict:
        return {
            "message": self.message.to_dict(),
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "depth": self.depth,
            "distortion_applied": self.distortion_applied,
            "trust_at_reception": self.trust_at_reception,
        }


# ============================================================================
# World State Models
# ============================================================================


@dataclass
class Topic:
    """Represents a topic in the world model.

    Attributes:
        id: Topic ID
        name: Topic name
        description: Topic description
        sensitivity: Sensitivity level (0.0-1.0)
    """

    id: str = ""
    name: str = ""
    description: str = ""
    sensitivity: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sensitivity": self.sensitivity,
        }


@dataclass
class ExternalEvent:
    """External event injected into simulation.

    Attributes:
        id: Event ID
        description: Event description
        topic_id: Related topic
        importance: Importance (0.0-1.0)
        timestamp: When event occurs
        source: Source of event (external_system/news_api/etc.)
    """

    id: str = field(default_factory=lambda: f"EXT{uuid.uuid4().hex[:8]}")
    description: str = ""
    topic_id: str = ""
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "external"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "topic_id": self.topic_id,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }
