"""LLM-based extractor for entities, relations, and stakeholders.

This module uses LLM calls to extract structured information from raw text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from utopia.core.models import (
    Entity,
    EntityType,
    Event,
    Intent,
    Relation,
    RelationType,
    SeedMaterial,
    Stakeholder,
    StakeholderRole,
)


# ============================================================================
# Prompt Templates
# ============================================================================

EXTRACTION_PROMPT_TEMPLATE = """你是一个结构化信息抽取专家。从以下文本中抽取所有重要实体和它们之间的关系。

文本：
{raw_text}

请以JSON格式输出：
{{
  "entities": [
    {{
      "id": "E1",
      "name": "实体名称",
      "type": "person|org|concept|location",
      "attributes": {{"role": "角色", "sector": "领域"}},
      "influence_score": 0.0-1.0,
      "initial_stance": {{"议题1": 立场(-1到1), "议题2": 立场}}
    }}
  ],
  "relationships": [
    {{
      "from": "E1",
      "to": "E2",
      "type": "关系类型",
      "strength": 0.0-1.0
    }}
  ],
  "events": [
    {{
      "id": "EV1",
      "description": "事件描述",
      "participants": ["E1", "E2"],
      "timestamp": "ISO时间戳",
      "causality": "因果描述",
      "importance": 0.0-1.0
    }}
  ]
}}

注意：
- influence_score：该实体在当前话题中的影响力，普通人0.1-0.3，KOL 0.6-0.9
- initial_stance：基于文本推断该实体对各议题的倾向（-1强烈反对，0中立，1强烈支持）
- 关系类型可选：works_for, opposes, allies, competes, belongs_to, part_of, knows
- 如果文本中没有事件，events数组为空
"""


STAKEHOLDER_PROMPT_TEMPLATE = """分析文本中各方的利益关系和态度。

文本：
{raw_text}

输出JSON：
{{
  "stakeholders": [
    {{
      "entity_id": "E1",
      "role": "winner|loser|bystander",
      "interest": "该方在此事件中的核心利益描述",
      "capacity": 0.0-1.0,
      "sentiment_toward": {{"议题": 态度}}
    }}
  ],
  "intent": "inform|persuade|entertain",
  "credibility": 0.0-1.0,
  "target_audience": ["受众标签1", "受众标签2"]
}}
"""


TOPIC_EXTRACTION_PROMPT = """从以下事件中提取核心议题（Topic）。

事件：
{events}

输出JSON：
{{
  "topics": [
    {{
      "id": "T1",
      "name": "议题名称",
      "description": "描述",
      "sensitivity": 0.0-1.0
    }}
  ]
}}

要求：提取5-10个核心议题，每个议题用一句话描述，要有区分度。
"""


# ============================================================================
# LLM Interface
# ============================================================================


def llm_json(prompt: str, model: str = "minimax-m2.7") -> dict[str, Any]:
    """Call LLM and parse JSON response.

    This is a placeholder that should be replaced with actual LLM API integration.

    Args:
        prompt: Prompt to send
        model: Model to use

    Returns:
        dict: Parsed JSON response

    Raises:
        RuntimeError: If LLM call fails or returns invalid JSON
    """
    # TODO: Replace with actual LLM API call
    # For MVP, raise NotImplementedError to indicate this needs integration
    raise NotImplementedError(
        "LLM API integration not yet implemented. "
        "Please provide an LLM backend (MiniMax/aiohttp) to enable extraction."
    )


def llm_mini(prompt: str) -> str:
    """Call smaller/faster LLM for simple tasks.

    Args:
        prompt: Prompt to send

    Returns:
        str: Text response
    """
    # TODO: Replace with actual LLM API call
    raise NotImplementedError("LLM API integration not yet implemented.")


# ============================================================================
# Entity/Relation Extraction
# ============================================================================


def extract_entities_relations(
    text: str,
    model: str = "minimax-m2.7",
    use_cache: bool = True,
) -> tuple[list[Entity], list[Relation], list[Event]]:
    """Extract entities, relations, and events from text using LLM.

    Args:
        text: Raw text to analyze
        model: LLM model to use
        use_cache: Whether to use result caching

    Returns:
        tuple: (entities, relations, events)
    """
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(raw_text=text)

    try:
        result = llm_json(prompt, model)
    except NotImplementedError:
        # Return empty results if LLM not available
        return [], [], []

    entities = []
    for e in result.get("entities", []):
        try:
            entity = Entity(
                id=e.get("id", f"E{len(entities)}"),
                name=e.get("name", ""),
                type=EntityType(e.get("type", "person")),
                attributes=e.get("attributes", {}),
                influence_score=float(e.get("influence_score", 0.5)),
                initial_stance=e.get("initial_stance", {}),
            )
            entities.append(entity)
        except (ValueError, TypeError):
            continue

    relations = []
    for r in result.get("relationships", []):
        try:
            relation = Relation(
                from_entity=r.get("from", ""),
                to_entity=r.get("to", ""),
                type=RelationType(r.get("type", "knows")),
                strength=float(r.get("strength", 0.5)),
            )
            relations.append(relation)
        except (ValueError, TypeError):
            continue

    events = []
    for ev in result.get("events", []):
        try:
            event = Event(
                id=ev.get("id", f"EV{len(events)}"),
                description=ev.get("description", ""),
                participants=ev.get("participants", []),
                timestamp=ev.get("timestamp", ""),
                causality=ev.get("causality", ""),
                importance=float(ev.get("importance", 0.5)),
            )
            events.append(event)
        except (ValueError, TypeError):
            continue

    return entities, relations, events


# ============================================================================
# Stakeholder Analysis
# ============================================================================


def extract_stakeholders(
    text: str,
    model: str = "minimax-m2.7",
) -> tuple[list[Stakeholder], Intent, float, list[str]]:
    """Extract stakeholders, intent, and metadata from text.

    Args:
        text: Raw text to analyze
        model: LLM model to use

    Returns:
        tuple: (stakeholders, intent, credibility, target_audience)
    """
    prompt = STAKEHOLDER_PROMPT_TEMPLATE.format(raw_text=text)

    try:
        result = llm_json(prompt, model)
    except NotImplementedError:
        return [], Intent.INFORM, 0.8, []

    stakeholders = []
    for s in result.get("stakeholders", []):
        try:
            stakeholder = Stakeholder(
                entity_id=s.get("entity_id", ""),
                role=StakeholderRole(s.get("role", "bystander")),
                interest=s.get("interest", ""),
                capacity=float(s.get("capacity", 0.5)),
                sentiment_toward=s.get("sentiment_toward", {}),
            )
            stakeholders.append(stakeholder)
        except (ValueError, TypeError):
            continue

    try:
        intent = Intent(result.get("intent", "inform"))
    except ValueError:
        intent = Intent.INFORM

    credibility = float(result.get("credibility", 0.8))
    target_audience = result.get("target_audience", [])

    return stakeholders, intent, credibility, target_audience


# ============================================================================
# Topic Extraction
# ============================================================================


def extract_topics(
    events: list[Event],
    model: str = "minimax-m2.7",
) -> list[dict[str, Any]]:
    """Extract topics from events.

    Args:
        events: List of events
        model: LLM model to use

    Returns:
        list[dict]: List of topic dictionaries
    """
    if not events:
        return []

    events_text = "\n".join([f"- {e.description}" for e in events])
    prompt = TOPIC_EXTRACTION_PROMPT.format(events=events_text)

    try:
        result = llm_json(prompt, model)
    except NotImplementedError:
        return []

    return result.get("topics", [])


# ============================================================================
# LLM Extractor Class
# ============================================================================


class LLMExtractor:
    """Complete LLM-based extraction pipeline.

    This class orchestrates the full extraction process for a seed material.
    """

    def __init__(self, model: str = "minimax-m2.7", use_cache: bool = True):
        """Initialize extractor.

        Args:
            model: LLM model to use
            use_cache: Whether to cache results
        """
        self.model = model
        self.use_cache = use_cache
        self._cache: dict[str, dict[str, Any]] = {}

    def extract(self, seed: SeedMaterial) -> SeedMaterial:
        """Perform complete extraction on seed material.

        Args:
            seed: SeedMaterial with raw_text populated

        Returns:
            SeedMaterial: Fully populated with extracted data
        """
        if not seed.raw_text:
            raise ValueError("Seed material has no raw_text")

        # Check cache
        cache_key = seed.raw_text[:100]
        if self.use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            return self._apply_cached(seed, cached)

        # Extract entities, relations, events
        entities, relations, events = extract_entities_relations(
            seed.raw_text, self.model, self.use_cache
        )

        # Extract stakeholders, intent, etc.
        stakeholders, intent, credibility, audience = extract_stakeholders(
            seed.raw_text, self.model
        )

        # Build sentiment map from stakeholders
        sentiment_map = {}
        for s in stakeholders:
            for topic, sentiment in s.sentiment_toward.items():
                sentiment_map[f"{s.entity_id}_{topic}"] = sentiment

        # Update seed material
        seed.entities = entities
        seed.relationships = relations
        seed.events = events
        seed.stakeholders = stakeholders
        seed.intent = intent
        seed.credibility = credibility
        seed.target_audience = audience
        seed.sentiment_map = sentiment_map

        return seed

    def _apply_cached(self, seed: SeedMaterial, cached: dict[str, Any]) -> SeedMaterial:
        """Apply cached extraction results to seed."""
        seed.entities = [Entity(**e) for e in cached.get("entities", [])]
        seed.relationships = [Relation(**r) for r in cached.get("relationships", [])]
        seed.events = [Event(**e) for e in cached.get("events", [])]
        seed.stakeholders = [Stakeholder(**s) for s in cached.get("stakeholders", [])]
        seed.intent = Intent(cached.get("intent", "inform"))
        seed.credibility = float(cached.get("credibility", 0.8))
        seed.target_audience = cached.get("target_audience", [])
        seed.sentiment_map = cached.get("sentiment_map", {})
        return seed
