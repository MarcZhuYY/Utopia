"""Seed material merger for combining multiple materials.

Handles deduplication and fusion when processing multiple seed materials.
"""

from __future__ import annotations

from typing import Optional

from utopia.core.models import Entity, Event, Relation, SeedMaterial


class SeedMerger:
    """Merger for combining multiple SeedMaterial instances.

    When merging:
    - Same entity name -> merge (unify ID)
    - Conflicting stances -> keep divergence, don't force unify
    - Timeline conflicts -> earlier events have priority
    """

    def __init__(self):
        """Initialize merger."""
        self._entity_map: dict[str, str] = {}  # name -> canonical_id

    def merge(self, materials: list[SeedMaterial]) -> SeedMaterial:
        """Merge multiple seed materials into one.

        Args:
            materials: List of SeedMaterial instances to merge

        Returns:
            SeedMaterial: Merged result
        """
        if not materials:
            return SeedMaterial()

        if len(materials) == 1:
            return materials[0]

        # Start with first material as base
        result = materials[0]
        self._entity_map.clear()

        # Build entity name map from first material
        for entity in result.entities:
            self._entity_map[entity.name.lower()] = entity.id

        # Merge remaining materials
        for material in materials[1:]:
            result = self._merge_two(result, material)

        return result

    def _merge_two(self, base: SeedMaterial, incoming: SeedMaterial) -> SeedMaterial:
        """Merge two seed materials.

        Args:
            base: Base material
            incoming: Material to merge in

        Returns:
            SeedMaterial: Merged result
        """
        # Merge entities
        for entity in incoming.entities:
            merged = self._merge_entity(entity)
            if merged not in base.entities:
                base.entities.append(merged)

        # Merge relationships (with remapped IDs)
        for relation in incoming.relationships:
            remapped = self._remap_relation(relation)
            if remapped not in base.relationships:
                base.relationships.append(remapped)

        # Merge events (keep all, dedupe by ID)
        existing_ids = {e.id for e in base.events}
        for event in incoming.events:
            if event.id not in existing_ids:
                base.events.append(event)
                existing_ids.add(event.id)

        # Merge stakeholders
        existing_stakeholder_ids = {s.entity_id for s in base.stakeholders}
        for stakeholder in incoming.stakeholders:
            if stakeholder.entity_id not in existing_stakeholder_ids:
                base.stakeholders.append(stakeholder)

        # Merge sentiment maps
        base.sentiment_map.update(incoming.sentiment_map)

        # Merge timeline
        base.timeline.extend(incoming.timeline)
        base.timeline.sort(key=lambda x: x.timestamp or "")

        # Take average credibility (can be weighted by text length)
        base.credibility = (base.credibility + incoming.credibility) / 2

        # Merge target audience (dedupe)
        base.target_audience = list(set(base.target_audience) | set(incoming.target_audience))

        return base

    def _merge_entity(self, entity: Entity) -> Entity:
        """Merge entity, unifying with existing if same name.

        Args:
            entity: Entity to merge

        Returns:
            Entity: Merged entity
        """
        name_key = entity.name.lower()

        if name_key in self._entity_map:
            # Entity exists - merge attributes (prefer higher influence)
            existing_id = self._entity_map[name_key]
            return Entity(
                id=existing_id,
                name=entity.name,
                type=entity.type,
                attributes=entity.attributes,
                influence_score=max(entity.influence_score, 0.5),
                initial_stance=entity.initial_stance,
            )

        # New entity
        self._entity_map[name_key] = entity.id
        return entity

    def _remap_relation(self, relation: Relation) -> Relation:
        """Remap entity IDs in relation to canonical IDs.

        Args:
            relation: Relation to remap

        Returns:
            Relation: Remapped relation
        """
        from_entity = self._get_canonical_id(relation.from_entity)
        to_entity = self._get_canonical_id(relation.to_entity)

        return Relation(
            from_entity=from_entity,
            to_entity=to_entity,
            type=relation.type,
            strength=relation.strength,
        )

    def _get_canonical_id(self, entity_ref: str) -> str:
        """Get canonical entity ID from any reference.

        If entity_ref is a name, return mapped ID.
        If entity_ref is already an ID, return as-is.

        Args:
            entity_ref: Entity name or ID

        Returns:
            str: Canonical entity ID
        """
        # Check if it's a name (lowercase match)
        name_key = entity_ref.lower()
        if name_key in self._entity_map:
            return self._entity_map[name_key]

        # Check if it's already an ID
        for canonical_name, canonical_id in self._entity_map.items():
            if canonical_id == entity_ref:
                return entity_ref

        # Unknown reference - return as-is but also map it
        self._entity_map[name_key] = entity_ref
        return entity_ref


def merge_seed_materials(materials: list[SeedMaterial]) -> SeedMaterial:
    """Convenience function to merge multiple seed materials.

    Args:
        materials: List of SeedMaterial instances

    Returns:
        SeedMaterial: Merged result
    """
    merger = SeedMerger()
    return merger.merge(materials)
