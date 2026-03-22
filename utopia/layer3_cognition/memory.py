"""Memory system for agents.

Implements short-term (FIFO) and long-term memory with importance-based promotion.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from utopia.core.models import MemoryItem
from utopia.core.config import WorldRules


class MemorySystem:
    """Agent memory system with short-term and long-term storage.

    Memory flow:
    1. New information enters short-term memory
    2. When capacity exceeded, weighted eviction occurs
    3. High-importance or emotional items promote to long-term
    """

    MAX_SHORT_TERM = 20  # Default capacity
    LONG_TERM_PROMOTION_THRESHOLD = 0.7
    EMOTIONAL_PROMOTION_THRESHOLD = 0.8

    def __init__(
        self,
        short_term_capacity: int = MAX_SHORT_TERM,
        promotion_threshold: float = LONG_TERM_PROMOTION_THRESHOLD,
        emotional_threshold: float = EMOTIONAL_PROMOTION_THRESHOLD,
        half_life_hours: float = 48.0,
    ):
        """Initialize memory system.

        Args:
            short_term_capacity: Max short-term memory items
            promotion_threshold: Importance threshold for long-term promotion
            emotional_threshold: Emotional valence threshold for promotion
            half_life_hours: Memory half-life for time decay
        """
        self.short_term: deque[MemoryItem] = deque(maxlen=short_term_capacity)
        self.long_term: list[MemoryItem] = []
        self._short_term_capacity = short_term_capacity
        self._promotion_threshold = promotion_threshold
        self._emotional_threshold = emotional_threshold
        self._half_life_hours = half_life_hours

    def add(
        self,
        item: MemoryItem,
        agent_id: str,
        topic_id: Optional[str] = None,
    ) -> None:
        """Add item to memory.

        Args:
            item: Memory item to add
            agent_id: Agent adding this memory
            topic_id: Optional topic association
        """
        # Check for duplicates
        if self._is_duplicate(item, agent_id):
            return

        # Add to short-term
        self.short_term.append(item)

        # Check promotion criteria
        if self._should_promote(item):
            self._promote_to_long_term(item)

    def _is_duplicate(self, item: MemoryItem, agent_id: str) -> bool:
        """Check if item is duplicate of recent memory.

        Args:
            item: Item to check
            agent_id: Agent ID

        Returns:
            bool: True if duplicate
        """
        # Simple content-based dedup for recent items
        content_hash = hash(item.content.lower().strip())
        for existing in list(self.short_term)[-5:]:  # Check last 5
            if hash(existing.content.lower().strip()) == content_hash:
                return True
        return False

    def _should_promote(self, item: MemoryItem) -> bool:
        """Check if item should promote to long-term.

        Args:
            item: Item to check

        Returns:
            bool: True if should promote
        """
        return (
            item.importance >= self._promotion_threshold
            or abs(item.emotional_valence) >= self._emotional_threshold
        )

    def _promote_to_long_term(self, item: MemoryItem) -> None:
        """Promote item to long-term memory.

        Generates a summary to save storage.

        Args:
            item: Item to promote
        """
        # For MVP, just store as-is (can add summarization later)
        self.long_term.append(item)

    def retrieve(
        self,
        query: str,
        topic_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve relevant memories.

        For MVP, uses simple relevance scoring.
        Phase 2: Use vector embeddings for semantic search.

        Args:
            query: Query string
            topic_id: Filter by topic
            limit: Maximum results

        Returns:
            list[MemoryItem]: Retrieved memories
        """
        # Combine short and long term
        candidates = list(self.short_term) + self.long_term

        # Filter by topic if specified
        if topic_id:
            # For now, topic_id is stored in content or via association
            # Simple filter: check if topic appears in content
            candidates = [
                c for c in candidates
                if topic_id.lower() in c.content.lower()
            ]

        # Score by relevance, importance, and recency
        scored = []
        for item in candidates:
            # Simple relevance: keyword overlap
            query_words = set(query.lower().split())
            content_words = set(item.content.lower().split())
            relevance = len(query_words & content_words) / max(len(query_words), 1)

            # Recency decay
            recency = self._time_decay(item.timestamp)

            # Combined score
            score = relevance * 0.4 + item.importance * 0.4 + recency * 0.2
            scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scored[:limit]]

    def _time_decay(self, timestamp: datetime) -> float:
        """Calculate time decay factor.

        Uses exponential decay with half-life.

        Args:
            timestamp: Item timestamp

        Returns:
            float: Decay factor (0-1)
        """
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return math.exp(-age_hours / self._half_life_hours)

    def get_recent(self, limit: int = 10) -> list[MemoryItem]:
        """Get most recent memories.

        Args:
            limit: Maximum items

        Returns:
            list[MemoryItem]: Recent memories
        """
        combined = list(self.short_term) + sorted(
            self.long_term, key=lambda x: x.timestamp, reverse=True
        )
        return combined[:limit]

    def get_all(self) -> dict[str, list[dict]]:
        """Get all memories organized by type.

        Returns:
            dict: All memories
        """
        return {
            "short_term": [m.to_dict() for m in self.short_term],
            "long_term": [m.to_dict() for m in self.long_term],
        }

    def clear(self) -> None:
        """Clear all memories."""
        self.short_term.clear()
        self.long_term.clear()

    def __len__(self) -> int:
        """Get total memory count."""
        return len(self.short_term) + len(self.long_term)
