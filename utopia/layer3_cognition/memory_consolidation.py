"""Memory consolidation with Ebbinghaus decay and semantic clustering.

This module implements memory compression based on:
1. Ebbinghaus forgetting curve: V(t) = Importance * exp(-lambda * delta_tick)
2. Semantic vector similarity clustering (threshold 0.85)
3. LLM-based summarization of clustered memories
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from utopia.core.pydantic_models import (
    BigFiveTraits,
    ConsolidatedExperience,
    MemoryEntry,
    MemoryVector,
)

if TYPE_CHECKING:
    pass


@dataclass
class MemoryCluster:
    """A cluster of similar memories for consolidation.

    Attributes:
        memory_ids: IDs of memories in this cluster
        centroid: Vector centroid of the cluster
        average_similarity: Average pairwise similarity
        topic_ids: Related topic IDs
    """

    memory_ids: list[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    average_similarity: float = 0.0
    topic_ids: set[str] = field(default_factory=set)

    def add_memory(self, memory_id: str, vector: np.ndarray) -> None:
        """Add a memory to the cluster.

        Args:
            memory_id: Memory identifier
            vector: Memory embedding vector
        """
        self.memory_ids.append(memory_id)
        if self.centroid is None:
            self.centroid = vector.copy()
        else:
            # Update centroid incrementally
            n = len(self.memory_ids)
            self.centroid = ((n - 1) * self.centroid + vector) / n


class MemoryConsolidationSystem:
    """Advanced memory system with Ebbinghaus decay and consolidation.

    Implements:
    - Short-term buffer with FIFO eviction
    - Long-term storage for consolidated experiences
    - Ebbinghaus forgetting curve per memory
    - Semantic clustering for consolidation (similarity > 0.85)
    - LLM-based summarization
    """

    # Consolidation threshold for semantic similarity
    SIMILARITY_THRESHOLD: float = 0.85

    # Short-term buffer capacity before triggering consolidation
    SHORT_TERM_CAPACITY: int = 20

    # Memory strength threshold for forgetting
    FORGET_THRESHOLD: float = 0.1

    def __init__(
        self,
        traits: Optional[BigFiveTraits] = None,
        short_term_capacity: int = SHORT_TERM_CAPACITY,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embedding_func: Optional[Callable[[str], list[float]]] = None,
        llm_summarize_func: Optional[Callable[[list[str]], str]] = None,
    ):
        """Initialize memory consolidation system.

        Args:
            traits: Big Five personality traits (affects decay rate)
            short_term_capacity: Max short-term memories before consolidation
            similarity_threshold: Cosine similarity threshold for clustering
            embedding_func: Function to compute text embeddings
            llm_summarize_func: Function to summarize memory clusters
        """
        self.traits = traits or BigFiveTraits()
        self._short_term_capacity = short_term_capacity
        self._similarity_threshold = similarity_threshold
        self._embedding_func = embedding_func
        self._llm_summarize_func = llm_summarize_func

        # Memory storage
        self.short_term: deque[MemoryEntry] = deque(maxlen=short_term_capacity)
        self.long_term: list[MemoryEntry] = []
        self.consolidated_experiences: list[ConsolidatedExperience] = []

        # Vector cache for short-term memories
        self._vector_cache: dict[str, MemoryVector] = {}

        # Statistics
        self._consolidation_count = 0
        self._forget_count = 0

    def add_memory(
        self,
        content: str,
        importance: float,
        current_tick: int,
        source_agent_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        emotional_valence: float = 0.0,
    ) -> MemoryEntry:
        """Add a new memory to short-term buffer.

        If short-term buffer is full, triggers consolidation.

        Args:
            content: Memory content
            importance: Initial importance [0, 1]
            current_tick: Current simulation tick
            source_agent_id: Source agent
            topic_id: Related topic
            emotional_valence: Emotional tone [-1, 1]

        Returns:
            Created memory entry
        """
        # Check for duplicates
        if self._is_duplicate(content):
            # Update existing memory timestamp instead
            return self._update_duplicate(content, current_tick)

        # Generate unique ID
        memory_id = self._generate_id(content, current_tick)

        entry = MemoryEntry(
            id=memory_id,
            content=content,
            importance=float(np.clip(importance, 0.0, 1.0)),
            creation_tick=current_tick,
            source_agent_id=source_agent_id,
            topic_id=topic_id,
            emotional_valence=float(np.clip(emotional_valence, -1.0, 1.0)),
        )

        # Check if buffer will overflow
        if len(self.short_term) >= self._short_term_capacity - 1:
            # Consolidate before adding
            self.consolidate(current_tick)

        # Add to short-term
        self.short_term.append(entry)

        # Compute and cache vector
        if self._embedding_func:
            vector = self._embedding_func(content)
            self._vector_cache[memory_id] = MemoryVector(
                memory_id=memory_id,
                vector=vector,
            )

        return entry

    def consolidate(self, current_tick: int) -> list[ConsolidatedExperience]:
        """Perform memory consolidation.

        1. Compute current strength for all short-term memories
        2. Cluster memories by semantic similarity (> 0.85)
        3. Call LLM to summarize each cluster
        4. Move summaries to long-term storage
        5. Delete original clustered memories

        Args:
            current_tick: Current simulation tick

        Returns:
            List of consolidated experiences
        """
        if not self.short_term:
            return []

        # Filter memories that haven't decayed below threshold
        valid_memories = self._get_valid_memories(current_tick)

        if len(valid_memories) < 2:
            return []

        # Cluster memories by semantic similarity
        clusters = self._cluster_memories(valid_memories)

        # Create consolidated experiences
        consolidated = []
        for cluster in clusters:
            if len(cluster.memory_ids) >= 2:
                experience = self._create_consolidated_experience(
                    cluster, current_tick
                )
                if experience:
                    consolidated.append(experience)
                    self.consolidated_experiences.append(experience)

                    # Remove original memories from short-term
                    self._remove_memories(cluster.memory_ids)

        self._consolidation_count += len(consolidated)
        return consolidated

    def retrieve(
        self,
        query: str,
        current_tick: int,
        limit: int = 5,
        topic_id: Optional[str] = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve relevant memories with strength-weighted scoring.

        Scoring formula:
        score = relevance * 0.4 + current_strength * 0.4 + importance * 0.2

        Args:
            query: Query text
            current_tick: Current tick for decay calculation
            limit: Max results
            topic_id: Optional topic filter

        Returns:
            List of (memory, score) tuples sorted by score
        """
        # Combine all memories
        all_memories = list(self.short_term) + self.long_term

        # Filter by topic if specified
        if topic_id:
            all_memories = [m for m in all_memories if m.topic_id == topic_id]

        # Score each memory
        scored = []
        for memory in all_memories:
            # Compute current strength with Ebbinghaus decay
            strength = memory.compute_strength(
                current_tick, self.traits.conscientiousness
            )

            # Skip forgotten memories
            if strength < self.FORGET_THRESHOLD:
                continue

            # Compute relevance (simple keyword overlap for MVP)
            relevance = self._compute_relevance(query, memory.content)

            # Combined score
            score = (
                relevance * 0.4
                + strength * 0.4
                + memory.importance * 0.2
            )

            scored.append((memory, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def forget_weak_memories(self, current_tick: int) -> int:
        """Remove memories that have decayed below threshold.

        Args:
            current_tick: Current simulation tick

        Returns:
            Number of memories forgotten
        """
        forgotten = 0

        # Check short-term
        to_remove_short = []
        for memory in self.short_term:
            if memory.is_forgotten(
                current_tick, self.traits.conscientiousness, self.FORGET_THRESHOLD
            ):
                to_remove_short.append(memory.id)

        for mid in to_remove_short:
            self._remove_from_short_term(mid)
            forgotten += 1

        # Check long-term
        to_remove_long = []
        for i, memory in enumerate(self.long_term):
            if memory.is_forgotten(
                current_tick, self.traits.conscientiousness, self.FORGET_THRESHOLD
            ):
                to_remove_long.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(to_remove_long):
            del self.long_term[i]
            forgotten += 1

        self._forget_count += forgotten
        return forgotten

    def _get_valid_memories(
        self, current_tick: int
    ) -> list[tuple[MemoryEntry, MemoryVector]]:
        """Get memories that haven't decayed below threshold.

        Args:
            current_tick: Current tick

        Returns:
            List of (memory, vector) tuples
        """
        valid = []
        for memory in self.short_term:
            strength = memory.compute_strength(
                current_tick, self.traits.conscientiousness
            )
            if strength >= self.FORGET_THRESHOLD:
                vector = self._vector_cache.get(memory.id)
                if vector:
                    valid.append((memory, vector))
        return valid

    def _cluster_memories(
        self, memories: list[tuple[MemoryEntry, MemoryVector]]
    ) -> list[MemoryCluster]:
        """Cluster memories by semantic similarity.

        Uses greedy clustering: assign each memory to the most similar
        existing cluster, or create new cluster if no match > threshold.

        Args:
            memories: List of (memory, vector) tuples

        Returns:
            List of memory clusters
        """
        clusters: list[MemoryCluster] = []

        for memory, vector in memories:
            best_cluster = None
            best_similarity = 0.0

            # Find most similar cluster
            for cluster in clusters:
                if cluster.centroid is not None:
                    similarity = self._cosine_similarity(vector.vector, cluster.centroid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster

            # Assign to cluster or create new
            if best_similarity >= self._similarity_threshold and best_cluster:
                best_cluster.add_memory(memory.id, np.array(vector.vector))
                if memory.topic_id:
                    best_cluster.topic_ids.add(memory.topic_id)
            else:
                new_cluster = MemoryCluster()
                new_cluster.add_memory(memory.id, np.array(vector.vector))
                if memory.topic_id:
                    new_cluster.topic_ids.add(memory.topic_id)
                clusters.append(new_cluster)

        return clusters

    def _create_consolidated_experience(
        self, cluster: MemoryCluster, current_tick: int
    ) -> Optional[ConsolidatedExperience]:
        """Create consolidated experience from a cluster.

        Args:
            cluster: Memory cluster
            current_tick: Current tick

        Returns:
            Consolidated experience or None if LLM not available
        """
        # Get memory contents
        contents = []
        total_importance = 0.0

        for memory in self.short_term:
            if memory.id in cluster.memory_ids:
                contents.append(memory.content)
                total_importance += memory.importance

        if not contents:
            return None

        # Generate summary using LLM if available
        if self._llm_summarize_func:
            summary = self._llm_summarize_func(contents)
        else:
            # Fallback: concatenate with ellipsis
            summary = f"[Consolidated] {contents[0][:100]}... (+{len(contents)-1} related)"

        avg_importance = total_importance / len(contents)

        return ConsolidatedExperience(
            id=f"EXP_{hashlib.md5(str(cluster.memory_ids).encode()).hexdigest()[:8]}",
            summary=summary,
            original_memory_ids=cluster.memory_ids,
            average_importance=float(np.clip(avg_importance, 0.0, 1.0)),
            creation_tick=current_tick,
            topic_ids=list(cluster.topic_ids),
        )

    def _remove_memories(self, memory_ids: list[str]) -> None:
        """Remove memories from short-term storage.

        Args:
            memory_ids: IDs to remove
        """
        id_set = set(memory_ids)

        # Rebuild short-term deque without removed memories
        new_short_term = deque(maxlen=self._short_term_capacity)
        for memory in self.short_term:
            if memory.id not in id_set:
                new_short_term.append(memory)

        self.short_term = new_short_term

        # Clean up vector cache
        for mid in memory_ids:
            self._vector_cache.pop(mid, None)

    def _remove_from_short_term(self, memory_id: str) -> None:
        """Remove a single memory from short-term.

        Args:
            memory_id: Memory to remove
        """
        self.short_term = deque(
            [m for m in self.short_term if m.id != memory_id],
            maxlen=self._short_term_capacity,
        )
        self._vector_cache.pop(memory_id, None)

    def _is_duplicate(self, content: str) -> bool:
        """Check if content is duplicate of recent memory.

        Args:
            content: Content to check

        Returns:
            True if duplicate exists
        """
        content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()
        for memory in list(self.short_term)[-5:]:
            existing_hash = hashlib.md5(
                memory.content.lower().strip().encode()
            ).hexdigest()
            if existing_hash == content_hash:
                return True
        return False

    def _update_duplicate(
        self, content: str, current_tick: int
    ) -> Optional[MemoryEntry]:
        """Update existing duplicate memory.

        Args:
            content: Content that is duplicate
            current_tick: Current tick

        Returns:
            Updated memory or None
        """
        content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()
        for memory in self.short_term:
            existing_hash = hashlib.md5(
                memory.content.lower().strip().encode()
            ).hexdigest()
            if existing_hash == content_hash:
                # Refresh the memory by updating importance
                memory.importance = min(1.0, memory.importance + 0.1)
                return memory
        return None

    def _generate_id(self, content: str, tick: int) -> str:
        """Generate unique memory ID.

        Args:
            content: Memory content
            tick: Creation tick

        Returns:
            Unique ID string
        """
        hash_input = f"{content}:{tick}:{np.random.random()}"
        return f"M{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    def _compute_relevance(self, query: str, content: str) -> float:
        """Compute simple keyword relevance.

        Args:
            query: Query string
            content: Memory content

        Returns:
            Relevance score in [0, 1]
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & content_words)
        return overlap / len(query_words)

    def _cosine_similarity(self, v1: list[float], v2: np.ndarray) -> float:
        """Compute cosine similarity between vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity in [-1, 1]
        """
        vec1 = np.array(v1)
        dot = np.dot(vec1, v2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(v2)
        return float(dot / norm) if norm > 0 else 0.0

    def get_statistics(self) -> dict:
        """Get memory system statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "consolidated_count": len(self.consolidated_experiences),
            "consolidation_events": self._consolidation_count,
            "forgotten_count": self._forget_count,
        }
