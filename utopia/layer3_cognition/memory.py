"""3-tier memory system implementation.

Hot/Warm/Cold memory architecture with Smart RAG retrieval.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from utopia.core.models import MemoryItem
from utopia.layer3_cognition.warm_memory_models import (
    ColdMemory,
    HotMemoryItem,
    PendingEmbeddingItem,
    RetrievedMemory,
    WarmMemoryItem,
)

if TYPE_CHECKING:
    pass


class MemorySystem3Tier:
    """3-tier 分层内存系统.

    Architecture:
    - Cold Memory: Persona, goals, high-confidence stances - always in system prompt
    - Hot Memory: Recent 1-3 ticks, no embedding, fast keyword search
    - Warm Memory: Consolidated experiences with vector embeddings

    Attributes:
        agent_id: Agent identifier
        cold: ColdMemory - static persona and beliefs
        hot: deque[HotMemoryItem] - recent experiences (no embedding)
        warm: list[WarmMemoryItem] - consolidated experiences (with vectors)
        pending_embeddings: list[PendingEmbeddingItem] - queue for batch processing
    """

    DEFAULT_HOT_MAXLEN = 10
    DEFAULT_WARM_MAX_SIZE = 1000
    DEFAULT_IMPORTANCE_THRESHOLD = 0.3
    DEFAULT_VECTOR_DIM = 768  # text-embedding-3-small

    def __init__(
        self,
        agent_id: str,
        cold_memory: ColdMemory,
        hot_memory_maxlen: int = DEFAULT_HOT_MAXLEN,
        warm_memory_max_size: int = DEFAULT_WARM_MAX_SIZE,
        importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD,
    ):
        """Initialize 3-tier memory system.

        Args:
            agent_id: Agent identifier
            cold_memory: Cold memory (persona, goals, etc.)
            hot_memory_maxlen: Max size of hot memory buffer
            warm_memory_max_size: Max size of warm memory
            importance_threshold: Min importance to enter warm memory queue
        """
        self.agent_id = agent_id
        self.cold = cold_memory

        # Hot Memory: collections.deque，无需 embedding
        self.hot: deque[HotMemoryItem] = deque(maxlen=hot_memory_maxlen)

        # Warm Memory: 带向量的 consolidation 经验
        self.warm: list[WarmMemoryItem] = []
        self.warm_max_size = warm_memory_max_size
        self._importance_threshold = importance_threshold

        # 待处理的 embedding 队列 (Lazy Batch Embedding 关键)
        self.pending_embeddings: list[PendingEmbeddingItem] = []

        # 向量索引 (numpy 暴力搜索，可替换为 FAISS)
        self._vector_index: Optional[np.ndarray] = None
        self._vector_index_dirty: bool = True

        # CRITICAL FIX: Threading locks for vector index safety
        self._vector_lock = threading.RLock()  # Protects vector index
        self._warm_lock = threading.Lock()  # Protects warm memory list
        self._hot_lock = threading.Lock()  # Protects hot memory deque

    def add_experience(
        self,
        content: str,
        topic_id: str,
        importance: float,
        source_agent: Optional[str] = None,
        emotional_valence: float = 0.0,
        keywords: Optional[list[str]] = None,
    ) -> None:
        """添加新经验到 Hot Memory，同时加入 pending_embeddings.

        Args:
            content: Experience content
            topic_id: Associated topic
            importance: Importance score [0, 1]
            source_agent: Source agent ID (if from another agent)
            emotional_valence: Emotional valence [-1, 1]
            keywords: Keywords for fast matching
        """
        # 1. 加入 Hot Memory (立即可用，无需 embedding)
        hot_item = HotMemoryItem(
            content=content,
            topic_id=topic_id,
            timestamp=datetime.now(),
            importance=importance,
            source_agent=source_agent,
            emotional_valence=emotional_valence,
            keywords=keywords or [],
        )
        self.hot.append(hot_item)

        # 2. 加入 pending queue，等待批量 embedding
        # 只有当 importance > 阈值时才进入 Warm Memory
        if importance >= self._importance_threshold:
            pending = PendingEmbeddingItem(
                agent_id=self.agent_id,
                text=content,
                metadata={
                    "topic_id": topic_id,
                    "importance": importance,
                    "timestamp": datetime.now().isoformat(),
                    "source_agent": source_agent,
                    "emotional_valence": emotional_valence,
                },
            )
            self.pending_embeddings.append(pending)

    def retrieve_relevant(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        topic_id: Optional[str] = None,
        k: int = 5,
    ) -> list[RetrievedMemory]:
        """Smart RAG 检索 - Hot -> Warm 优先级.

        Args:
            query: Query string
            query_vector: Pre-computed query embedding (optional)
            topic_id: Filter by topic
            k: Max results to return

        Returns:
            list[RetrievedMemory]: Retrieved memories, Hot first then Warm
        """
        results: list[RetrievedMemory] = []

        # Step 1: Hot Memory 关键词匹配 (O(n) 线性扫描)
        hot_matches = self._search_hot_memory(query, topic_id)
        results.extend([r.to_retrieved() for r in hot_matches])

        # 如果 Hot Memory 已足够，直接返回 (Early Return)
        if len(results) >= k:
            return results[:k]

        # Step 2: Warm Memory 向量相似度搜索
        remaining = k - len(results)
        if self.warm and (query_vector is not None or query):
            warm_matches = self._search_warm_memory(
                query_vector=query_vector,
                query_text=query if query_vector is None else None,
                topic_id=topic_id,
                k=remaining,
            )
            results.extend(warm_matches)

        return results[:k]

    def _search_hot_memory(
        self,
        query: str,
        topic_id: Optional[str] = None,
    ) -> list[HotMemoryItem]:
        """Hot Memory 关键词搜索 - 无需 embedding.

        Args:
            query: Query string
            topic_id: Filter by topic

        Returns:
            list[HotMemoryItem]: Matching hot memories
        """
        query_keywords = set(query.lower().split()) if query else set()
        matches: list[tuple[int, HotMemoryItem]] = []

        for item in self.hot:
            # 主题过滤
            if topic_id and item.topic_id != topic_id:
                continue

            # 如果没有 query，直接加入（已通过 topic 过滤）
            if not query_keywords:
                matches.append((1, item))
                continue

            # 关键词匹配
            match_score = 0
            item_text = item.content.lower()

            # 内容关键词匹配
            for kw in query_keywords:
                if kw in item_text:
                    match_score += 1

            # 预提取关键词匹配 (bonus)
            for kw in item.keywords:
                if kw.lower() in query_keywords:
                    match_score += 2

            if match_score > 0:
                matches.append((match_score, item))

        # 按匹配分数降序
        matches.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in matches]

    def _search_warm_memory(
        self,
        query_vector: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        topic_id: Optional[str] = None,
        k: int = 3,
    ) -> list[RetrievedMemory]:
        """Warm Memory 向量相似度搜索.

        Args:
            query_vector: Query embedding vector
            query_text: Query text (fallback if no vector)
            topic_id: Filter by topic
            k: Max results

        Returns:
            list[RetrievedMemory]: Retrieved warm memories
        """
        if not self.warm:
            return []

        # 如果没有 query_vector，暂时返回基于文本的匹配
        # 实际使用中，query_vector 应该由调用方预计算
        if query_vector is None:
            return self._search_warm_memory_fallback(query_text or "", topic_id, k)

        # CRITICAL FIX: Thread-safe vector index access
        with self._vector_lock:
            # Rebuild if needed
            if self._vector_index_dirty or self._vector_index is None:
                self._rebuild_vector_index_unsafe()

            if self._vector_index is None or len(self._vector_index) == 0:
                return []

            # NumPy 向量搜索 (点积相似度)
            similarities = self._vector_index @ query_vector  # shape: (N,)

        # 获取 top-k (after releasing lock)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results: list[RetrievedMemory] = []
        for idx in top_k_indices:
            if idx >= len(self.warm):
                continue

            # CRITICAL FIX: Access warm memory under lock
            with self._warm_lock:
                warm_item = self.warm[idx]

            # 主题过滤
            if topic_id and topic_id not in warm_item.topic_ids:
                continue

            # 更新访问统计
            warm_item.access_count += 1
            warm_item.last_accessed = datetime.now()

            results.append(
                RetrievedMemory(
                    text=warm_item.text,
                    source="warm",
                    importance=warm_item.importance_score,
                    timestamp=warm_item.timestamp,
                    similarity=float(similarities[idx]),
                    topic_id=warm_item.topic_ids[0] if warm_item.topic_ids else None,
                )
            )

        return results

    def _search_warm_memory_fallback(
        self,
        query: str,
        topic_id: Optional[str] = None,
        k: int = 3,
    ) -> list[RetrievedMemory]:
        """Warm Memory 文本回退搜索 (无向量时)."""
        query_keywords = set(query.lower().split())
        matches: list[tuple[int, WarmMemoryItem]] = []

        for item in self.warm:
            if topic_id and topic_id not in item.topic_ids:
                continue

            item_text = item.text.lower()
            match_score = sum(1 for kw in query_keywords if kw in item_text)

            if match_score > 0:
                matches.append((match_score, item))

        matches.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievedMemory(
                text=item.text,
                source="warm",
                importance=item.importance_score,
                timestamp=item.timestamp,
                similarity=0.0,  # 无相似度分数
                topic_id=item.topic_ids[0] if item.topic_ids else None,
            )
            for _, item in matches[:k]
        ]

    def _rebuild_vector_index_unsafe(self) -> None:
        """重建向量索引 - 必须在持有 _vector_lock 时调用.

        CRITICAL FIX: Unsafe version - caller must hold lock!
        """
        if not self.warm:
            self._vector_index = None
            self._vector_index_dirty = False
            return

        # 收集所有向量
        vectors = []
        for item in self.warm:
            if item.vector is not None:
                vectors.append(item.vector)
            else:
                # 填充零向量 (不应发生)
                vectors.append(np.zeros(self.DEFAULT_VECTOR_DIM, dtype=np.float32))

        if vectors:
            self._vector_index = np.stack(vectors)
        else:
            self._vector_index = None

        self._vector_index_dirty = False

    def on_batch_embeddings_received(
        self,
        embeddings: list[tuple[str, np.ndarray, dict[str, Any]]],
    ) -> None:
        """接收引擎分发的批量 embedding 结果.

        Args:
            embeddings: List of (text, vector, metadata) tuples
        """
        # CRITICAL FIX: Thread-safe batch embedding reception
        with self._warm_lock:
            for text, vector, metadata in embeddings:
                warm_item = WarmMemoryItem(
                    text=text,
                    vector=vector,
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    importance_score=metadata.get("importance", 0.5),
                    last_accessed=datetime.now(),
                    topic_ids=[metadata.get("topic_id", "")] if metadata.get("topic_id") else [],
                    related_agents=[metadata.get("source_agent")] if metadata.get("source_agent") else [],
                )
                self.warm.append(warm_item)

        # CRITICAL FIX: Mark index dirty under vector lock
        with self._vector_lock:
            self._vector_index_dirty = True

        # 限制 Warm Memory 大小 (outside lock)
        self._enforce_warm_memory_limit()

    def _enforce_warm_memory_limit(self) -> None:
        """LRU + 重要性淘汰策略."""
        # CRITICAL FIX: Thread-safe memory limit enforcement
        with self._warm_lock:
            if len(self.warm) <= self.warm_max_size:
                return

            # 计算综合分数: 重要性 * 0.7 + 访问频率 * 0.2 + 时间衰减 * 0.1
            now = datetime.now()
            scored: list[tuple[float, WarmMemoryItem]] = []

            for item in self.warm:
                # 时间衰减 (最近访问的分数高)
                hours_since_access = (now - item.last_accessed).total_seconds() / 3600
                recency_score = np.exp(-hours_since_access / 48.0)  # 48小时半衰期

                # 综合分数
                score = (
                    item.importance_score * 0.7
                    + min(item.access_count / 10.0, 1.0) * 0.2
                    + recency_score * 0.1
                )
                scored.append((score, item))

            # 按分数降序，保留前 N
            scored.sort(key=lambda x: x[0], reverse=True)
            self.warm = [item for _, item in scored[: self.warm_max_size]]

        # CRITICAL FIX: Mark index dirty under vector lock
        with self._vector_lock:
            self._vector_index_dirty = True

    def get_recent(self, limit: int = 10) -> list[HotMemoryItem]:
        """获取最近的 Hot Memory.

        Args:
            limit: Max items to return

        Returns:
            list[HotMemoryItem]: Recent hot memories
        """
        return list(self.hot)[-limit:]

    def get_all_warm(self) -> list[WarmMemoryItem]:
        """获取所有 Warm Memory.

        Returns:
            list[WarmMemoryItem]: All warm memories
        """
        return self.warm.copy()

    def clear(self) -> None:
        """Clear all memories (except Cold)."""
        self.hot.clear()
        self.warm.clear()
        self.pending_embeddings.clear()
        self._vector_index = None
        self._vector_index_dirty = True

    def __len__(self) -> int:
        """Get total memory count (Hot + Warm)."""
        return len(self.hot) + len(self.warm)

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            dict: Memory stats
        """
        return {
            "agent_id": self.agent_id,
            "cold_memory_size": len(self.cold.core_goals) + len(self.cold.high_confidence_stances),
            "hot_memory_size": len(self.hot),
            "warm_memory_size": len(self.warm),
            "pending_embeddings": len(self.pending_embeddings),
            "vector_index_dirty": self._vector_index_dirty,
        }


# Backward compatibility aliases
MemorySystem = MemorySystem3Tier

__all__ = [
    "MemorySystem3Tier",
    "MemorySystem",
    "MemoryItem",
    "ColdMemory",
    "HotMemoryItem",
    "WarmMemoryItem",
    "PendingEmbeddingItem",
    "RetrievedMemory",
]
