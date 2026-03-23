"""Engine-level batch embedding processor.

Implements Lazy Batch Embedding: collects pending embeddings from all agents
at the end of each tick and processes them in a single batch API call.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np

from utopia.layer3_cognition.warm_memory_models import PendingEmbeddingItem

if TYPE_CHECKING:
    from utopia.layer3_cognition.agent import Agent


class EmbeddingClient(Protocol):
    """Protocol for embedding API clients."""

    async def embeddings_create(
        self,
        model: str,
        input: list[str],
    ) -> Any:
        """Create embeddings for texts."""
        ...


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding processing."""

    agent_id: str
    text: str
    vector: np.ndarray
    metadata: dict[str, Any]


class BatchEmbeddingProcessor:
    """
    引擎级批量向量化处理器.

    核心职责:
    1. 收集所有 Agent 的 pending_embeddings
    2. 批量调用 embedding API (单次调用)
    3. 分发向量回各 Agent

    性能目标:
    - 将 N agents × M experiences 次 API 调用压缩为 1 次 batch 调用
    - 减少 90%+ 的 API 开销
    """

    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_EMBEDDING_DIM = 768  # text-embedding-3-small
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        api_key: Optional[str] = None,
    ):
        """Initialize batch embedding processor.

        Args:
            embedding_model: Embedding model name
            batch_size: Max texts per batch
            embedding_dim: Embedding dimension
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._embedding_client: Optional[Any] = None

    async def process_tick_embeddings(
        self,
        agents: dict[str, Agent],
    ) -> dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]]:
        """
        处理当前 Tick 的所有 pending embeddings.

        这是核心方法：
        1. 遍历所有 Agent，收集 pending_embeddings
        2. 批量调用 embedding API (单次调用！)
        3. 按 agent_id 分组，返回结果

        Args:
            agents: Agent ID -> Agent 对象映射

        Returns:
            dict: Agent ID -> 该 Agent 的 embedding 结果列表 [(text, vector, metadata), ...]
        """
        # 1. 收集所有 pending items
        all_pending: list[PendingEmbeddingItem] = []
        for agent in agents.values():
            if hasattr(agent, "memory") and hasattr(agent.memory, "pending_embeddings"):
                pending = agent.memory.pending_embeddings
                all_pending.extend(pending)
                pending.clear()  # 清空已提交的

        if not all_pending:
            return {}

        # 2. 批量调用 embedding API (单次调用！)
        texts = [item.text for item in all_pending]
        vectors = await self._batch_embed(texts)

        # 3. 按 agent_id 分组结果
        results: dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]] = defaultdict(list)
        for item, vector in zip(all_pending, vectors):
            results[item.agent_id].append((item.text, vector, item.metadata))

        return dict(results)

    async def _batch_embed(self, texts: list[str]) -> list[np.ndarray]:
        """
        批量 embedding API 调用.

        注意：这是唯一的 embedding API 调用点！
        所有 texts 被分批处理，但总共只发起最少的 API 调用。

        Args:
            texts: List of texts to embed

        Returns:
            list[np.ndarray]: Embedding vectors
        """
        if not texts:
            return []

        # 初始化客户端
        if self._embedding_client is None:
            self._embedding_client = self._init_embedding_client()

        # 分批处理（避免超过 API 限制）
        all_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_vectors = await self._embed_batch(batch)
            all_vectors.extend(batch_vectors)

        return all_vectors

    async def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a single batch of texts.

        Args:
            texts: Batch of texts

        Returns:
            list[np.ndarray]: Embedding vectors for this batch
        """
        try:
            # OpenAI API 调用
            response = await self._embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )

            # 转换为 numpy 数组
            vectors = [
                np.array(data.embedding, dtype=np.float32)
                for data in response.data
            ]
            return vectors

        except Exception as e:
            # 降级：返回随机向量 (用于测试)
            import warnings
            warnings.warn(f"Embedding API call failed: {e}. Returning zero vectors.")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]

    def _init_embedding_client(self) -> Any:
        """Initialize embedding API client.

        Returns:
            AsyncOpenAI client
        """
        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "openai package is required for embedding. "
                "Install with: pip install openai"
            )

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text (for queries).

        Args:
            text: Text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        vectors = await self._batch_embed([text])
        return vectors[0]

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics.

        Returns:
            dict: Processor configuration
        """
        return {
            "embedding_model": self.embedding_model,
            "batch_size": self.batch_size,
            "embedding_dim": self.embedding_dim,
            "client_initialized": self._embedding_client is not None,
        }


class MockBatchEmbeddingProcessor(BatchEmbeddingProcessor):
    """Mock processor for testing - returns random vectors without API calls."""

    def __init__(
        self,
        embedding_dim: int = 768,
        batch_size: int = 1000,
        **kwargs,
    ):
        """Initialize mock processor.

        Args:
            embedding_dim: Dimension of mock vectors
            batch_size: Batch size for processing
            **kwargs: Ignored (for API compatibility)
        """
        self.embedding_dim = embedding_dim
        self.embedding_model = "mock-model"
        self.batch_size = batch_size
        self._api_key = None
        self._embedding_client = None
        self._call_count = 0

    async def _batch_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Override to avoid client initialization - return random vectors."""
        if not texts:
            return []

        all_vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_vectors = await self._embed_batch(batch)
            all_vectors.extend(batch_vectors)

        return all_vectors

    async def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Return random vectors instead of calling API."""
        self._call_count += 1
        return [
            np.random.randn(self.embedding_dim).astype(np.float32)
            for _ in texts
        ]

    @property
    def call_count(self) -> int:
        """Number of batch embed calls made."""
        return self._call_count

    def reset_count(self) -> None:
        """Reset call counter."""
        self._call_count = 0
