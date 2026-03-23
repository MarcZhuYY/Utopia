"""Pydantic v2 models for 3-tier memory system.

Defines HotMemoryItem, WarmMemoryItem, and related models for the
Hot/Warm/Cold memory architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class HotMemoryItem(BaseModel):
    """热记忆 - 最近 1-3 ticks 的原始记忆，无需 embedding.

    直接存储在 deque 中，支持快速关键词匹配，无需向量计算。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    content: str
    topic_id: str
    timestamp: datetime
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    source_agent: Optional[str] = None
    emotional_valence: float = Field(ge=-1.0, le=1.0, default=0.0)

    # 用于快速关键词匹配，无需向量计算
    keywords: list[str] = Field(default_factory=list)

    def to_retrieved(self) -> RetrievedMemory:
        """转换为检索结果格式."""
        return RetrievedMemory(
            text=self.content,
            source="hot",
            importance=self.importance,
            timestamp=self.timestamp,
            topic_id=self.topic_id,
        )


class WarmMemoryItem(BaseModel):
    """温记忆 - 经过 consolidation 的经验，带有向量 embedding (Pydantic v2).

    Attributes:
        text: 归并后的文本摘要
        vector: embedding 向量 (768/1536-dim)
        timestamp: 创建时间
        importance_score: 重要性评分 [0, 1]
        access_count: 访问计数，用于 LRU 淘汰
        last_accessed: 最后访问时间
        topic_ids: 关联的主题列表
        related_agents: 相关的 Agent 列表
        memory_type: 记忆类型 (experience | fact | emotion)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    text: str
    vector: Optional[np.ndarray] = None  # 768 or 1536 dimensional
    timestamp: datetime
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    access_count: int = Field(ge=0, default=0)
    last_accessed: datetime

    # 元数据用于过滤
    topic_ids: list[str] = Field(default_factory=list)
    related_agents: list[str] = Field(default_factory=list)
    memory_type: str = Field(default="experience")  # experience | fact | emotion

    def to_retrieved(self, similarity: float = 0.0) -> RetrievedMemory:
        """转换为检索结果格式."""
        return RetrievedMemory(
            text=self.text,
            source="warm",
            importance=self.importance_score,
            timestamp=self.timestamp,
            similarity=similarity,
            topic_id=self.topic_ids[0] if self.topic_ids else None,
        )


@dataclass
class PendingEmbeddingItem:
    """待处理的 embedding 请求.

    用于收集所有 Agent 的 pending embeddings，等待批量处理。
    """

    agent_id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """验证必填字段."""
        if not self.agent_id:
            raise ValueError("agent_id is required")
        if not self.text:
            raise ValueError("text is required")


@dataclass
class RetrievedMemory:
    """检索结果记忆项.

    统一 Hot Memory 和 Warm Memory 的检索结果格式。
    """

    text: str
    source: str  # "hot" | "warm" | "cold"
    importance: float = 0.5
    timestamp: Optional[datetime] = None
    similarity: float = 0.0  # 向量相似度 (仅 warm)
    topic_id: Optional[str] = None


@dataclass
class ColdMemory:
    """冷记忆 - 始终保留在系统提示中.

    包含 Agent 的静态信息，如角色定义、高置信度立场等。
    """

    persona_summary: str = ""  # Agent 角色、目标、核心特质
    high_confidence_stances: dict[str, float] = field(default_factory=dict)  # 置信度 > 0.95 的立场
    static_knowledge: dict[str, Any] = field(default_factory=dict)  # 领域知识、不变的事实
    core_goals: list[str] = field(default_factory=list)  # 核心目标

    def to_system_prompt_section(self) -> str:
        """转换为系统提示的一部分."""
        lines = [
            "## Core Identity",
            self.persona_summary,
            "",
            "## High Confidence Beliefs",
        ]
        for topic, stance in self.high_confidence_stances.items():
            lines.append(f"- {topic}: {stance:+.2f}")

        if self.core_goals:
            lines.extend(["", "## Core Goals"])
            for goal in self.core_goals:
                lines.append(f"- {goal}")

        return "\n".join(lines)
