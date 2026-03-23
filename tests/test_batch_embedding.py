"""Tests for batch embedding processor.

Tests the Lazy Batch Embedding mechanism:
- 3 agents × 2 experiences → 1 API call
- Correct vector distribution to agents
"""

import pytest
import numpy as np
from datetime import datetime

from utopia.layer3_cognition.warm_memory_models import ColdMemory
from utopia.layer3_cognition.memory_3tier import MemorySystem3Tier
from utopia.layer5_engine.batch_embedding_processor import (
    BatchEmbeddingProcessor,
    MockBatchEmbeddingProcessor,
)


def create_test_agent(agent_id: str) -> MockAgent:
    """Helper to create test agent with 3-tier memory."""
    cold = ColdMemory(
        persona_summary=f"Test agent {agent_id}",
        high_confidence_stances={},
    )
    memory = MemorySystem3Tier(
        agent_id=agent_id,
        cold_memory=cold,
        hot_memory_maxlen=10,
    )
    return MockAgent(agent_id=agent_id, memory=memory)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, agent_id: str, memory: MemorySystem3Tier):
        self.agent_id = agent_id
        self.memory = memory


@pytest.mark.asyncio
async def test_batch_embedding_processor_single_api_call():
    """
    核心测试：3 个 Agent 各产生 2 条经验，断言只发起 1 次 mock API 调用。

    This validates the key Phase 8 requirement:
    - N agents × M experiences → 1 batch API call
    - 90%+ API cost reduction
    """
    # 创建 3 个 Agent
    agents = {
        f"agent_{i}": create_test_agent(f"agent_{i}")
        for i in range(3)
    }

    # 每个 Agent 添加 2 条经验
    for agent in agents.values():
        agent.memory.add_experience(
            content="Experience 1 from agent",
            topic_id="topic_1",
            importance=0.5,
        )
        agent.memory.add_experience(
            content="Experience 2 from agent",
            topic_id="topic_2",
            importance=0.6,
        )

    # 创建 Mock Embedding Processor (跟踪调用次数)
    processor = MockBatchEmbeddingProcessor(embedding_dim=768)

    # 执行批量处理
    results = await processor.process_tick_embeddings(agents)

    # ===== 关键断言 =====
    # 断言：只有 1 次 API 调用 (核心指标！)
    assert processor.call_count == 1, (
        f"Expected 1 API call, got {processor.call_count}. "
        "Batch embedding failed - API calls not consolidated!"
    )

    # 断言：6 个 vectors 正确分发 (3 agents × 2 experiences)
    assert len(results) == 3, f"Expected results for 3 agents, got {len(results)}"
    total_vectors = sum(len(v) for v in results.values())
    assert total_vectors == 6, f"Expected 6 vectors total, got {total_vectors}"

    # 断言：每个 Agent 收到 2 个 vectors
    for agent_id in agents:
        assert agent_id in results, f"Missing results for {agent_id}"
        assert len(results[agent_id]) == 2, (
            f"Expected 2 vectors for {agent_id}, got {len(results[agent_id])}"
        )

    # 断言：每个 vector 维度正确
    for agent_id, embeddings in results.items():
        for text, vector, metadata in embeddings:
            assert isinstance(vector, np.ndarray), f"Vector should be numpy array"
            assert vector.shape == (768,), f"Expected shape (768,), got {vector.shape}"
            assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"


@pytest.mark.asyncio
async def test_batch_embedding_no_pending():
    """Test: 没有 pending embeddings 时不发起 API 调用."""
    # 创建 agents 但不添加经验
    agents = {
        f"agent_{i}": create_test_agent(f"agent_{i}")
        for i in range(3)
    }

    processor = MockBatchEmbeddingProcessor(embedding_dim=768)
    results = await processor.process_tick_embeddings(agents)

    # 断言：0 次 API 调用
    assert processor.call_count == 0
    assert len(results) == 0


@pytest.mark.asyncio
async def test_batch_embedding_vector_distribution():
    """Test: vectors 正确分发给对应 agents."""
    agents = {
        "agent_A": create_test_agent("agent_A"),
        "agent_B": create_test_agent("agent_B"),
    }

    # Agent A: 1 条经验
    agents["agent_A"].memory.add_experience(
        content="Agent A experience",
        topic_id="topic_A",
        importance=0.7,
    )

    # Agent B: 2 条经验
    agents["agent_B"].memory.add_experience(
        content="Agent B first experience",
        topic_id="topic_B",
        importance=0.8,
    )
    agents["agent_B"].memory.add_experience(
        content="Agent B second experience",
        topic_id="topic_B",
        importance=0.9,
    )

    processor = MockBatchEmbeddingProcessor(embedding_dim=768)
    results = await processor.process_tick_embeddings(agents)

    # 验证分发
    assert len(results["agent_A"]) == 1
    assert len(results["agent_B"]) == 2

    # 验证内容匹配
    assert results["agent_A"][0][0] == "Agent A experience"
    assert results["agent_B"][0][0] == "Agent B first experience"
    assert results["agent_B"][1][0] == "Agent B second experience"


@pytest.mark.asyncio
async def test_batch_embedding_pending_queue_cleared():
    """Test: 处理后 pending_embeddings 被清空."""
    agent = create_test_agent("test_agent")

    # 添加经验
    agent.memory.add_experience(
        content="Test experience",
        topic_id="test_topic",
        importance=0.5,
    )

    # 断言：pending 队列有内容
    assert len(agent.memory.pending_embeddings) == 1

    # 执行批量处理
    processor = MockBatchEmbeddingProcessor(embedding_dim=768)
    await processor.process_tick_embeddings({"test_agent": agent})

    # 断言：pending 队列被清空
    assert len(agent.memory.pending_embeddings) == 0


@pytest.mark.asyncio
async def test_batch_embedding_large_batch():
    """Test: 大批量数据分批处理."""
    # 创建单个 agent，添加大量经验
    agent = create_test_agent("test_agent")

    # 添加 150 条经验 (超过默认 batch_size=100)
    for i in range(150):
        agent.memory.add_experience(
            content=f"Experience {i}",
            topic_id=f"topic_{i % 10}",
            importance=0.5,
        )

    processor = MockBatchEmbeddingProcessor(embedding_dim=768)
    processor.batch_size = 50  # 小批次便于测试

    results = await processor.process_tick_embeddings({"test_agent": agent})

    # 断言：分 3 批处理 (150 / 50 = 3)
    assert processor.call_count == 3

    # 断言：所有 vectors 都收到
    assert len(results["test_agent"]) == 150


def test_mock_processor_reset():
    """Test: Mock processor reset_count 功能."""
    processor = MockBatchEmbeddingProcessor(embedding_dim=768)

    # 手动增加调用计数
    processor._call_count = 5
    assert processor.call_count == 5

    # 重置
    processor.reset_count()
    assert processor.call_count == 0
