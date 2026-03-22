"""Tests for Layer 3 (Individual Cognition) - Memory System."""

import pytest
from datetime import datetime

from utopia.core.models import MemoryItem, Persona
from utopia.layer3_cognition.agent import Agent
from utopia.layer3_cognition.memory import MemorySystem


class TestMemorySystem:
    """Test memory system."""

    def test_empty_memory(self):
        """Test empty memory."""
        memory = MemorySystem()
        assert len(memory) == 0

    def test_add_memory(self):
        """Test adding memory item."""
        memory = MemorySystem()
        item = MemoryItem(
            content="Test event happened",
            importance=0.5,  # Below promotion threshold
        )

        memory.add(item, "A1")

        assert len(memory) == 1

    def test_memory_retrieval(self):
        """Test memory retrieval."""
        memory = MemorySystem()

        # Add items
        memory.add(MemoryItem(content="AI conference today", importance=0.8), "A1")
        memory.add(MemoryItem(content="Stock market up", importance=0.6), "A1")
        memory.add(MemoryItem(content="Weather nice", importance=0.2), "A1")

        # Retrieve
        results = memory.retrieve("AI", limit=2)
        assert len(results) >= 1
        assert "AI" in results[0].content

    def test_short_term_capacity(self):
        """Test short term capacity limit."""
        memory = MemorySystem(short_term_capacity=3)

        for i in range(5):
            memory.add(MemoryItem(content=f"Item {i}"), "A1")

        # Should evict old items
        assert len(memory.short_term) <= 3


class TestAgent:
    """Test Agent model."""

    def test_agent_creation(self):
        """Test agent creation."""
        agent = Agent()
        assert agent.id.startswith("A")
        assert agent.persona.name == f"Agent_{agent.id}"

    def test_agent_from_entity(self):
        """Test creating agent from entity."""
        agent = Agent.from_entity(
            entity_id="E1",
            name="John Doe",
            role="investor",
            expertise=["finance"],
            base_stances={"regulation": 0.3},
            influence=0.7,
        )

        assert agent.id == "E1"
        assert agent.persona.name == "John Doe"
        assert agent.persona.role == "investor"
        assert agent.get_stance("regulation") is not None

    def test_agent_add_memory(self):
        """Test adding memory to agent."""
        agent = Agent()
        agent.add_memory(
            content="Important news",
            topic_id="tech",
            importance=0.8,
        )

        memories = agent.retrieve_memories("important")
        assert len(memories) >= 1
