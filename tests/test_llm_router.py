"""Tests for LLM Router (capability-based routing)."""

import asyncio
import pytest

from utopia.core.pydantic_models import (
    LLMModel,
    TaskRequest,
    TaskType,
)
from utopia.layer5_engine.llm_router import (
    LLMRouter,
    ExponentialBackoff,
    LONG_CONTEXT_THRESHOLD,
    MODEL_CONCURRENCY,
    MODEL_CAPABILITIES,
    FALLBACK_CHAIN,
)


class TestExponentialBackoff:
    """Test exponential backoff mechanism."""

    def test_delay_calculation(self):
        """Test delay = base * 2^attempt + jitter."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=30.0, jitter=0.5)

        # Attempt 0: delay = 1.0 * 1 + jitter
        delay0 = backoff.compute_delay(0)
        assert 1.0 <= delay0 <= 1.5

        # Attempt 1: delay = 1.0 * 2 + jitter
        delay1 = backoff.compute_delay(1)
        assert 2.0 <= delay1 <= 2.5

        # Attempt 2: delay = 1.0 * 4 + jitter
        delay2 = backoff.compute_delay(2)
        assert 4.0 <= delay2 <= 4.5

    def test_max_delay_cap(self):
        """Test delay doesn't exceed max_delay."""
        backoff = ExponentialBackoff(base_delay=10.0, max_delay=30.0)

        # Attempt 5: 10 * 32 = 320, but capped at 30
        delay = backoff.compute_delay(5)
        assert delay <= 30.5  # including jitter


class TestModelSelection:
    """Test capability-based model selection."""

    @pytest.fixture
    def router(self):
        return LLMRouter(enable_fallback=False, enable_caching=False)

    def test_long_context_forces_kimi(self, router):
        """Test context > 30K forces routing to Kimi."""
        request = TaskRequest(
            task_id="test_1",
            task_type=TaskType.DEFAULT,  # Would normally go to MiniMax
            agent_id="agent_1",
            prompt="x" * 40000,
            context_length=40000,  # > 30K threshold
        )

        model = router._select_model(request)
        assert model == LLMModel.KIMI_K25

    def test_bayesian_routes_to_deepseek(self, router):
        """Test BAYESIAN_STANCE_UPDATE routes to DeepSeek."""
        request = TaskRequest(
            task_id="test_2",
            task_type=TaskType.BAYESIAN_STANCE_UPDATE,
            agent_id="agent_1",
            prompt="Calculate Bayesian update",
        )

        model = router._select_model(request)
        assert model == LLMModel.DEEPSEEK_R1

    def test_memory_routes_to_kimi(self, router):
        """Test MEMORY_CONSOLIDATION routes to Kimi."""
        request = TaskRequest(
            task_id="test_3",
            task_type=TaskType.MEMORY_CONSOLIDATION,
            agent_id="agent_1",
            prompt="Consolidate memories",
        )

        model = router._select_model(request)
        assert model == LLMModel.KIMI_K25

    def test_rule_validation_routes_to_glm(self, router):
        """Test RULE_VALIDATION routes to GLM."""
        request = TaskRequest(
            task_id="test_4",
            task_type=TaskType.RULE_VALIDATION,
            agent_id="system",
            prompt="Validate agent behavior",
        )

        model = router._select_model(request)
        assert model == LLMModel.GLM_5

    def test_high_freq_routes_to_qwen(self, router):
        """Test HIGH_FREQ_INTERACT routes to Qwen."""
        request = TaskRequest(
            task_id="test_5",
            task_type=TaskType.HIGH_FREQ_INTERACT,
            agent_id="agent_1",
            prompt="Quick interaction",
        )

        model = router._select_model(request)
        assert model == LLMModel.QWEN_35_PLUS

    def test_default_routes_to_minimax(self, router):
        """Test DEFAULT routes to MiniMax."""
        request = TaskRequest(
            task_id="test_6",
            task_type=TaskType.DEFAULT,
            agent_id="agent_1",
            prompt="Complex decision",
        )

        model = router._select_model(request)
        assert model == LLMModel.MINIMAX_M27


class TestFallback:
    """Test fallback mechanism."""

    def test_fallback_chain(self):
        """Test all models fallback to MiniMax."""
        assert FALLBACK_CHAIN[LLMModel.DEEPSEEK_R1] == LLMModel.MINIMAX_M27
        assert FALLBACK_CHAIN[LLMModel.KIMI_K25] == LLMModel.MINIMAX_M27
        assert FALLBACK_CHAIN[LLMModel.GLM_5] == LLMModel.MINIMAX_M27
        assert FALLBACK_CHAIN[LLMModel.QWEN_35_PLUS] == LLMModel.MINIMAX_M27
        assert FALLBACK_CHAIN[LLMModel.MINIMAX_M27] is None

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test fallback when primary model fails."""

        async def failing_api(prompt, model):
            raise Exception("API Error")

        router = LLMRouter(
            api_call_func=failing_api,
            enable_fallback=True,
        )

        request = TaskRequest(
            task_id="test_fallback",
            task_type=TaskType.BAYESIAN_STANCE_UPDATE,
            agent_id="agent_1",
            prompt="Test fallback",
            timeout=0.1,
        )

        response = await router.route(request)

        # Should fail (both primary and fallback fail)
        assert not response.success
        # Primary model was DeepSeek
        assert response.model_used == LLMModel.DEEPSEEK_R1


class TestConcurrency:
    """Test semaphore-based concurrency control."""

    def test_semaphore_limits(self):
        """Test each model has correct concurrency limit."""
        router = LLMRouter()

        for model, expected_limit in MODEL_CONCURRENCY.items():
            semaphore = router._semaphores[model]
            # Semaphore value equals max_concurrent
            assert semaphore._value == expected_limit

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """Test multiple requests execute concurrently."""
        execution_order = []

        async def tracking_api(prompt, model):
            execution_order.append((model, "start"))
            await asyncio.sleep(0.01)  # Small delay
            execution_order.append((model, "end"))
            return f"Response from {model.value}"

        router = LLMRouter(api_call_func=tracking_api)

        # Create 5 concurrent requests
        requests = [
            TaskRequest(
                task_id=f"concurrent_{i}",
                task_type=TaskType.DEFAULT,
                agent_id=f"agent_{i}",
                prompt=f"Request {i}",
            )
            for i in range(5)
        ]

        responses = await router.route_batch(requests)

        # All should succeed
        assert all(r.success for r in responses)
        # Should have started multiple before any ended
        assert len(execution_order) > 5


class TestCaching:
    """Test response caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test identical requests hit cache."""
        call_count = 0

        async def counting_api(prompt, model):
            nonlocal call_count
            call_count += 1
            return f"Response {call_count}"

        router = LLMRouter(api_call_func=counting_api, enable_caching=True)

        request = TaskRequest(
            task_id="cache_test_1",
            task_type=TaskType.DEFAULT,
            agent_id="agent_1",
            prompt="Same prompt",
        )

        # First call
        response1 = await router.route(request)
        assert response1.success
        assert not response1.fallback_used
        assert call_count == 1

        # Second call (should hit cache)
        response2 = await router.route(request)
        assert response2.success
        # Cache hit has 0 latency
        assert response2.latency_ms == 0.0
        # API not called again
        assert call_count == 1

    def test_cache_stats(self):
        """Test cache statistics."""
        router = LLMRouter(enable_caching=True)

        # Initially empty
        stats = router.get_cache_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["hit_rate"] == 0.0


class TestStatistics:
    """Test router statistics."""

    @pytest.mark.asyncio
    async def test_model_distribution(self):
        """Test model usage distribution tracking."""
        router = LLMRouter()

        # Route different tasks
        requests = [
            TaskRequest(
                task_id="stat_1",
                task_type=TaskType.BAYESIAN_STANCE_UPDATE,
                agent_id="agent_1",
                prompt="Test",
            ),
            TaskRequest(
                task_id="stat_2",
                task_type=TaskType.MEMORY_CONSOLIDATION,
                agent_id="agent_2",
                prompt="Test",
            ),
            TaskRequest(
                task_id="stat_3",
                task_type=TaskType.DEFAULT,
                agent_id="agent_3",
                prompt="Test",
            ),
        ]

        await router.route_batch(requests)

        stats = router.get_stats()
        assert stats.total_requests == 3
        assert stats.successful_requests == 3

        # Check distribution
        assert LLMModel.DEEPSEEK_R1 in stats.model_distribution
        assert LLMModel.KIMI_K25 in stats.model_distribution
        assert LLMModel.MINIMAX_M27 in stats.model_distribution


class TestModelCapabilities:
    """Test model capability configurations."""

    def test_minimax_config(self):
        """Test MiniMax capability configuration."""
        config = MODEL_CAPABILITIES[LLMModel.MINIMAX_M27]
        assert config.model == LLMModel.MINIMAX_M27
        assert TaskType.DEFAULT in config.specialties
        assert config.max_concurrent == 20
        assert config.provider == "minimax"

    def test_deepseek_config(self):
        """Test DeepSeek capability configuration."""
        config = MODEL_CAPABILITIES[LLMModel.DEEPSEEK_R1]
        assert TaskType.BAYESIAN_STANCE_UPDATE in config.specialties
        assert TaskType.COGNITIVE_DISSONANCE in config.specialties
        assert config.context_window == 64000
        assert config.max_concurrent == 10

    def test_kimi_config(self):
        """Test Kimi capability configuration."""
        config = MODEL_CAPABILITIES[LLMModel.KIMI_K25]
        assert TaskType.MEMORY_CONSOLIDATION in config.specialties
        assert TaskType.LONG_CONTEXT_PARSE in config.specialties
        assert config.context_window == 200000

    def test_glm_config(self):
        """Test GLM capability configuration."""
        config = MODEL_CAPABILITIES[LLMModel.GLM_5]
        assert TaskType.RULE_VALIDATION in config.specialties
        assert config.context_window == 8000

    def test_qwen_config(self):
        """Test Qwen capability configuration."""
        config = MODEL_CAPABILITIES[LLMModel.QWEN_35_PLUS]
        assert TaskType.HIGH_FREQ_INTERACT in config.specialties
        assert TaskType.INFO_PROPAGATION in config.specialties
        assert config.max_concurrent == 30
