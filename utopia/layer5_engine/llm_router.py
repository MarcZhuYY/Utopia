"""LLM Router - Capability-Based Model Routing

全旗舰模型专长路由分发器

设计思想:
- 引擎不考虑预算降级，只要求最高保真度
- 根据"任务属性"将Agent的思考请求路由给特定专长的顶级大模型
- Minimax m2.7作为算力充裕的主力底座，担任统一降级目标

模型专长分配:
- MINIMAX_M27: 主力底座，默认路由，处理复杂Agent协作
- DEEPSEEK_R1: 数学与逻辑审核，专攻纯逻辑推理
- KIMI_K25: 记忆档案馆，专攻超长文本无损压缩
- GLM_5: 世界法则裁判，担任RuleValidator
- QWEN_35_PLUS: 神经末梢，处理高频浅层交互
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

from utopia.core.pydantic_models import (
    LLMModel,
    LLMResponse,
    ModelCapability,
    RouterStats,
    TaskRequest,
    TaskType,
)
from utopia.core.utils import ExponentialBackoff


# =============================================================================
# Model Configuration
# =============================================================================

# 长文本阈值 (30K tokens ≈ 30KB for estimation)
LONG_CONTEXT_THRESHOLD: int = 30000

# 各模型并发限制配置
MODEL_CONCURRENCY: dict[LLMModel, int] = {
    # Minimax算力充裕，配置较高并发
    LLMModel.MINIMAX_M27: 20,
    # 其他模型标准并发
    LLMModel.DEEPSEEK_R1: 10,
    LLMModel.KIMI_K25: 10,
    LLMModel.GLM_5: 15,
    LLMModel.QWEN_35_PLUS: 30,  # 高频交互需要更高并发
}

# 模型能力配置
MODEL_CAPABILITIES: dict[LLMModel, ModelCapability] = {
    LLMModel.MINIMAX_M27: ModelCapability(
        model=LLMModel.MINIMAX_M27,
        specialties=[TaskType.DEFAULT],
        context_window=32000,
        max_concurrent=20,
        provider="minimax",
        api_key_env="MINIMAX_API_KEY",
    ),
    LLMModel.DEEPSEEK_R1: ModelCapability(
        model=LLMModel.DEEPSEEK_R1,
        specialties=[TaskType.BAYESIAN_STANCE_UPDATE, TaskType.COGNITIVE_DISSONANCE],
        context_window=64000,
        max_concurrent=10,
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    LLMModel.KIMI_K25: ModelCapability(
        model=LLMModel.KIMI_K25,
        specialties=[TaskType.MEMORY_CONSOLIDATION, TaskType.LONG_CONTEXT_PARSE],
        context_window=200000,
        max_concurrent=10,
        provider="kimi",
        api_key_env="KIMI_API_KEY",
    ),
    LLMModel.GLM_5: ModelCapability(
        model=LLMModel.GLM_5,
        specialties=[TaskType.RULE_VALIDATION],
        context_window=8000,
        max_concurrent=15,
        provider="zhipu",
        api_key_env="ZHIPU_API_KEY",
    ),
    LLMModel.QWEN_35_PLUS: ModelCapability(
        model=LLMModel.QWEN_35_PLUS,
        specialties=[TaskType.HIGH_FREQ_INTERACT, TaskType.INFO_PROPAGATION],
        context_window=128000,
        max_concurrent=30,
        provider="qwen",
        api_key_env="QWEN_API_KEY",
    ),
}

# 统一降级路径: 所有模型都降级到 MINIMAX_M27
FALLBACK_CHAIN: dict[LLMModel, LLMModel] = {
    LLMModel.DEEPSEEK_R1: LLMModel.MINIMAX_M27,
    LLMModel.KIMI_K25: LLMModel.MINIMAX_M27,
    LLMModel.GLM_5: LLMModel.MINIMAX_M27,
    LLMModel.QWEN_35_PLUS: LLMModel.MINIMAX_M27,
    LLMModel.MINIMAX_M27: None,  # type: ignore  # 主力底座无降级路径
}


# =============================================================================
# LLM Router
# =============================================================================


class LLMRouter:
    """全旗舰模型专长路由器

    核心职责:
    1. 基于任务类型路由到专长模型
    2. 管理各模型并发信号量
    3. 实现指数退避重试
    4. 统一降级到 MINIMAX_M27

    Usage:
        router = LLMRouter()
        request = TaskRequest(
            task_id="task_001",
            task_type=TaskType.BAYESIAN_STANCE_UPDATE,
            agent_id="agent_1",
            prompt="计算贝叶斯立场更新...",
        )
        response = await router.route(request)
    """

    def __init__(
        self,
        api_call_func: Optional[
            Callable[[str, LLMModel], Coroutine[Any, Any, str]]
        ] = None,
        enable_fallback: bool = True,
        enable_caching: bool = True,
    ):
        """初始化LLM路由器

        Args:
            api_call_func: 自定义API调用函数(prompt, model) -> response
            enable_fallback: 是否启用降级机制
            enable_caching: 是否启用结果缓存
        """
        self._api_call_func = api_call_func or self._default_api_call
        self._enable_fallback = enable_fallback
        self._enable_caching = enable_caching

        # 初始化各模型信号量
        self._semaphores: dict[LLMModel, asyncio.Semaphore] = {
            model: asyncio.Semaphore(MODEL_CONCURRENCY[model])
            for model in LLMModel
        }

        # 指数退避配置
        self._backoff = ExponentialBackoff()

        # 结果缓存
        self._cache: dict[str, str] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # 统计信息
        self._stats = RouterStats()

    async def route(self, request: TaskRequest) -> LLMResponse:
        """主路由方法

        路由流程:
        1. 检查缓存
        2. 根据task_type和context_length选择模型
        3. 获取模型信号量
        4. 执行调用（带重试和降级）

        Args:
            request: 任务请求

        Returns:
            LLM响应结果
        """
        # 检查缓存
        if self._enable_caching:
            cached = self._check_cache(request)
            if cached:
                return cached

        # 更新统计
        self._stats.total_requests += 1
        self._update_task_distribution(request.task_type)

        # 选择目标模型
        target_model = self._select_model(request)

        # 执行调用
        return await self._execute_with_fallback(request, target_model)

    def _select_model(self, request: TaskRequest) -> LLMModel:
        """基于专长选择模型

        选择优先级:
        1. 长文本(>30K)强制路由Kimi
        2. 任务类型匹配专长
        3. 默认MINIMAX_M27

        Args:
            request: 任务请求

        Returns:
            选定的模型
        """
        # 优先级1: 长文本强制路由Kimi
        if request.context_length > LONG_CONTEXT_THRESHOLD:
            return LLMModel.KIMI_K25

        # 优先级2: 任务类型匹配专长
        task_type = request.task_type

        for model, capability in MODEL_CAPABILITIES.items():
            if task_type in capability.specialties:
                return model

        # 优先级3: 默认MINIMAX_M27
        return LLMModel.MINIMAX_M27

    async def _execute_with_fallback(
        self,
        request: TaskRequest,
        primary_model: LLMModel,
    ) -> LLMResponse:
        """执行调用，失败时降级

        Args:
            request: 任务请求
            primary_model: 首选模型

        Returns:
            LLM响应
        """
        models_to_try = [primary_model]

        # 添加降级路径
        if self._enable_fallback:
            fallback = FALLBACK_CHAIN.get(primary_model)
            if fallback:
                models_to_try.append(fallback)

        last_error = ""

        for model in models_to_try:
            response = await self._execute_with_retry(request, model)

            if response.success:
                # 更新模型分布统计
                self._update_model_distribution(model)
                if model != primary_model:
                    self._stats.fallback_requests += 1
                return response

            last_error = response.error_message

        # 所有模型都失败
        return LLMResponse(
            task_id=request.task_id,
            success=False,
            model_used=primary_model,
            error_message=f"All models failed. Last error: {last_error}",
            latency_ms=0.0,
            fallback_used=len(models_to_try) > 1,
        )

    async def _execute_with_retry(
        self,
        request: TaskRequest,
        model: LLMModel,
    ) -> LLMResponse:
        """带重试的调用执行

        Args:
            request: 任务请求
            model: 目标模型

        Returns:
            LLM响应
        """
        provider = MODEL_CAPABILITIES[model].provider
        semaphore = self._semaphores[model]

        start_time = time.time()
        last_error = ""

        async with semaphore:
            for attempt in range(self._backoff.max_attempts + 1):
                try:
                    # 执行API调用
                    response_text = await asyncio.wait_for(
                        self._api_call_func(request.prompt, model),
                        timeout=request.timeout,
                    )

                    # 成功
                    latency_ms = (time.time() - start_time) * 1000
                    self._stats.successful_requests += 1
                    self._stats.total_latency_ms += latency_ms

                    # 缓存结果
                    if self._enable_caching:
                        self._cache_result(request, response_text)

                    return LLMResponse(
                        task_id=request.task_id,
                        success=True,
                        model_used=model,
                        content=response_text,
                        latency_ms=latency_ms,
                        retries=attempt,
                        fallback_used=False,
                    )

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {request.timeout}s"
                    if attempt < self._backoff.max_attempts:
                        delay = self._backoff.compute_delay(attempt)
                        await asyncio.sleep(delay)

                except Exception as e:
                    last_error = str(e)
                    if attempt < self._backoff.max_attempts:
                        delay = self._backoff.compute_delay(attempt)
                        await asyncio.sleep(delay)

        # 所有重试失败
        self._stats.failed_requests += 1
        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            task_id=request.task_id,
            success=False,
            model_used=model,
            error_message=last_error,
            latency_ms=latency_ms,
            retries=self._backoff.max_attempts,
        )

    async def route_batch(
        self,
        requests: list[TaskRequest],
    ) -> list[LLMResponse]:
        """批量路由请求

        Args:
            requests: 任务请求列表

        Returns:
            响应结果列表(与请求顺序一致)
        """
        tasks = [self.route(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _check_cache(self, request: TaskRequest) -> Optional[LLMResponse]:
        """检查缓存

        Args:
            request: 任务请求

        Returns:
            缓存的响应或None
        """
        cache_key = self._generate_cache_key(request)

        if cache_key in self._cache:
            self._cache_hits += 1
            return LLMResponse(
                task_id=request.task_id,
                success=True,
                model_used=LLMModel.MINIMAX_M27,  # 缓存不记录模型
                content=self._cache[cache_key],
                latency_ms=0.0,
                retries=0,
                fallback_used=False,
            )

        self._cache_misses += 1
        return None

    def _cache_result(self, request: TaskRequest, response: str) -> None:
        """缓存结果

        Args:
            request: 任务请求
            response: 响应内容
        """
        cache_key = self._generate_cache_key(request)
        self._cache[cache_key] = response

    def _generate_cache_key(self, request: TaskRequest) -> str:
        """生成缓存键

        Args:
            request: 任务请求

        Returns:
            缓存键
        """
        content = f"{request.task_type}:{request.prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _update_model_distribution(self, model: LLMModel) -> None:
        """更新模型使用分布统计

        Args:
            model: 使用的模型
        """
        if model not in self._stats.model_distribution:
            self._stats.model_distribution[model] = 0
        self._stats.model_distribution[model] += 1

    def _update_task_distribution(self, task_type: TaskType) -> None:
        """更新任务类型分布统计

        Args:
            task_type: 任务类型
        """
        if task_type not in self._stats.task_type_distribution:
            self._stats.task_type_distribution[task_type] = 0
        self._stats.task_type_distribution[task_type] += 1

    def get_stats(self) -> RouterStats:
        """获取路由统计信息

        Returns:
            统计信息
        """
        return self._stats

    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计

        Returns:
            缓存统计信息
        """
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def _default_api_call(
        self,
        prompt: str,
        model: LLMModel,
    ) -> str:
        """默认API调用函数

        当未提供自定义API调用函数时使用。
        从环境变量读取API密钥并调用对应模型。

        Args:
            prompt: 提示词
            model: 模型

        Returns:
            响应文本
        """
        # 获取模型配置
        capability = MODEL_CAPABILITIES[model]

        # 检查API密钥（仅记录警告，不阻止模拟）
        api_key = os.getenv(capability.api_key_env)
        if not api_key:
            # 模拟模式下，生成模拟响应
            await asyncio.sleep(0.05)  # 模拟50ms网络延迟
            return (
                f"[Mock Response from {model.value}] "
                f"Task completed successfully. Prompt: {prompt[:30]}..."
            )

        # TODO: 实现真实的API调用
        # 实际使用时，根据provider选择对应的SDK调用
        await asyncio.sleep(0.05)  # 模拟延迟
        return f"[Response from {model.value}] {prompt[:50]}..."


# =============================================================================
# Test and Simulation
# =============================================================================


async def simulate_100_concurrent_requests() -> dict[str, Any]:
    """模拟100个混合并发请求测试

    Returns:
        测试结果统计
    """
    print("=" * 60)
    print("LLM Router - 100 Concurrent Mixed Requests Simulation")
    print("=" * 60)

    # 创建路由器
    router = LLMRouter(enable_fallback=True, enable_caching=True)

    # 构造混合任务请求
    requests: list[TaskRequest] = []

    # 1. 贝叶斯立场更新 (DeepSeek) - 10个
    for i in range(10):
        requests.append(
            TaskRequest(
                task_id=f"bayesian_{i}",
                task_type=TaskType.BAYESIAN_STANCE_UPDATE,
                agent_id=f"agent_{i}",
                agent_importance=0.9,
                prompt=f"计算贝叶斯立场更新 Delta = (S_m - S_i) * T_ij * I_m, agent {i}",
                priority=10,
            )
        )

    # 2. 认知失调处理 (DeepSeek) - 10个
    for i in range(10):
        requests.append(
            TaskRequest(
                task_id=f"dissonance_{i}",
                task_type=TaskType.COGNITIVE_DISSONANCE,
                agent_id=f"agent_{i + 10}",
                agent_importance=0.8,
                prompt=f"处理认知失调，重新平衡立场，agent {i + 10}",
                priority=9,
            )
        )

    # 3. 记忆归并 (Kimi) - 15个
    for i in range(15):
        requests.append(
            TaskRequest(
                task_id=f"memory_{i}",
                task_type=TaskType.MEMORY_CONSOLIDATION,
                agent_id=f"agent_{i + 20}",
                agent_importance=0.6,
                prompt=f"归并短期记忆至长期记忆，相似度聚类，agent {i + 20}",
                priority=7,
            )
        )

    # 4. 超长文本解析 (Kimi) - 10个
    long_context = "x" * 40000  # 40K字符模拟长文本
    for i in range(10):
        requests.append(
            TaskRequest(
                task_id=f"long_ctx_{i}",
                task_type=TaskType.LONG_CONTEXT_PARSE,
                agent_id=f"agent_{i + 35}",
                agent_importance=0.7,
                prompt=f"解析超长新闻事件: {long_context[:100]}...",
                context_length=40000,
                priority=8,
            )
        )

    # 5. 规则校验 (GLM) - 15个
    for i in range(15):
        requests.append(
            TaskRequest(
                task_id=f"rule_{i}",
                task_type=TaskType.RULE_VALIDATION,
                agent_id=f"system",
                agent_importance=1.0,
                prompt=f"校验Agent行为是否违反物理约束和社会规则 {i}",
                priority=10,
                require_validation=True,
            )
        )

    # 6. 高频交互 (Qwen) - 20个
    for i in range(20):
        requests.append(
            TaskRequest(
                task_id=f"interact_{i}",
                task_type=TaskType.HIGH_FREQ_INTERACT,
                agent_id=f"agent_{i + 50}",
                agent_importance=0.3,
                prompt=f"浅层社交互动: 点赞、评论、简单回应 {i}",
                priority=3,
            )
        )

    # 7. 信息传播 (Qwen) - 10个
    for i in range(10):
        requests.append(
            TaskRequest(
                task_id=f"propagate_{i}",
                task_type=TaskType.INFO_PROPAGATION,
                agent_id=f"agent_{i + 70}",
                agent_importance=0.4,
                prompt=f"信息扭曲与传播: 转发新闻并添加认知偏差 {i}",
                priority=4,
            )
        )

    # 8. 默认任务 (MiniMax) - 10个
    for i in range(10):
        requests.append(
            TaskRequest(
                task_id=f"default_{i}",
                task_type=TaskType.DEFAULT,
                agent_id=f"agent_{i + 80}",
                agent_importance=0.5,
                prompt=f"复杂Agent团队协作与环境感知决策 {i}",
                priority=5,
            )
        )

    print(f"\nTotal requests: {len(requests)}")
    print("\nTask distribution:")
    task_counts: dict[str, int] = {}
    for req in requests:
        task_type = req.task_type.value
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    # 执行并发请求
    print("\nExecuting 100 concurrent requests...")
    start_time = time.time()

    responses = await router.route_batch(requests)

    elapsed_time = (time.time() - start_time) * 1000

    # 统计结果
    success_count = sum(1 for r in responses if isinstance(r, LLMResponse) and r.success)
    failure_count = sum(1 for r in responses if isinstance(r, LLMResponse) and not r.success)
    fallback_count = sum(1 for r in responses if isinstance(r, LLMResponse) and r.fallback_used)

    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    print(f"Total time: {elapsed_time:.2f} ms")
    print(f"Average latency per request: {elapsed_time / len(requests):.2f} ms")
    print(f"Successful: {success_count}/{len(requests)}")
    print(f"Failed: {failure_count}/{len(requests)}")
    print(f"Fallback used: {fallback_count}/{len(requests)}")

    # 模型使用分布
    stats = router.get_stats()
    print(f"\nModel distribution:")
    for model, count in sorted(stats.model_distribution.items(), key=lambda x: -x[1]):
        print(f"  {model.value}: {count}")

    # 任务类型分布
    print(f"\nTask type distribution:")
    for task, count in sorted(stats.task_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  {task.value}: {count}")

    # 缓存统计
    cache_stats = router.get_cache_stats()
    print(f"\nCache stats:")
    print(f"  Hits: {cache_stats['cache_hits']}")
    print(f"  Misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")

    return {
        "total_requests": len(requests),
        "successful": success_count,
        "failed": failure_count,
        "fallback_used": fallback_count,
        "total_time_ms": elapsed_time,
        "avg_latency_ms": elapsed_time / len(requests),
        "model_distribution": stats.model_distribution,
        "task_distribution": stats.task_type_distribution,
        "cache_stats": cache_stats,
    }


if __name__ == "__main__":
    # 运行100并发测试
    result = asyncio.run(simulate_100_concurrent_requests())

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print(f"{'=' * 60}")
