"""Async LLM scheduler with semaphore limiting and exponential backoff.

Manages concurrent LLM API calls with:
- Semaphore-based concurrency limiting (prevent rate limits)
- Exponential backoff for retries (handle transient failures)
- Priority queue for important requests
- Timeout handling
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from utopia.core.pydantic_models import AsyncLLMCall


class CallStatus(Enum):
    """Status of an LLM call."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExponentialBackoff:
    """Exponential backoff configuration.

    Delay = base_delay * (2 ^ attempt) + jitter
    """
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1
    max_attempts: int = 3

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for given attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        # Add small jitter to prevent thundering herd
        import random
        delay += random.uniform(0, self.jitter)
        return delay


@dataclass
class LLMResult:
    """Result of an LLM call.

    Attributes:
        call_id: Unique call identifier
        success: Whether call succeeded
        content: Response content (if success)
        error: Error message (if failed)
        latency_ms: Call latency in milliseconds
        tokens_used: Token usage (if available)
        retries: Number of retries attempted
    """
    call_id: str
    success: bool
    content: str = ""
    error: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    retries: int = 0


class AsyncLLMScheduler:
    """Scheduler for async LLM API calls with rate limiting.

    Features:
    - Semaphore limits concurrent calls (prevents rate limiting)
    - Exponential backoff for retries
    - Priority queue for important requests
    - Timeout handling
    - Result caching

    Usage:
        scheduler = AsyncLLMScheduler(max_concurrent=5)
        result = await scheduler.call("What is the meaning of life?")
    """

    def __init__(
        self,
        llm_call_func: Optional[Callable[[str], Coroutine[Any, Any, str]]] = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        enable_caching: bool = True,
    ):
        """Initialize async LLM scheduler.

        Args:
            llm_call_func: Async function to call LLM (prompt -> response)
            max_concurrent: Maximum concurrent calls
            max_retries: Maximum retry attempts
            timeout_seconds: Call timeout
            enable_caching: Enable result caching
        """
        self.llm_call_func = llm_call_func
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_caching = enable_caching

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._backoff = ExponentialBackoff(max_attempts=max_retries)

        # Request tracking
        self._pending_calls: dict[str, AsyncLLMCall] = {}
        self._completed_calls: dict[str, LLMResult] = {}
        self._in_progress: set[str] = set()

        # Cache
        self._cache: dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Statistics
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_retries": 0,
            "total_latency_ms": 0.0,
        }

    async def call(
        self,
        prompt: str,
        agent_id: str = "",
        priority: int = 0,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        metadata: Optional[dict] = None,
    ) -> LLMResult:
        """Make async LLM call with rate limiting and retries.

        Args:
            prompt: LLM prompt
            agent_id: Requesting agent
            priority: Call priority (higher = more important)
            timeout: Override default timeout
            use_cache: Whether to use cache
            metadata: Additional metadata

        Returns:
            LLM result
        """
        call_id = self._generate_call_id(prompt, agent_id)
        timeout = timeout or self.timeout_seconds

        # Check cache
        if use_cache and self.enable_caching:
            cache_key = self._cache_key(prompt)
            if cache_key in self._cache:
                self._cache_hits += 1
                return LLMResult(
                    call_id=call_id,
                    success=True,
                    content=self._cache[cache_key],
                    latency_ms=0.0,
                )
            self._cache_misses += 1

        # Create call record
        call = AsyncLLMCall(
            call_id=call_id,
            prompt=prompt,
            agent_id=agent_id,
            tick=0,  # Will be set by caller
            priority=priority,
        )
        self._pending_calls[call_id] = call

        # Execute with semaphore
        async with self._semaphore:
            return await self._execute_call(call, timeout, use_cache)

    async def call_batch(
        self,
        prompts: list[tuple[str, str, int]],  # (prompt, agent_id, priority)
        timeout: Optional[float] = None,
    ) -> list[LLMResult]:
        """Execute multiple LLM calls concurrently.

        Args:
            prompts: List of (prompt, agent_id, priority) tuples
            timeout: Per-call timeout

        Returns:
            List of results in same order
        """
        tasks = [
            self.call(prompt, agent_id, priority, timeout)
            for prompt, agent_id, priority in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_call(
        self,
        call: AsyncLLMCall,
        timeout: float,
        use_cache: bool,
    ) -> LLMResult:
        """Execute single LLM call with retries.

        Args:
            call: Call to execute
            timeout: Timeout seconds
            use_cache: Whether to cache result

        Returns:
            LLM result
        """
        if not self.llm_call_func:
            return LLMResult(
                call_id=call.call_id,
                success=False,
                error="No LLM call function configured",
            )

        call.status = "in_progress"
        self._in_progress.add(call.call_id)

        start_time = time.time()
        last_error = ""

        for attempt in range(self.max_retries + 1):
            try:
                call.attempt_count = attempt
                call.last_attempt_time = datetime.now()

                # Execute with timeout
                response = await asyncio.wait_for(
                    self.llm_call_func(call.prompt),
                    timeout=timeout,
                )

                # Success
                latency_ms = (time.time() - start_time) * 1000

                # Cache result
                if use_cache and self.enable_caching:
                    cache_key = self._cache_key(call.prompt)
                    self._cache[cache_key] = response

                # Update stats
                self._stats["total_calls"] += 1
                self._stats["successful_calls"] += 1
                self._stats["total_retries"] += attempt
                self._stats["total_latency_ms"] += latency_ms

                # Update call record
                call.status = "completed"
                call.result = response
                self._completed_calls[call.call_id] = result
                self._in_progress.discard(call.call_id)

                result = LLMResult(
                    call_id=call.call_id,
                    success=True,
                    content=response,
                    latency_ms=latency_ms,
                    retries=attempt,
                )
                self._completed_calls[call.call_id] = result
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                if attempt < self.max_retries:
                    delay = self._backoff.compute_delay(attempt)
                    await asyncio.sleep(delay)

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    delay = self._backoff.compute_delay(attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        latency_ms = (time.time() - start_time) * 1000

        self._stats["total_calls"] += 1
        self._stats["failed_calls"] += 1
        self._stats["total_retries"] += self.max_retries

        call.status = "failed"
        call.error_message = last_error
        self._in_progress.discard(call.call_id)

        result = LLMResult(
            call_id=call.call_id,
            success=False,
            error=last_error,
            latency_ms=latency_ms,
            retries=self.max_retries,
        )
        self._completed_calls[call.call_id] = result
        return result

    def _generate_call_id(self, prompt: str, agent_id: str) -> str:
        """Generate unique call ID.

        Args:
            prompt: Prompt text
            agent_id: Agent ID

        Returns:
            Unique ID string
        """
        content = f"{agent_id}:{prompt}:{time.time()}"
        return f"LLM_{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt.

        Args:
            prompt: Prompt text

        Returns:
            Cache key
        """
        return hashlib.md5(prompt.encode()).hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Statistics dictionary
        """
        total = self._stats["total_calls"]
        success = self._stats["successful_calls"]
        avg_latency = (
            self._stats["total_latency_ms"] / total
            if total > 0 else 0.0
        )

        return {
            "total_calls": total,
            "successful_calls": success,
            "failed_calls": self._stats["failed_calls"],
            "success_rate": success / total if total > 0 else 0.0,
            "average_latency_ms": avg_latency,
            "total_retries": self._stats["total_retries"],
            "pending_calls": len(self._pending_calls),
            "in_progress": len(self._in_progress),
            "completed_calls": len(self._completed_calls),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear result cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def wait_for_pending(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending calls to complete.

        Args:
            timeout: Maximum wait time
        """
        if not self._in_progress:
            return

        start = time.time()
        while self._in_progress:
            if timeout and (time.time() - start) > timeout:
                break
            await asyncio.sleep(0.1)


from datetime import datetime


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Prevents exceeding rate limits by controlling request velocity.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._tokens = max_requests
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token if available.

        Returns:
            True if token acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update

            # Replenish tokens
            self._tokens = min(
                self.max_requests,
                self._tokens + elapsed * (self.max_requests / self.window_seconds)
            )
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    async def wait_for_token(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)
