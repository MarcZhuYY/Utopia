"""Shared utility functions for Utopia.

Common utilities used across multiple modules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ExponentialBackoff:
    """Exponential backoff configuration for retries.

    Calculates delay as: base_delay * (2 ^ attempt) + jitter

    Attributes:
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: Random jitter range (0 to jitter seconds)
        max_attempts: Maximum retry attempts
    """

    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1
    max_attempts: int = 3

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for a given attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        # Add jitter to prevent thundering herd
        delay += random.uniform(0, self.jitter)
        return delay


def sanitize_float(value: float, default: float = 0.0) -> float:
    """Sanitize float value by replacing NaN and Inf with default.

    Args:
        value: Float value to sanitize
        default: Default value to use for NaN/Inf

    Returns:
        Sanitized float value
    """
    import math

    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return value


__all__ = [
    "ExponentialBackoff",
    "sanitize_float",
]
