"""
Rate limiting utilities for API requests.
"""
import time
import asyncio
from typing import Callable, Any
from functools import wraps
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter implementation.

    Supports both synchronous and asynchronous rate limiting.
    """

    def __init__(self, requests_per_second: float, burst_size: int = None):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (defaults to requests_per_second)
        """
        self.rate = requests_per_second
        self.burst_size = burst_size or int(requests_per_second)
        self.tokens = self.burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            float: Time waited in seconds
        """
        self._refill_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return 0.0

        # Calculate wait time
        deficit = tokens - self.tokens
        wait_time = deficit / self.rate

        time.sleep(wait_time)
        self._refill_tokens()
        self.tokens -= tokens

        return wait_time

    async def acquire_async(self, tokens: int = 1) -> float:
        """
        Acquire tokens asynchronously.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            float: Time waited in seconds
        """
        async with self.lock:
            self._refill_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            deficit = tokens - self.tokens
            wait_time = deficit / self.rate

            await asyncio.sleep(wait_time)
            self._refill_tokens()
            self.tokens -= tokens

            return wait_time


class ExponentialBackoff:
    """
    Exponential backoff implementation for retry logic.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize exponential backoff.

        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation
            jitter: Whether to add random jitter
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.attempt = 0

    def get_delay(self) -> float:
        """
        Get delay for current attempt.

        Returns:
            float: Delay in seconds
        """
        import random

        delay = min(
            self.max_delay,
            self.base_delay * (self.exponential_base ** self.attempt)
        )

        if self.jitter:
            # Add random jitter (0-50% of delay)
            delay = delay * (0.5 + random.random() * 0.5)

        self.attempt += 1
        return delay

    def reset(self):
        """Reset attempt counter."""
        self.attempt = 0


def rate_limit(requests_per_second: float):
    """
    Decorator for rate limiting function calls.

    Args:
        requests_per_second: Maximum requests per second

    Example:
        @rate_limit(3)
        def fetch_data():
            return requests.get(url)
    """
    limiter = RateLimiter(requests_per_second)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = limiter.acquire()
            if wait_time > 0:
                logger.debug(f"Rate limited: waited {wait_time:.2f}s")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def async_rate_limit(requests_per_second: float):
    """
    Decorator for rate limiting async function calls.

    Args:
        requests_per_second: Maximum requests per second

    Example:
        @async_rate_limit(3)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    """
    limiter = RateLimiter(requests_per_second)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            wait_time = await limiter.acquire_async()
            if wait_time > 0:
                logger.debug(f"Rate limited: waited {wait_time:.2f}s")
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    base_delay: float = 1.0
):
    """
    Decorator for retrying with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch
        base_delay: Initial delay in seconds

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            backoff = ExponentialBackoff(base_delay=base_delay)

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded")
                        raise

                    delay = backoff.get_delay()
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    base_delay: float = 1.0
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch
        base_delay: Initial delay in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            backoff = ExponentialBackoff(base_delay=base_delay)

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded")
                        raise

                    delay = backoff.get_delay()
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)

        return wrapper
    return decorator


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for more precise rate control.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize sliding window rate limiter.

        Args:
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    def is_allowed(self) -> bool:
        """
        Check if request is allowed.

        Returns:
            bool: True if request can proceed
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        # Check if we can add new request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def wait_time(self) -> float:
        """
        Calculate time to wait before next request.

        Returns:
            float: Seconds to wait
        """
        if not self.requests:
            return 0.0

        oldest = self.requests[0]
        cutoff = oldest + timedelta(seconds=self.window_seconds)
        now = datetime.now()

        if now >= cutoff:
            return 0.0

        return (cutoff - now).total_seconds()
