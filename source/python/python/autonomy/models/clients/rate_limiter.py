"""
Adaptive Rate Limiter using AIMD (Additive Increase, Multiplicative Decrease) algorithm.

This module provides client-side rate limiting that automatically adapts to backend
capacity without requiring centralized coordination. Multiple clients will naturally
converge to a fair share of available capacity.

Key features:
- Token bucket for smooth rate limiting
- AIMD for adaptive rate adjustment
- Async-first design with semaphore-based concurrency control
- Circuit breaker state awareness (from gateway hints)
- Metrics exposure for monitoring
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from autonomy.logs import get_logger


@dataclass
class RateLimiterStats:
  """Statistics for rate limiter monitoring."""

  current_rpm: float
  min_rpm: float
  max_rpm: float
  tokens_available: float
  total_requests: int
  successful_requests: int
  rate_limited_requests: int
  circuit_open_events: int
  last_rate_limit_time: Optional[float]
  last_circuit_state: Optional[str]


@dataclass
class RateLimiterConfig:
  """Configuration for AdaptiveRateLimiter."""

  # Initial requests per minute
  initial_rpm: float = 60.0

  # Minimum RPM (floor to prevent starvation)
  min_rpm: float = 1.0

  # Maximum RPM (ceiling to prevent overwhelming backend)
  max_rpm: float = 1000.0

  # Additive increase: tokens per second added on success
  # Default: add 1 RPM worth of capacity per minute of successful requests
  additive_increase: float = 1.0 / 60.0

  # Multiplicative decrease factor on rate limit (0 < beta < 1)
  # Default: cut rate in half on rate limit
  multiplicative_decrease: float = 0.5

  # Maximum concurrent requests
  max_concurrent: int = 10

  # Cooldown period after rate limit before increasing again (seconds)
  cooldown_period: float = 5.0

  # Half-open decrease factor (less aggressive than full circuit open)
  half_open_decrease: float = 0.7


class AdaptiveRateLimiter:
  """
  Adaptive rate limiter using AIMD algorithm.

  This rate limiter automatically adjusts its rate based on success/failure signals:
  - On success: additively increase the rate (slow growth)
  - On rate limit: multiplicatively decrease the rate (fast reduction)

  This creates a sawtooth pattern that converges to the maximum sustainable rate.
  When multiple clients use AIMD, they naturally share capacity fairly.

  Usage:
      limiter = AdaptiveRateLimiter(model="claude-sonnet-4")

      # Before making a request
      await limiter.acquire()

      try:
          response = await make_request()
          limiter.record_success()
      except RateLimitError:
          limiter.record_rate_limit()
  """

  def __init__(
    self,
    model: str,
    config: Optional[RateLimiterConfig] = None,
  ):
    """
    Initialize the adaptive rate limiter.

    :param model: Model name (for logging and metrics)
    :param config: Rate limiter configuration
    """
    self.model = model
    self.config = config or RateLimiterConfig()
    self.logger = get_logger("rate_limiter")

    # Current rate (tokens per second)
    self._current_rate = self.config.initial_rpm / 60.0

    # Token bucket state
    # Start with at least 1 token so first request can proceed immediately
    self._tokens = max(1.0, self._current_rate)
    self._last_refill = time.monotonic()

    # Concurrency control
    self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    # Cooldown tracking
    self._last_rate_limit_time: Optional[float] = None

    # Statistics
    self._total_requests = 0
    self._successful_requests = 0
    self._rate_limited_requests = 0
    self._circuit_open_events = 0
    self._last_circuit_state: Optional[str] = None

    # Lock for thread-safe updates
    self._lock = asyncio.Lock()

    self.logger.debug(
      f"Initialized rate limiter for {model}: "
      f"initial_rpm={self.config.initial_rpm}, "
      f"max_concurrent={self.config.max_concurrent}"
    )

  async def acquire(self, timeout: Optional[float] = None) -> bool:
    """
    Acquire permission to make a request.

    This method:
    1. Waits for a concurrency slot (semaphore)
    2. Waits for a rate limit token (token bucket)

    :param timeout: Maximum time to wait (seconds), None for no timeout
    :return: True if acquired, False if timed out
    :raises asyncio.TimeoutError: If timeout is specified and exceeded
    """
    start_time = time.monotonic()

    # First, acquire concurrency slot
    try:
      if timeout is not None:
        await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
      else:
        await self._semaphore.acquire()
    except asyncio.TimeoutError:
      self.logger.debug(f"Timeout waiting for concurrency slot for {self.model}")
      raise

    # Calculate remaining timeout
    elapsed = time.monotonic() - start_time
    remaining_timeout = None if timeout is None else max(0, timeout - elapsed)

    # Then, wait for rate limit token
    try:
      await self._acquire_token(timeout=remaining_timeout)
    except asyncio.TimeoutError:
      # Release semaphore if we couldn't get a token
      self._semaphore.release()
      self.logger.debug(f"Timeout waiting for rate limit token for {self.model}")
      raise

    self._total_requests += 1
    return True

  async def _acquire_token(self, timeout: Optional[float] = None) -> None:
    """
    Wait for a rate limit token from the token bucket.

    :param timeout: Maximum time to wait
    :raises asyncio.TimeoutError: If timeout exceeded
    """
    start_time = time.monotonic()

    while True:
      async with self._lock:
        self._refill_tokens()

        if self._tokens >= 1.0:
          self._tokens -= 1.0
          return

        # Calculate wait time for next token
        tokens_needed = 1.0 - self._tokens
        wait_time = tokens_needed / self._current_rate

      # Check timeout
      if timeout is not None:
        elapsed = time.monotonic() - start_time
        if elapsed + wait_time > timeout:
          raise asyncio.TimeoutError()

      # Wait for tokens to refill
      await asyncio.sleep(min(wait_time, 0.1))  # Cap at 100ms for responsiveness

  def _refill_tokens(self) -> None:
    """Refill tokens based on elapsed time."""
    now = time.monotonic()
    elapsed = now - self._last_refill
    self._last_refill = now

    # Add tokens based on current rate
    tokens_to_add = elapsed * self._current_rate

    # Cap at bucket size (1 second of tokens, but at least 1 token)
    max_tokens = max(1.0, self._current_rate)
    self._tokens = min(self._tokens + tokens_to_add, max_tokens)

  def release(self) -> None:
    """
    Release the concurrency slot.

    Call this after the request completes (success or failure).
    """
    self._semaphore.release()

  def record_success(
    self,
    hint_rpm: Optional[float] = None,
    circuit_state: Optional[str] = None,
  ) -> None:
    """
    Record a successful request and potentially increase rate.

    :param hint_rpm: Optional hint from gateway about recommended RPM
    :param circuit_state: Optional circuit breaker state from gateway (closed, half-open, open)
    """
    self._successful_requests += 1
    self._last_circuit_state = circuit_state

    # If circuit is half-open, slow down proactively even on success
    if circuit_state == "half-open":
      self._apply_half_open_slowdown()
      return

    # If circuit is open, apply aggressive slowdown
    if circuit_state == "open":
      self.record_circuit_open()
      return

    # Check if we're still in cooldown
    now = time.monotonic()
    if self._last_rate_limit_time is not None:
      time_since_limit = now - self._last_rate_limit_time
      if time_since_limit < self.config.cooldown_period:
        return  # Still in cooldown, don't increase

    # Use hint if provided and it's higher than current rate
    if hint_rpm is not None:
      hint_rate = hint_rpm / 60.0
      if hint_rate > self._current_rate:
        # Jump to hint rate (capped by max)
        old_rpm = self._current_rate * 60.0
        self._current_rate = min(hint_rate, self.config.max_rpm / 60.0)
        new_rpm = self._current_rate * 60.0
        self.logger.debug(f"Rate limiter for {self.model}: jumped to hint {old_rpm:.1f} -> {new_rpm:.1f} RPM")
        return

    # Additive increase
    old_rate = self._current_rate
    self._current_rate = min(
      self._current_rate + self.config.additive_increase,
      self.config.max_rpm / 60.0,
    )

    if self._current_rate > old_rate:
      self.logger.debug(
        f"Rate limiter for {self.model}: increased {old_rate * 60:.1f} -> {self._current_rate * 60:.1f} RPM"
      )

  def record_rate_limit(self, retry_after: Optional[float] = None) -> None:
    """
    Record a rate limit response and decrease rate.

    :param retry_after: Optional Retry-After value from response (seconds)
    """
    self._rate_limited_requests += 1
    self._last_rate_limit_time = time.monotonic()

    # Multiplicative decrease
    old_rate = self._current_rate
    self._current_rate = max(
      self._current_rate * self.config.multiplicative_decrease,
      self.config.min_rpm / 60.0,
    )

    self.logger.info(
      f"Rate limiter for {self.model}: decreased {old_rate * 60:.1f} -> {self._current_rate * 60:.1f} RPM"
      + (f" (retry_after={retry_after}s)" if retry_after else "")
    )

  def record_circuit_open(self) -> None:
    """
    Record that the circuit breaker is open (provider is down).

    This triggers an aggressive rate reduction to give the provider time to recover.
    """
    self._last_rate_limit_time = time.monotonic()
    self._circuit_open_events += 1
    self._last_circuit_state = "open"

    # More aggressive decrease for circuit open
    old_rate = self._current_rate
    self._current_rate = max(
      self._current_rate * 0.25,  # Cut to 25%
      self.config.min_rpm / 60.0,
    )

    self.logger.warning(
      f"Rate limiter for {self.model}: circuit open, decreased {old_rate * 60:.1f} -> {self._current_rate * 60:.1f} RPM"
    )

  def _apply_half_open_slowdown(self) -> None:
    """
    Apply a moderate slowdown when circuit is half-open.

    This is less aggressive than full circuit open - we reduce rate but
    don't trigger the full cooldown period.
    """
    old_rate = self._current_rate
    self._current_rate = max(
      self._current_rate * self.config.half_open_decrease,
      self.config.min_rpm / 60.0,
    )

    if self._current_rate < old_rate:
      self.logger.info(
        f"Rate limiter for {self.model}: circuit half-open, "
        f"decreased {old_rate * 60:.1f} -> {self._current_rate * 60:.1f} RPM"
      )

  def record_circuit_exhausted(self) -> None:
    """
    Record that all providers are exhausted.

    This is the most severe state - all providers are unavailable.
    Apply very aggressive backoff.
    """
    self._last_rate_limit_time = time.monotonic()
    self._circuit_open_events += 1
    self._last_circuit_state = "exhausted"

    # Very aggressive decrease for exhausted state
    old_rate = self._current_rate
    self._current_rate = self.config.min_rpm / 60.0  # Drop to minimum

    self.logger.warning(
      f"Rate limiter for {self.model}: all providers exhausted, "
      f"decreased {old_rate * 60:.1f} -> {self._current_rate * 60:.1f} RPM"
    )

  @property
  def current_rpm(self) -> float:
    """Get current rate in requests per minute."""
    return self._current_rate * 60.0

  @property
  def stats(self) -> RateLimiterStats:
    """Get current statistics."""
    return RateLimiterStats(
      current_rpm=self._current_rate * 60.0,
      min_rpm=self.config.min_rpm,
      max_rpm=self.config.max_rpm,
      tokens_available=self._tokens,
      total_requests=self._total_requests,
      successful_requests=self._successful_requests,
      rate_limited_requests=self._rate_limited_requests,
      circuit_open_events=self._circuit_open_events,
      last_rate_limit_time=self._last_rate_limit_time,
      last_circuit_state=self._last_circuit_state,
    )


class RateLimiterManager:
  """
  Manages rate limiters for multiple models.

  This is a singleton that provides per-model rate limiters with lazy initialization.
  """

  _instance: Optional["RateLimiterManager"] = None
  _lock: asyncio.Lock = asyncio.Lock()

  def __init__(self):
    self._limiters: dict[str, AdaptiveRateLimiter] = {}
    self._configs: dict[str, RateLimiterConfig] = {}
    self._default_config = RateLimiterConfig()
    self.logger = get_logger("rate_limiter")

  @classmethod
  async def get_instance(cls) -> "RateLimiterManager":
    """Get the singleton instance."""
    if cls._instance is None:
      async with cls._lock:
        if cls._instance is None:
          cls._instance = RateLimiterManager()
    return cls._instance

  def configure_model(self, model: str, config: RateLimiterConfig) -> None:
    """
    Configure rate limiting for a specific model.

    :param model: Model name
    :param config: Rate limiter configuration
    """
    self._configs[model] = config
    # Update existing limiter if present
    if model in self._limiters:
      self._limiters[model] = AdaptiveRateLimiter(model, config)
      self.logger.info(f"Reconfigured rate limiter for {model}")

  def set_default_config(self, config: RateLimiterConfig) -> None:
    """
    Set the default configuration for new models.

    :param config: Default rate limiter configuration
    """
    self._default_config = config

  def get_limiter(self, model: str) -> AdaptiveRateLimiter:
    """
    Get or create a rate limiter for a model.

    :param model: Model name
    :return: Rate limiter for the model
    """
    if model not in self._limiters:
      config = self._configs.get(model, self._default_config)
      self._limiters[model] = AdaptiveRateLimiter(model, config)
      self.logger.debug(f"Created rate limiter for {model}")
    return self._limiters[model]

  def get_all_stats(self) -> dict[str, RateLimiterStats]:
    """Get statistics for all models."""
    return {model: limiter.stats for model, limiter in self._limiters.items()}

  def reset(self) -> None:
    """Reset all rate limiters (for testing)."""
    self._limiters.clear()
    self._configs.clear()
    self._default_config = RateLimiterConfig()
