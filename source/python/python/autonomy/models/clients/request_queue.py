"""
Model Request Queue for rate-limited request handling.

This module provides a queue-based request system that integrates with
the AdaptiveRateLimiter to handle concurrent requests with backpressure.

Key features:
- Per-model request queuing
- Priority support for urgent requests
- Backpressure handling when queue is full
- Integration with AdaptiveRateLimiter
- Retry logic with exponential backoff
- Graceful shutdown support
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

from autonomy.logs import get_logger

from .rate_limiter import AdaptiveRateLimiter, RateLimiterConfig, RateLimiterManager


class RequestPriority(IntEnum):
  """Priority levels for queued requests."""

  LOW = 0
  NORMAL = 1
  HIGH = 2
  CRITICAL = 3


T = TypeVar("T")


@dataclass(order=True)
class QueuedRequest(Generic[T]):
  """A request waiting in the queue."""

  priority: int
  sequence: int  # For FIFO ordering within same priority
  request_fn: Callable[[], Awaitable[T]] = field(compare=False)
  future: asyncio.Future[T] = field(compare=False)
  created_at: float = field(compare=False)
  retry_count: int = field(default=0, compare=False)


@dataclass
class QueueStats:
  """Statistics for the request queue."""

  queue_depth: int
  pending_requests: int
  completed_requests: int
  failed_requests: int
  rate_limited_requests: int
  average_wait_time_ms: float
  is_running: bool


@dataclass
class RetryConfig:
  """Configuration for retry behavior."""

  # Maximum number of retry attempts
  max_retry_attempts: int = 3

  # Initial backoff between retries (seconds)
  initial_seconds_between_retry_attempts: float = 5.0

  # Maximum backoff between retries (seconds)
  max_seconds_between_retry_attempts: float = 60.0


@dataclass
class QueueConfig:
  """Configuration for ModelRequestQueue."""

  # Maximum queue depth (requests waiting)
  max_queue_depth: int = 1000

  # Timeout for queue operations (seconds)
  queue_timeout: float = 300.0

  # Number of dispatcher workers
  num_workers: int = 5

  # Rate limiter configuration
  rate_limiter_config: Optional[RateLimiterConfig] = None

  # Retry configuration
  retry_config: Optional[RetryConfig] = None


class ModelRequestQueue:
  """
  Queue for managing model requests with rate limiting.

  This queue:
  - Accepts requests and returns futures
  - Dispatches requests through the rate limiter
  - Handles backpressure when queue is full
  - Supports priority ordering

  Usage:
      queue = ModelRequestQueue(model="claude-sonnet-4")
      await queue.start()

      # Submit a request
      async def make_request():
          return await client.complete(...)

      result = await queue.submit(make_request)

      # Shutdown
      await queue.stop()
  """

  def __init__(
    self,
    model: str,
    config: Optional[QueueConfig] = None,
    rate_limiter: Optional[AdaptiveRateLimiter] = None,
  ):
    """
    Initialize the request queue.

    :param model: Model name
    :param config: Queue configuration
    :param rate_limiter: Optional rate limiter (created if not provided)
    """
    self.model = model
    self.config = config or QueueConfig()
    self.retry_config = self.config.retry_config or RetryConfig()
    self.logger = get_logger("request_queue")

    # Create or use provided rate limiter
    if rate_limiter is not None:
      self._rate_limiter = rate_limiter
    else:
      limiter_config = self.config.rate_limiter_config or RateLimiterConfig()
      self._rate_limiter = AdaptiveRateLimiter(model, limiter_config)

    # Priority queue (using heapq via PriorityQueue)
    self._queue: asyncio.PriorityQueue[QueuedRequest] = asyncio.PriorityQueue(maxsize=self.config.max_queue_depth)

    # Sequence counter for FIFO within priority
    self._sequence = 0
    self._sequence_lock = asyncio.Lock()

    # Worker tasks
    self._workers: list[asyncio.Task] = []
    self._running = False
    self._shutdown_event = asyncio.Event()

    # Statistics
    self._pending_requests = 0
    self._completed_requests = 0
    self._failed_requests = 0
    self._rate_limited_requests = 0
    self._retried_requests = 0
    self._total_wait_time_ms = 0.0

    self.logger.debug(
      f"Initialized request queue for {model}: "
      f"max_depth={self.config.max_queue_depth}, "
      f"workers={self.config.num_workers}, "
      f"max_retries={self.retry_config.max_retry_attempts}"
    )

  async def start(self) -> None:
    """Start the queue dispatcher workers."""
    if self._running:
      return

    self._running = True
    self._shutdown_event.clear()

    # Start worker tasks
    for i in range(self.config.num_workers):
      worker = asyncio.create_task(self._worker(i))
      self._workers.append(worker)

    self.logger.info(f"Started request queue for {self.model} with {self.config.num_workers} workers")

  async def stop(self, timeout: float = 30.0) -> None:
    """
    Stop the queue and wait for pending requests.

    :param timeout: Maximum time to wait for pending requests
    """
    if not self._running:
      return

    self._running = False
    self._shutdown_event.set()

    # Wait for workers to finish
    if self._workers:
      try:
        await asyncio.wait_for(
          asyncio.gather(*self._workers, return_exceptions=True),
          timeout=timeout,
        )
      except asyncio.TimeoutError:
        self.logger.warning(f"Timeout waiting for queue workers to finish for {self.model}")
        for worker in self._workers:
          worker.cancel()

    self._workers.clear()
    self.logger.info(f"Stopped request queue for {self.model}")

  async def submit(
    self,
    request_fn: Callable[[], Awaitable[T]],
    priority: RequestPriority = RequestPriority.NORMAL,
    timeout: Optional[float] = None,
  ) -> T:
    """
    Submit a request to the queue.

    :param request_fn: Async function that makes the actual request
    :param priority: Request priority
    :param timeout: Optional timeout for the entire operation
    :return: Result from request_fn
    :raises asyncio.TimeoutError: If timeout exceeded
    :raises asyncio.QueueFull: If queue is full
    """
    if not self._running:
      # If queue not running, execute directly (fallback behavior)
      return await request_fn()

    created_at = time.monotonic()
    timeout = timeout or self.config.queue_timeout

    # Create future for result
    loop = asyncio.get_event_loop()
    future: asyncio.Future[T] = loop.create_future()

    # Get sequence number for FIFO ordering
    async with self._sequence_lock:
      sequence = self._sequence
      self._sequence += 1

    # Create queued request (negative priority for max-heap behavior)
    queued = QueuedRequest(
      priority=-priority,  # Negate for correct ordering
      sequence=sequence,
      request_fn=request_fn,
      future=future,
      created_at=created_at,
    )

    # Add to queue
    try:
      self._queue.put_nowait(queued)
      self._pending_requests += 1
    except asyncio.QueueFull:
      self.logger.warning(f"Queue full for {self.model}, rejecting request")
      raise

    # Wait for result
    try:
      result = await asyncio.wait_for(future, timeout=timeout)

      # Update wait time stats
      wait_time_ms = (time.monotonic() - created_at) * 1000
      self._total_wait_time_ms += wait_time_ms

      return result
    except asyncio.TimeoutError:
      # Try to cancel the request if still pending
      if not future.done():
        future.cancel()
      raise

  async def _worker(self, worker_id: int) -> None:
    """
    Worker that processes requests from the queue.

    :param worker_id: Worker identifier for logging
    """
    self.logger.debug(f"Worker {worker_id} started for {self.model}")

    while self._running or not self._queue.empty():
      try:
        # Wait for a request with timeout
        try:
          queued = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
          if not self._running:
            break
          continue

        self._pending_requests -= 1

        # Skip if already cancelled
        if queued.future.cancelled():
          self._queue.task_done()
          continue

        # Execute the request through rate limiter
        try:
          await self._execute_request(queued)
        except Exception as e:
          self.logger.error(f"Worker {worker_id} error executing request: {e}")
        finally:
          self._queue.task_done()

      except asyncio.CancelledError:
        break
      except Exception as e:
        self.logger.error(f"Worker {worker_id} unexpected error: {e}")

    self.logger.debug(f"Worker {worker_id} stopped for {self.model}")

  async def _execute_request(self, queued: QueuedRequest) -> None:
    """
    Execute a queued request through the rate limiter with retry logic.

    Retry behavior:
    - 429 (rate limit): retry with exponential backoff
    - 5xx (server error): retry with exponential backoff
    - 4xx (client error, except 429): no retry

    :param queued: The queued request to execute
    """
    try:
      # Acquire rate limit token
      await self._rate_limiter.acquire()

      try:
        # Execute the request
        result = await queued.request_fn()

        # Try to extract rate limit hints from response
        # Note: The result may include headers if using with_raw_response
        hint_rpm, circuit_state = self._extract_hints_from_result(result)

        # Record success with optional hints
        self._rate_limiter.record_success(hint_rpm=hint_rpm, circuit_state=circuit_state)
        self._rate_limiter.release()

        # Set result on future
        if not queued.future.done():
          queued.future.set_result(result)
        self._completed_requests += 1

      except Exception as e:
        self._rate_limiter.release()

        # Check for different error types and handle accordingly
        circuit_state = self._extract_circuit_state_from_error(e)
        is_rate_limit = self._is_rate_limit_error(e)
        is_server_error = self._is_server_error(e)

        if circuit_state == "exhausted":
          # All providers exhausted - very aggressive backoff
          self._rate_limiter.record_circuit_exhausted()
          self._rate_limited_requests += 1
        elif is_rate_limit:
          # Extract retry_after and hint_rpm from error response
          retry_after, hint_rpm = self._extract_rate_limit_info(e)
          self._rate_limiter.record_rate_limit(retry_after=retry_after)
          self._rate_limited_requests += 1

        # Determine if we should retry
        should_retry = (is_rate_limit or is_server_error) and not circuit_state == "exhausted"

        if should_retry and queued.retry_count < self.retry_config.max_retry_attempts:
          # Calculate exponential backoff
          backoff = min(
            self.retry_config.initial_seconds_between_retry_attempts * (2 ** queued.retry_count),
            self.retry_config.max_seconds_between_retry_attempts,
          )

          self.logger.info(
            f"Retrying request for {self.model} (attempt {queued.retry_count + 1}/{self.retry_config.max_retry_attempts}) "
            f"after {backoff:.1f}s backoff"
          )

          # Increment retry count
          queued.retry_count += 1
          self._retried_requests += 1

          # Wait for backoff period
          await asyncio.sleep(backoff)

          # Re-queue the request (it will be processed by next available worker)
          try:
            self._queue.put_nowait(queued)
            self._pending_requests += 1
          except asyncio.QueueFull:
            # Queue is full, fail the request
            if not queued.future.done():
              queued.future.set_exception(e)
            self._failed_requests += 1
        else:
          # No more retries or non-retryable error - propagate error to future
          if not queued.future.done():
            queued.future.set_exception(e)
          self._failed_requests += 1

    except asyncio.TimeoutError:
      if not queued.future.done():
        queued.future.set_exception(asyncio.TimeoutError("Rate limit acquire timeout"))
      self._failed_requests += 1

  def _is_rate_limit_error(self, error: Exception) -> bool:
    """
    Check if an error is a rate limit error (429).

    :param error: The exception to check
    :return: True if this is a rate limit error
    """
    # Check for OpenAI rate limit errors
    error_str = str(error).lower()
    if "rate" in error_str and "limit" in error_str:
      return True
    if "429" in error_str:
      return True

    # Check for specific exception types
    try:
      from openai import RateLimitError

      if isinstance(error, RateLimitError):
        return True
    except ImportError:
      pass

    try:
      from anthropic import RateLimitError as AnthropicRateLimitError

      if isinstance(error, AnthropicRateLimitError):
        return True
    except ImportError:
      pass

    return False

  def _is_server_error(self, error: Exception) -> bool:
    """
    Check if an error is a server error (5xx).

    :param error: The exception to check
    :return: True if this is a server error
    """
    error_str = str(error).lower()

    # Check for 5xx status codes in error message
    for code in ["500", "502", "503", "504"]:
      if code in error_str:
        return True

    # Check for specific exception types
    try:
      from openai import InternalServerError, APIConnectionError

      if isinstance(error, (InternalServerError, APIConnectionError)):
        return True
    except ImportError:
      pass

    try:
      from anthropic import InternalServerError as AnthropicInternalServerError
      from anthropic import APIConnectionError as AnthropicAPIConnectionError

      if isinstance(error, (AnthropicInternalServerError, AnthropicAPIConnectionError)):
        return True
    except ImportError:
      pass

    return False

  def _extract_rate_limit_info(self, error: Exception) -> tuple[Optional[float], Optional[float]]:
    """
    Extract rate limit information from an error response.

    :param error: The rate limit exception
    :return: Tuple of (retry_after_seconds, hint_rpm)
    """
    retry_after: Optional[float] = None
    hint_rpm: Optional[float] = None

    try:
      # OpenAI SDK exposes response headers via the response attribute
      from openai import RateLimitError

      if isinstance(error, RateLimitError) and hasattr(error, "response"):
        response = error.response
        if hasattr(response, "headers"):
          headers = response.headers

          # Extract Retry-After header
          if "retry-after" in headers:
            try:
              retry_after = float(headers["retry-after"])
            except (ValueError, TypeError):
              pass

          # Extract X-RateLimit-Hint-RPM header (from our gateway)
          if "x-ratelimit-hint-rpm" in headers:
            try:
              hint_rpm = float(headers["x-ratelimit-hint-rpm"])
            except (ValueError, TypeError):
              pass

    except ImportError:
      pass
    except Exception as e:
      self.logger.debug(f"Error extracting rate limit info: {e}")

    return retry_after, hint_rpm

  def _extract_hints_from_result(self, result: Any) -> tuple[Optional[float], Optional[str]]:
    """
    Extract rate limit hints from a successful response.

    The GatewayClient attaches gateway hints to the response object as attributes.
    If using with_raw_response directly, the result includes headers.
    Otherwise, we rely on AIMD for rate discovery.

    :param result: The response object (may have attached hints or headers)
    :return: Tuple of (hint_rpm, circuit_state)
    """
    hint_rpm: Optional[float] = None
    circuit_state: Optional[str] = None

    # First, check for hints attached by GatewayClient
    if hasattr(result, "_gateway_hint_rpm"):
      hint_rpm = getattr(result, "_gateway_hint_rpm", None)
    if hasattr(result, "_gateway_circuit_state"):
      circuit_state = getattr(result, "_gateway_circuit_state", None)

    # If hints were attached, use them
    if hint_rpm is not None or circuit_state is not None:
      return hint_rpm, circuit_state

    # Fall back to checking raw response headers
    if hasattr(result, "headers"):
      headers = result.headers

      # Extract X-RateLimit-Hint-RPM
      hint_rpm_str = headers.get("x-ratelimit-hint-rpm")
      if hint_rpm_str:
        try:
          hint_rpm = float(hint_rpm_str)
        except (ValueError, TypeError):
          pass

      # Extract X-Gateway-Circuit-State
      circuit_state = headers.get("x-gateway-circuit-state")

    return hint_rpm, circuit_state

  def _extract_circuit_state_from_error(self, error: Exception) -> Optional[str]:
    """
    Extract circuit breaker state from an error response.

    :param error: The exception
    :return: Circuit state if available (closed, half-open, open, exhausted)
    """
    try:
      # Check for APIStatusError with response headers
      if hasattr(error, "response") and hasattr(error.response, "headers"):
        headers = error.response.headers
        return headers.get("x-gateway-circuit-state")
    except Exception:
      pass

    return None

  @property
  def rate_limiter(self) -> AdaptiveRateLimiter:
    """Get the rate limiter for this queue."""
    return self._rate_limiter

  @property
  def stats(self) -> QueueStats:
    """Get current queue statistics."""
    avg_wait = 0.0
    total_completed = self._completed_requests + self._failed_requests
    if total_completed > 0:
      avg_wait = self._total_wait_time_ms / total_completed

    return QueueStats(
      queue_depth=self._queue.qsize(),
      pending_requests=self._pending_requests,
      completed_requests=self._completed_requests,
      failed_requests=self._failed_requests,
      rate_limited_requests=self._rate_limited_requests,
      average_wait_time_ms=avg_wait,
      is_running=self._running,
    )


class QueueManager:
  """
  Manages request queues for multiple models.

  This is a singleton that provides per-model queues with lazy initialization.
  """

  _instance: Optional["QueueManager"] = None
  _lock: asyncio.Lock = asyncio.Lock()

  def __init__(self):
    self._queues: dict[str, ModelRequestQueue] = {}
    self._configs: dict[str, QueueConfig] = {}
    self._default_config = QueueConfig()
    self._rate_limiter_manager: Optional[RateLimiterManager] = None
    self.logger = get_logger("queue_manager")

  @classmethod
  async def get_instance(cls) -> "QueueManager":
    """Get the singleton instance."""
    if cls._instance is None:
      async with cls._lock:
        if cls._instance is None:
          cls._instance = QueueManager()
          cls._instance._rate_limiter_manager = await RateLimiterManager.get_instance()
    return cls._instance

  def configure_model(self, model: str, config: QueueConfig) -> None:
    """
    Configure queue for a specific model.

    :param model: Model name
    :param config: Queue configuration
    """
    self._configs[model] = config

  def set_default_config(self, config: QueueConfig) -> None:
    """
    Set the default configuration for new models.

    :param config: Default queue configuration
    """
    self._default_config = config

  async def get_queue(self, model: str, auto_start: bool = True) -> ModelRequestQueue:
    """
    Get or create a queue for a model.

    :param model: Model name
    :param auto_start: Whether to automatically start the queue
    :return: Queue for the model
    """
    if model not in self._queues:
      config = self._configs.get(model, self._default_config)

      # Get rate limiter from manager for shared state
      rate_limiter = None
      if self._rate_limiter_manager:
        rate_limiter = self._rate_limiter_manager.get_limiter(model)

      queue = ModelRequestQueue(model, config, rate_limiter)
      self._queues[model] = queue

      if auto_start:
        await queue.start()

      self.logger.debug(f"Created queue for {model}")

    return self._queues[model]

  async def shutdown(self, timeout: float = 30.0) -> None:
    """
    Shutdown all queues.

    :param timeout: Maximum time to wait per queue
    """
    self.logger.info(f"Shutting down {len(self._queues)} queues")

    # Stop all queues concurrently
    await asyncio.gather(
      *[queue.stop(timeout=timeout) for queue in self._queues.values()],
      return_exceptions=True,
    )

    self._queues.clear()
    self.logger.info("All queues shut down")

  def get_all_stats(self) -> dict[str, QueueStats]:
    """Get statistics for all queues."""
    return {model: queue.stats for model, queue in self._queues.items()}

  def reset(self) -> None:
    """Reset all queues (for testing)."""
    self._queues.clear()
    self._configs.clear()
    self._default_config = QueueConfig()
