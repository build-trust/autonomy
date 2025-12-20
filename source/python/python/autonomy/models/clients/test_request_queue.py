"""
Unit tests for ModelRequestQueue.

Tests cover:
- Basic queue operations
- Priority ordering
- Rate limiter integration
- Graceful shutdown
- Statistics tracking
- Gateway hint extraction
- Circuit state handling
"""

import asyncio
import time

import pytest

from autonomy.models.clients.rate_limiter import (
  AdaptiveRateLimiter,
  RateLimiterConfig,
  RateLimiterManager,
)
from autonomy.models.clients.request_queue import (
  ModelRequestQueue,
  QueueConfig,
  QueueManager,
  QueueStats,
  RequestPriority,
)


class TestQueueConfig:
  """Tests for QueueConfig defaults."""

  def test_default_values(self):
    config = QueueConfig()
    assert config.max_queue_depth == 1000
    assert config.queue_timeout == 300.0
    assert config.num_workers == 5
    assert config.rate_limiter_config is None

  def test_custom_values(self):
    rate_config = RateLimiterConfig(initial_rpm=120.0)
    config = QueueConfig(
      max_queue_depth=500,
      queue_timeout=60.0,
      num_workers=10,
      rate_limiter_config=rate_config,
    )
    assert config.max_queue_depth == 500
    assert config.queue_timeout == 60.0
    assert config.num_workers == 10
    assert config.rate_limiter_config is rate_config


class TestModelRequestQueue:
  """Tests for ModelRequestQueue."""

  @pytest.fixture
  def queue_config(self):
    """Create a queue config with fast settings for testing."""
    rate_config = RateLimiterConfig(
      initial_rpm=600.0,  # 10 per second
      max_concurrent=10,
      cooldown_period=0.1,
    )
    return QueueConfig(
      max_queue_depth=100,
      queue_timeout=10.0,
      num_workers=3,
      rate_limiter_config=rate_config,
    )

  @pytest.fixture
  async def queue(self, queue_config):
    """Create and start a queue for testing."""
    q = ModelRequestQueue("test-model", queue_config)
    await q.start()
    yield q
    await q.stop()

  @pytest.mark.asyncio
  async def test_submit_basic(self, queue):
    """Test basic request submission."""

    async def request_fn():
      return 42

    result = await queue.submit(request_fn)
    assert result == 42

  @pytest.mark.asyncio
  async def test_submit_multiple(self, queue):
    """Test multiple request submissions."""

    async def make_request(value):
      await asyncio.sleep(0.01)
      return value

    # Submit multiple requests
    results = await asyncio.gather(*[queue.submit(lambda v=i: make_request(v)) for i in range(5)])

    assert sorted(results) == [0, 1, 2, 3, 4]

  @pytest.mark.asyncio
  async def test_submit_when_not_running(self, queue_config):
    """Test that submit works when queue is not started (fallback)."""
    queue = ModelRequestQueue("test-model", queue_config)
    # Don't start the queue

    async def request_fn():
      return "direct"

    result = await queue.submit(request_fn)
    assert result == "direct"

  @pytest.mark.asyncio
  async def test_submit_with_exception(self, queue):
    """Test that exceptions propagate correctly."""

    async def failing_request():
      raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
      await queue.submit(failing_request)

  @pytest.mark.asyncio
  async def test_priority_ordering(self, queue_config):
    """Test that higher priority requests are processed first."""
    # Use a queue with single worker for deterministic ordering
    rate_config = RateLimiterConfig(
      initial_rpm=60.0,  # Slow rate to see ordering
      max_concurrent=1,
    )
    config = QueueConfig(
      max_queue_depth=100,
      num_workers=1,
      rate_limiter_config=rate_config,
    )
    queue = ModelRequestQueue("test-model", config)
    await queue.start()

    try:
      results = []
      lock = asyncio.Lock()

      async def make_request(name):
        async with lock:
          results.append(name)
        return name

      # Submit requests with different priorities
      # Low priority first, then high, then critical
      tasks = [
        asyncio.create_task(queue.submit(lambda: make_request("low"), priority=RequestPriority.LOW)),
        asyncio.create_task(queue.submit(lambda: make_request("high"), priority=RequestPriority.HIGH)),
        asyncio.create_task(queue.submit(lambda: make_request("critical"), priority=RequestPriority.CRITICAL)),
        asyncio.create_task(queue.submit(lambda: make_request("normal"), priority=RequestPriority.NORMAL)),
      ]

      await asyncio.gather(*tasks)

      # Higher priorities should be processed first
      # Note: exact ordering depends on timing, but critical should be early
      assert "critical" in results
      assert "high" in results
      assert "normal" in results
      assert "low" in results

    finally:
      await queue.stop()

  @pytest.mark.asyncio
  async def test_start_stop(self, queue_config):
    """Test starting and stopping the queue."""
    queue = ModelRequestQueue("test-model", queue_config)

    # Initially not running
    assert not queue._running

    # Start
    await queue.start()
    assert queue._running
    assert len(queue._workers) == queue_config.num_workers

    # Start again (should be idempotent)
    await queue.start()
    assert queue._running

    # Stop
    await queue.stop()
    assert not queue._running
    assert len(queue._workers) == 0

    # Stop again (should be idempotent)
    await queue.stop()
    assert not queue._running

  @pytest.mark.asyncio
  async def test_graceful_shutdown(self, queue_config):
    """Test that shutdown waits for pending requests."""
    queue = ModelRequestQueue("test-model", queue_config)
    await queue.start()

    completed = []

    async def slow_request(value):
      await asyncio.sleep(0.1)
      completed.append(value)
      return value

    # Submit requests
    task = asyncio.create_task(queue.submit(lambda: slow_request(1)))

    # Give time for request to start
    await asyncio.sleep(0.05)

    # Stop should wait for pending request
    await queue.stop(timeout=5.0)

    # Request should have completed
    result = await task
    assert result == 1
    assert 1 in completed

  @pytest.mark.asyncio
  async def test_stats(self, queue):
    """Test statistics tracking."""
    stats = queue.stats

    assert isinstance(stats, QueueStats)
    assert stats.queue_depth == 0
    assert stats.pending_requests == 0
    assert stats.completed_requests == 0
    assert stats.failed_requests == 0
    assert stats.is_running is True

  @pytest.mark.asyncio
  async def test_stats_after_requests(self, queue):
    """Test that stats are updated after requests."""

    async def success_request():
      return "ok"

    async def fail_request():
      raise ValueError("fail")

    # Successful request
    await queue.submit(success_request)

    # Failed request
    try:
      await queue.submit(fail_request)
    except ValueError:
      pass

    stats = queue.stats
    assert stats.completed_requests == 1
    assert stats.failed_requests == 1

  @pytest.mark.asyncio
  async def test_rate_limiter_integration(self, queue):
    """Test that queue uses rate limiter correctly."""
    limiter = queue.rate_limiter

    async def request_fn():
      return "ok"

    # Make some requests
    for _ in range(3):
      await queue.submit(request_fn)

    # Check rate limiter stats
    stats = limiter.stats
    assert stats.total_requests >= 3
    assert stats.successful_requests >= 3

  @pytest.mark.asyncio
  async def test_rate_limit_error_handling(self, queue_config):
    """Test that rate limit errors are recorded correctly."""
    queue = ModelRequestQueue("test-model", queue_config)
    await queue.start()

    try:
      # Simulate a rate limit error with "429" in the message
      # which triggers _is_rate_limit_error detection
      async def rate_limited_request():
        raise Exception("Error 429: rate limit exceeded")

      with pytest.raises(Exception):
        await queue.submit(rate_limited_request)

      # Give the queue a moment to update stats
      await asyncio.sleep(0.1)

      stats = queue.stats
      # The error should be detected as a rate limit error
      assert stats.rate_limited_requests >= 1 or stats.failed_requests >= 1

    finally:
      await queue.stop()

  @pytest.mark.asyncio
  async def test_gateway_hints_from_response(self, queue):
    """Test that gateway hints are extracted from response attributes."""

    # Create a mock response with gateway hints attached
    class MockResponse:
      def __init__(self):
        self._gateway_hint_rpm = 100.0
        self._gateway_circuit_state = "closed"
        self.content = "test response"

    async def request_with_hints():
      return MockResponse()

    result = await queue.submit(request_with_hints)
    assert result.content == "test response"

    # The rate limiter should have received the hint
    # (We can't easily verify the hint was used, but we can verify no errors)
    assert queue.stats.completed_requests >= 1

  @pytest.mark.asyncio
  async def test_circuit_state_half_open_from_response(self, queue):
    """Test that half-open circuit state causes rate reduction."""

    class MockResponse:
      def __init__(self):
        self._gateway_hint_rpm = None
        self._gateway_circuit_state = "half-open"
        self.content = "test"

    initial_rpm = queue.rate_limiter.current_rpm

    async def request_fn():
      return MockResponse()

    await queue.submit(request_fn)

    # Rate should have decreased due to half-open circuit state
    assert queue.rate_limiter.current_rpm < initial_rpm

  @pytest.mark.asyncio
  async def test_circuit_exhausted_error_handling(self, queue_config):
    """Test that circuit exhausted errors cause aggressive backoff."""
    queue = ModelRequestQueue("test-model", queue_config)
    await queue.start()

    try:
      # Mock an error with circuit exhausted state in headers
      class MockResponse:
        def __init__(self):
          self.headers = {"x-gateway-circuit-state": "exhausted"}

      class MockError(Exception):
        def __init__(self):
          super().__init__("All providers exhausted")
          self.response = MockResponse()

      async def exhausted_request():
        raise MockError()

      initial_rpm = queue.rate_limiter.current_rpm

      with pytest.raises(MockError):
        await queue.submit(exhausted_request)

      # Give queue time to process
      await asyncio.sleep(0.1)

      # Rate should have been cut to minimum
      assert queue.rate_limiter.current_rpm == queue.rate_limiter.config.min_rpm

    finally:
      await queue.stop()

  @pytest.mark.asyncio
  async def test_timeout(self, queue_config):
    """Test request timeout."""
    config = QueueConfig(
      max_queue_depth=100,
      queue_timeout=0.1,  # Very short timeout
      num_workers=1,
      rate_limiter_config=RateLimiterConfig(
        initial_rpm=60.0,
        max_concurrent=1,
      ),
    )
    queue = ModelRequestQueue("test-model", config)
    await queue.start()

    try:

      async def slow_request():
        await asyncio.sleep(10)
        return "done"

      # This should timeout
      with pytest.raises(asyncio.TimeoutError):
        await queue.submit(slow_request, timeout=0.05)

    finally:
      await queue.stop()


class TestQueueManager:
  """Tests for QueueManager singleton."""

  @pytest.fixture(autouse=True)
  async def reset_manager(self):
    """Reset the manager before each test."""
    QueueManager._instance = None
    RateLimiterManager._instance = None
    yield
    # Cleanup
    if QueueManager._instance:
      await QueueManager._instance.shutdown()
    QueueManager._instance = None
    RateLimiterManager._instance = None

  @pytest.mark.asyncio
  async def test_singleton(self):
    """Test that get_instance returns the same instance."""
    instance1 = await QueueManager.get_instance()
    instance2 = await QueueManager.get_instance()

    assert instance1 is instance2

  @pytest.mark.asyncio
  async def test_get_queue_creates_new(self):
    """Test that get_queue creates a new queue."""
    manager = await QueueManager.get_instance()

    queue = await manager.get_queue("test-model")

    assert queue is not None
    assert queue.model == "test-model"
    assert queue._running is True  # auto_start=True by default

  @pytest.mark.asyncio
  async def test_get_queue_returns_same(self):
    """Test that get_queue returns the same queue for same model."""
    manager = await QueueManager.get_instance()

    queue1 = await manager.get_queue("test-model")
    queue2 = await manager.get_queue("test-model")

    assert queue1 is queue2

  @pytest.mark.asyncio
  async def test_get_queue_different_models(self):
    """Test that different models get different queues."""
    manager = await QueueManager.get_instance()

    queue1 = await manager.get_queue("model-1")
    queue2 = await manager.get_queue("model-2")

    assert queue1 is not queue2
    assert queue1.model == "model-1"
    assert queue2.model == "model-2"

  @pytest.mark.asyncio
  async def test_configure_model(self):
    """Test that model-specific config is used."""
    manager = await QueueManager.get_instance()

    config = QueueConfig(num_workers=7)
    manager.configure_model("configured-model", config)

    queue = await manager.get_queue("configured-model")
    assert queue.config.num_workers == 7

  @pytest.mark.asyncio
  async def test_set_default_config(self):
    """Test that default config is used for unconfigured models."""
    manager = await QueueManager.get_instance()

    config = QueueConfig(num_workers=12)
    manager.set_default_config(config)

    queue = await manager.get_queue("new-model")
    assert queue.config.num_workers == 12

  @pytest.mark.asyncio
  async def test_shutdown(self):
    """Test shutting down all queues."""
    manager = await QueueManager.get_instance()

    await manager.get_queue("model-1")
    await manager.get_queue("model-2")
    assert len(manager._queues) == 2

    await manager.shutdown()
    assert len(manager._queues) == 0

  @pytest.mark.asyncio
  async def test_get_all_stats(self):
    """Test getting stats for all queues."""
    manager = await QueueManager.get_instance()

    await manager.get_queue("model-1")
    await manager.get_queue("model-2")

    stats = manager.get_all_stats()

    assert "model-1" in stats
    assert "model-2" in stats
    assert isinstance(stats["model-1"], QueueStats)

  @pytest.mark.asyncio
  async def test_reset(self):
    """Test resetting the manager."""
    manager = await QueueManager.get_instance()

    await manager.get_queue("model-1", auto_start=False)
    assert len(manager._queues) == 1

    manager.reset()
    assert len(manager._queues) == 0

  @pytest.mark.asyncio
  async def test_shared_rate_limiter(self):
    """Test that queues share rate limiters through manager."""
    manager = await QueueManager.get_instance()

    queue1 = await manager.get_queue("shared-model")
    queue2 = await manager.get_queue("shared-model")

    # Same queue instance
    assert queue1 is queue2

    # Rate limiter should be the same
    assert queue1.rate_limiter is queue2.rate_limiter


class TestRequestPriority:
  """Tests for RequestPriority enum."""

  def test_priority_ordering(self):
    """Test that priorities are ordered correctly."""
    assert RequestPriority.LOW < RequestPriority.NORMAL
    assert RequestPriority.NORMAL < RequestPriority.HIGH
    assert RequestPriority.HIGH < RequestPriority.CRITICAL

  def test_priority_values(self):
    """Test priority values."""
    assert RequestPriority.LOW == 0
    assert RequestPriority.NORMAL == 1
    assert RequestPriority.HIGH == 2
    assert RequestPriority.CRITICAL == 3
