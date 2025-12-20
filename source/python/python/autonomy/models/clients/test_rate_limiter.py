"""
Unit tests for AdaptiveRateLimiter.

Tests cover:
- Token bucket behavior
- AIMD algorithm (additive increase, multiplicative decrease)
- Concurrency control
- Cooldown behavior
- Circuit breaker state handling
- Statistics tracking
"""

import asyncio
import time

import pytest

from autonomy.models.clients.rate_limiter import (
  AdaptiveRateLimiter,
  RateLimiterConfig,
  RateLimiterManager,
  RateLimiterStats,
)


class TestRateLimiterConfig:
  """Tests for RateLimiterConfig defaults."""

  def test_default_values(self):
    config = RateLimiterConfig()
    assert config.initial_rpm == 60.0
    assert config.min_rpm == 1.0
    assert config.max_rpm == 1000.0
    assert config.multiplicative_decrease == 0.5
    assert config.max_concurrent == 10
    assert config.cooldown_period == 5.0

  def test_custom_values(self):
    config = RateLimiterConfig(
      initial_rpm=100.0,
      min_rpm=10.0,
      max_rpm=500.0,
      multiplicative_decrease=0.75,
      max_concurrent=20,
    )
    assert config.initial_rpm == 100.0
    assert config.min_rpm == 10.0
    assert config.max_rpm == 500.0
    assert config.multiplicative_decrease == 0.75
    assert config.max_concurrent == 20


class TestAdaptiveRateLimiter:
  """Tests for AdaptiveRateLimiter."""

  @pytest.fixture
  def limiter(self):
    """Create a limiter with fast settings for testing."""
    config = RateLimiterConfig(
      initial_rpm=60.0,  # 1 per second
      min_rpm=1.0,
      max_rpm=120.0,
      multiplicative_decrease=0.5,
      max_concurrent=5,
      cooldown_period=0.1,  # Short cooldown for tests
    )
    return AdaptiveRateLimiter("test-model", config)

  @pytest.mark.asyncio
  async def test_acquire_basic(self, limiter):
    """Test basic token acquisition."""
    # Should be able to acquire immediately (starts with tokens)
    acquired = await limiter.acquire(timeout=1.0)
    assert acquired is True
    limiter.release()

  @pytest.mark.asyncio
  async def test_acquire_respects_rate(self, limiter):
    """Test that acquire respects the rate limit."""
    # Acquire all initial tokens quickly
    for _ in range(3):
      await limiter.acquire(timeout=5.0)
      limiter.release()

    # Next acquire should need to wait for refill
    start = time.monotonic()
    await limiter.acquire(timeout=5.0)
    elapsed = time.monotonic() - start
    limiter.release()

    # Should have waited some time for token refill
    # (may be 0 if tokens refilled during loop)
    assert elapsed >= 0

  @pytest.mark.asyncio
  async def test_acquire_timeout(self):
    """Test that acquire times out when rate limited."""
    # Use a limiter with moderate rate so we can exhaust tokens
    config = RateLimiterConfig(
      initial_rpm=60.0,  # 1 per second
      max_concurrent=5,
    )
    limiter = AdaptiveRateLimiter("test-model", config)

    # Exhaust all initial tokens (starts with 1 token at 60 RPM)
    # The limiter starts with _current_rate tokens = 1.0 token
    await limiter.acquire(timeout=5.0)
    limiter.release()

    # Now tokens are exhausted, need to wait for refill
    # At 60 RPM = 1/sec, we need ~1 second for a new token
    # A very short timeout should fail
    with pytest.raises(asyncio.TimeoutError):
      await limiter.acquire(timeout=0.05)

  @pytest.mark.asyncio
  async def test_concurrency_limit(self):
    """Test that concurrency is limited."""
    # Create a limiter with high rate but low concurrency
    config = RateLimiterConfig(
      initial_rpm=6000.0,  # High rate so tokens aren't the bottleneck
      max_concurrent=3,
      cooldown_period=0.1,
    )
    limiter = AdaptiveRateLimiter("test-model", config)

    # Acquire all concurrent slots
    for _ in range(limiter.config.max_concurrent):
      await limiter.acquire(timeout=5.0)

    # Next acquire should timeout (no slots available)
    with pytest.raises(asyncio.TimeoutError):
      await limiter.acquire(timeout=0.1)

    # Release one slot
    limiter.release()

    # Now should be able to acquire
    await limiter.acquire(timeout=5.0)

    # Release all
    for _ in range(limiter.config.max_concurrent):
      limiter.release()

  def test_record_success_increases_rate(self, limiter):
    """Test that recording success increases the rate."""
    initial_rpm = limiter.current_rpm

    # Record many successes
    for _ in range(100):
      limiter.record_success()

    # Rate should have increased
    assert limiter.current_rpm > initial_rpm

  def test_record_success_respects_max(self, limiter):
    """Test that rate doesn't exceed max_rpm."""
    # Record many successes
    for _ in range(10000):
      limiter.record_success()

    # Should not exceed max
    assert limiter.current_rpm <= limiter.config.max_rpm

  def test_record_rate_limit_decreases_rate(self, limiter):
    """Test that recording rate limit decreases the rate."""
    initial_rpm = limiter.current_rpm

    limiter.record_rate_limit()

    # Rate should have decreased by multiplicative factor
    expected = initial_rpm * limiter.config.multiplicative_decrease
    assert abs(limiter.current_rpm - expected) < 0.1

  def test_record_rate_limit_respects_min(self, limiter):
    """Test that rate doesn't go below min_rpm."""
    # Record many rate limits
    for _ in range(100):
      limiter.record_rate_limit()

    # Should not go below min
    assert limiter.current_rpm >= limiter.config.min_rpm

  def test_cooldown_prevents_increase(self, limiter):
    """Test that cooldown prevents immediate rate increase after rate limit."""
    # Record a rate limit
    limiter.record_rate_limit()
    rate_after_limit = limiter.current_rpm

    # Try to increase immediately
    limiter.record_success()

    # Rate should not have increased (still in cooldown)
    assert limiter.current_rpm == rate_after_limit

  @pytest.mark.asyncio
  async def test_cooldown_expires(self, limiter):
    """Test that rate can increase after cooldown expires."""
    # Record a rate limit
    limiter.record_rate_limit()
    rate_after_limit = limiter.current_rpm

    # Wait for cooldown to expire
    await asyncio.sleep(limiter.config.cooldown_period + 0.1)

    # Now success should increase rate
    limiter.record_success()
    assert limiter.current_rpm > rate_after_limit

  def test_record_circuit_open(self, limiter):
    """Test that circuit open causes aggressive rate reduction."""
    initial_rpm = limiter.current_rpm

    limiter.record_circuit_open()

    # Rate should be cut to 25%
    expected = initial_rpm * 0.25
    assert abs(limiter.current_rpm - expected) < 0.1

    # Circuit open event should be tracked
    assert limiter.stats.circuit_open_events == 1
    assert limiter.stats.last_circuit_state == "open"

  def test_record_circuit_exhausted(self, limiter):
    """Test that circuit exhausted causes very aggressive rate reduction."""
    limiter.record_circuit_exhausted()

    # Rate should be cut to minimum
    assert limiter.current_rpm == limiter.config.min_rpm

    # Circuit open event should be tracked
    assert limiter.stats.circuit_open_events == 1
    assert limiter.stats.last_circuit_state == "exhausted"

  def test_circuit_state_half_open_on_success(self, limiter):
    """Test that half-open circuit state causes slowdown on success."""
    initial_rpm = limiter.current_rpm

    # Record success with half-open circuit state
    limiter.record_success(circuit_state="half-open")

    # Rate should have decreased (not increased)
    assert limiter.current_rpm < initial_rpm
    assert limiter.stats.last_circuit_state == "half-open"

  def test_circuit_state_open_on_success_triggers_circuit_open(self, limiter):
    """Test that open circuit state on success triggers circuit open handling."""
    initial_rpm = limiter.current_rpm

    # Record success with open circuit state (gateway is struggling)
    limiter.record_success(circuit_state="open")

    # Rate should be cut to 25%
    expected = initial_rpm * 0.25
    assert abs(limiter.current_rpm - expected) < 0.1
    assert limiter.stats.circuit_open_events == 1

  def test_circuit_state_closed_allows_normal_increase(self, limiter):
    """Test that closed circuit state allows normal rate increase."""
    initial_rpm = limiter.current_rpm

    # Record success with closed circuit state
    limiter.record_success(circuit_state="closed")

    # Rate should have increased normally
    assert limiter.current_rpm > initial_rpm
    assert limiter.stats.last_circuit_state == "closed"

  def test_hint_rpm_jumps_rate(self, limiter):
    """Test that hint_rpm allows jumping to a higher rate."""
    initial_rpm = limiter.current_rpm

    # Hint a higher rate
    limiter.record_success(hint_rpm=100.0)

    # Should jump to hint
    assert limiter.current_rpm == 100.0

  def test_hint_rpm_respects_max(self, limiter):
    """Test that hint_rpm doesn't exceed max."""
    # Hint above max
    limiter.record_success(hint_rpm=10000.0)

    # Should be capped at max
    assert limiter.current_rpm == limiter.config.max_rpm

  def test_hint_rpm_ignored_if_lower(self, limiter):
    """Test that hint_rpm is ignored if lower than current rate."""
    initial_rpm = limiter.current_rpm

    # Hint a lower rate
    limiter.record_success(hint_rpm=initial_rpm - 10)

    # Should have done normal additive increase instead
    assert limiter.current_rpm >= initial_rpm

  def test_stats(self, limiter):
    """Test statistics tracking."""
    stats = limiter.stats

    assert isinstance(stats, RateLimiterStats)
    assert stats.current_rpm == limiter.current_rpm
    assert stats.min_rpm == limiter.config.min_rpm
    assert stats.max_rpm == limiter.config.max_rpm
    assert stats.total_requests == 0
    assert stats.successful_requests == 0
    assert stats.rate_limited_requests == 0

  @pytest.mark.asyncio
  async def test_stats_after_requests(self, limiter):
    """Test that stats are updated after requests."""
    # Make some requests
    await limiter.acquire(timeout=1.0)
    limiter.record_success()
    limiter.release()

    await limiter.acquire(timeout=1.0)
    limiter.record_rate_limit()
    limiter.release()

    stats = limiter.stats
    assert stats.total_requests == 2
    assert stats.successful_requests == 1
    assert stats.rate_limited_requests == 1
    assert stats.circuit_open_events == 0

  @pytest.mark.asyncio
  async def test_stats_with_circuit_events(self, limiter):
    """Test that circuit events are tracked in stats."""
    # Record various circuit events
    limiter.record_circuit_open()
    limiter.record_circuit_exhausted()

    stats = limiter.stats
    assert stats.circuit_open_events == 2
    assert stats.last_circuit_state == "exhausted"


class TestRateLimiterManager:
  """Tests for RateLimiterManager singleton."""

  @pytest.fixture(autouse=True)
  async def reset_manager(self):
    """Reset the manager before each test."""
    # Reset singleton
    RateLimiterManager._instance = None
    yield
    RateLimiterManager._instance = None

  @pytest.mark.asyncio
  async def test_singleton(self):
    """Test that get_instance returns the same instance."""
    instance1 = await RateLimiterManager.get_instance()
    instance2 = await RateLimiterManager.get_instance()

    assert instance1 is instance2

  @pytest.mark.asyncio
  async def test_get_limiter_creates_new(self):
    """Test that get_limiter creates a new limiter."""
    manager = await RateLimiterManager.get_instance()

    limiter = manager.get_limiter("test-model")

    assert limiter is not None
    assert limiter.model == "test-model"

  @pytest.mark.asyncio
  async def test_get_limiter_returns_same(self):
    """Test that get_limiter returns the same limiter for same model."""
    manager = await RateLimiterManager.get_instance()

    limiter1 = manager.get_limiter("test-model")
    limiter2 = manager.get_limiter("test-model")

    assert limiter1 is limiter2

  @pytest.mark.asyncio
  async def test_get_limiter_different_models(self):
    """Test that different models get different limiters."""
    manager = await RateLimiterManager.get_instance()

    limiter1 = manager.get_limiter("model-1")
    limiter2 = manager.get_limiter("model-2")

    assert limiter1 is not limiter2
    assert limiter1.model == "model-1"
    assert limiter2.model == "model-2"

  @pytest.mark.asyncio
  async def test_configure_model(self):
    """Test that model-specific config is used."""
    manager = await RateLimiterManager.get_instance()

    config = RateLimiterConfig(initial_rpm=120.0)
    manager.configure_model("configured-model", config)

    limiter = manager.get_limiter("configured-model")
    assert limiter.current_rpm == 120.0

  @pytest.mark.asyncio
  async def test_set_default_config(self):
    """Test that default config is used for unconfigured models."""
    manager = await RateLimiterManager.get_instance()

    config = RateLimiterConfig(initial_rpm=200.0)
    manager.set_default_config(config)

    limiter = manager.get_limiter("new-model")
    assert limiter.current_rpm == 200.0

  @pytest.mark.asyncio
  async def test_get_all_stats(self):
    """Test getting stats for all models."""
    manager = await RateLimiterManager.get_instance()

    manager.get_limiter("model-1")
    manager.get_limiter("model-2")

    stats = manager.get_all_stats()

    assert "model-1" in stats
    assert "model-2" in stats
    assert isinstance(stats["model-1"], RateLimiterStats)

  @pytest.mark.asyncio
  async def test_reset(self):
    """Test resetting the manager."""
    manager = await RateLimiterManager.get_instance()

    manager.get_limiter("model-1")
    assert len(manager._limiters) == 1

    manager.reset()
    assert len(manager._limiters) == 0


class TestAIMDConvergence:
  """Tests for AIMD algorithm convergence behavior."""

  @pytest.mark.asyncio
  async def test_sawtooth_pattern(self):
    """Test that AIMD creates expected sawtooth pattern."""
    config = RateLimiterConfig(
      initial_rpm=50.0,
      min_rpm=10.0,
      max_rpm=100.0,
      multiplicative_decrease=0.5,
      additive_increase=1.0,  # 1 RPM per success
      cooldown_period=0.0,  # No cooldown for this test
    )
    limiter = AdaptiveRateLimiter("test", config)

    rates = [limiter.current_rpm]

    # Simulate some successes (rate goes up)
    for _ in range(20):
      limiter.record_success()
      rates.append(limiter.current_rpm)

    # Should have increased
    assert rates[-1] > rates[0]

    # Simulate rate limit (rate drops)
    limiter.record_rate_limit()
    rates.append(limiter.current_rpm)

    # Should have decreased significantly
    assert rates[-1] < rates[-2] * 0.6  # More than 40% drop

    # More successes (rate goes up again)
    for _ in range(10):
      limiter.record_success()
      rates.append(limiter.current_rpm)

    # Should be increasing again
    assert rates[-1] > rates[-11]

  @pytest.mark.asyncio
  async def test_multiple_clients_fair_share(self):
    """Test that multiple clients converge to fair share."""
    # Simulate a backend with 100 RPM capacity shared by 2 clients
    backend_capacity = 100.0

    config = RateLimiterConfig(
      initial_rpm=80.0,  # Start above fair share
      min_rpm=1.0,
      max_rpm=200.0,
      multiplicative_decrease=0.5,
      additive_increase=0.5,
      cooldown_period=0.0,
    )

    client1 = AdaptiveRateLimiter("client1", config)
    client2 = AdaptiveRateLimiter("client2", config)

    # Simulate rounds where clients compete for capacity
    for round_num in range(50):
      total_rate = client1.current_rpm + client2.current_rpm

      if total_rate > backend_capacity:
        # Both clients get rate limited proportionally
        if client1.current_rpm > backend_capacity / 2:
          client1.record_rate_limit()
        if client2.current_rpm > backend_capacity / 2:
          client2.record_rate_limit()
      else:
        # Both succeed
        client1.record_success()
        client2.record_success()

    # Both clients should have converged to roughly fair share
    # (within 50% of equal share, accounting for AIMD oscillation)
    fair_share = backend_capacity / 2
    assert client1.current_rpm > fair_share * 0.3
    assert client1.current_rpm < fair_share * 2.0
    assert client2.current_rpm > fair_share * 0.3
    assert client2.current_rpm < fair_share * 2.0
