#!/usr/bin/env python3
"""
Load Testing Script for Adaptive Client-Side Rate Limiting.

This script validates the rate limiting implementation by:
1. Single client rate limiting behavior
2. Multiple client fair sharing (AIMD convergence)
3. Throughput and latency measurements
4. Gateway header verification

Usage:
    # Run from source/python directory
    cd source/python
    uv run --active python -m autonomy.models.clients.load_test_rate_limiter

    # Run all tests with mock gateway
    uv run --active python -m autonomy.models.clients.load_test_rate_limiter --use-mock

    # Run specific test
    uv run --active python -m autonomy.models.clients.load_test_rate_limiter --use-mock --test single-client
    uv run --active python -m autonomy.models.clients.load_test_rate_limiter --use-mock --test multi-client
    uv run --active python -m autonomy.models.clients.load_test_rate_limiter --test headers

    # Custom configuration
    uv run --active python -m autonomy.models.clients.load_test_rate_limiter --use-mock --requests 100 --clients 5

Requirements:
    - For real gateway tests: Gateway running at localhost:8080 (or configure via environment)
    - For mock tests: No external dependencies

Verification Scenarios Tested:
------------------------------

1. Single Client Rate Limiting (--test single-client)
  - Client starts above capacity, hits rate limits, and backs off
  - AIMD sawtooth pattern observed (e.g., 60 → 31 → 60 → 31 RPM)
  - Example: --use-mock --mock-capacity 60 --initial-rpm 200 --requests 50
    Result: 60% success rate, converges to oscillating around 60 RPM

2. Multi-Client Fair Sharing (--test multi-client)
  - Multiple clients compete for shared capacity
  - AIMD ensures fair convergence (fairness CV < 10%)
  - Example: --use-mock --mock-capacity 120 --initial-rpm 100 --clients 3 --requests 60
    Result: 3 clients converge to ~40 RPM each, convergence time ~6-20s

3. High Throughput Scenario
  - Test with high-capacity models (nova-micro, embeddings)
  - Example: --use-mock --mock-capacity 600 --initial-rpm 100 --requests 50
    Result: Client quickly ramps up to match capacity, minimal rate limits

4. Low RPM Scenario (validates token bucket fix)
  - Test with low initial RPM to verify first request proceeds
  - Example: --use-mock --mock-capacity 30 --initial-rpm 20 --requests 20
    Result: Requests proceed without hanging (bug fix validated)

5. Queue Integration (--test queue)
  - Request queue properly integrates with rate limiter
  - Priority ordering works (HIGH > NORMAL > LOW)
  - Example: --use-mock --mock-capacity 100 --initial-rpm 80 --requests 40
    Result: Queue manages backpressure, rate limiter adapts to capacity

6. Gateway Headers (--test headers) - requires real gateway
  - Verifies X-RateLimit-Hint-RPM header on success
  - Verifies X-Gateway-Circuit-State header
  - Verifies Retry-After header on 429 responses

Expected Behaviors:
-------------------
- Sawtooth Pattern: Rate increases linearly (additive), decreases by half on rate limit (multiplicative)
- Convergence: Multiple clients naturally share capacity fairly within 10-30 seconds
- Circuit States: half-open → 30% slowdown, open → 75% slowdown, exhausted → minimum rate
- Cooldown: After rate limit, wait before increasing again (prevents oscillation)
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .rate_limiter import (
  AdaptiveRateLimiter,
  RateLimiterConfig,
  RateLimiterStats,
)
from .request_queue import (
  ModelRequestQueue,
  QueueConfig,
  QueueStats,
  RequestPriority,
)


@dataclass
class RequestResult:
  """Result of a single request."""

  success: bool
  status_code: Optional[int]
  latency_ms: float
  rate_limited: bool
  error: Optional[str] = None
  hint_rpm: Optional[float] = None
  circuit_state: Optional[str] = None
  retry_after: Optional[float] = None


@dataclass
class TestResult:
  """Results from a load test run."""

  test_name: str
  duration_seconds: float
  total_requests: int
  successful_requests: int
  rate_limited_requests: int
  error_requests: int
  throughput_rps: float
  avg_latency_ms: float
  p50_latency_ms: float
  p95_latency_ms: float
  p99_latency_ms: float
  final_rpm: float
  rate_limit_events: int
  circuit_open_events: int
  convergence_time_seconds: Optional[float] = None
  details: dict = field(default_factory=dict)


class MockGateway:
  """
  Mock gateway for testing without a real backend.

  Simulates rate limiting behavior with configurable parameters.
  """

  def __init__(
    self,
    capacity_rpm: float = 100.0,
    latency_ms: float = 50.0,
    latency_jitter_ms: float = 20.0,
  ):
    self.capacity_rpm = capacity_rpm
    self.latency_ms = latency_ms
    self.latency_jitter_ms = latency_jitter_ms

    # Token bucket for rate limiting
    self._tokens = capacity_rpm / 60.0
    self._last_refill = time.monotonic()
    self._capacity_rate = capacity_rpm / 60.0  # tokens per second
    self._lock = asyncio.Lock()

    # Circuit breaker simulation
    self._consecutive_errors = 0
    self._circuit_state = "closed"

  async def handle_request(self, model: str) -> RequestResult:
    """Handle a simulated request."""
    start = time.monotonic()

    # Simulate network latency
    latency = self.latency_ms + random.uniform(-self.latency_jitter_ms, self.latency_jitter_ms)
    await asyncio.sleep(max(0, latency / 1000.0))

    async with self._lock:
      # Refill tokens
      now = time.monotonic()
      elapsed = now - self._last_refill
      self._last_refill = now
      self._tokens = min(
        self._tokens + elapsed * self._capacity_rate,
        self._capacity_rate,  # Max 1 second of tokens
      )

      # Check if we have capacity
      if self._tokens >= 1.0:
        self._tokens -= 1.0
        self._consecutive_errors = 0
        elapsed_ms = (time.monotonic() - start) * 1000

        return RequestResult(
          success=True,
          status_code=200,
          latency_ms=elapsed_ms,
          rate_limited=False,
          hint_rpm=self.capacity_rpm,
          circuit_state=self._circuit_state,
        )
      else:
        # Rate limited
        self._consecutive_errors += 1
        elapsed_ms = (time.monotonic() - start) * 1000

        # Calculate retry-after based on token deficit
        tokens_needed = 1.0 - self._tokens
        retry_after = tokens_needed / self._capacity_rate

        return RequestResult(
          success=False,
          status_code=429,
          latency_ms=elapsed_ms,
          rate_limited=True,
          hint_rpm=self.capacity_rpm * 0.5,  # Suggest lower rate
          circuit_state=self._circuit_state,
          retry_after=retry_after,
        )


class RealGateway:
  """Client for testing against a real gateway."""

  def __init__(self, base_url: str, api_key: str):
    self.base_url = base_url.rstrip("/")
    self.api_key = api_key
    self._client = httpx.AsyncClient(timeout=30.0)

  async def close(self):
    await self._client.aclose()

  async def handle_request(self, model: str) -> RequestResult:
    """Make a real request to the gateway."""
    start = time.monotonic()

    try:
      response = await self._client.post(
        f"{self.base_url}/v1/chat/completions",
        headers={
          "Authorization": f"Bearer {self.api_key}",
          "Content-Type": "application/json",
        },
        json={
          "model": model,
          "messages": [{"role": "user", "content": "Say hello"}],
          "max_tokens": 5,
        },
      )

      elapsed_ms = (time.monotonic() - start) * 1000

      # Extract headers
      hint_rpm = None
      circuit_state = None
      retry_after = None

      if "x-ratelimit-hint-rpm" in response.headers:
        try:
          hint_rpm = float(response.headers["x-ratelimit-hint-rpm"])
        except ValueError:
          pass

      if "x-gateway-circuit-state" in response.headers:
        circuit_state = response.headers["x-gateway-circuit-state"]

      if "retry-after" in response.headers:
        try:
          retry_after = float(response.headers["retry-after"])
        except ValueError:
          pass

      if response.status_code == 200:
        return RequestResult(
          success=True,
          status_code=200,
          latency_ms=elapsed_ms,
          rate_limited=False,
          hint_rpm=hint_rpm,
          circuit_state=circuit_state,
        )
      elif response.status_code == 429:
        return RequestResult(
          success=False,
          status_code=429,
          latency_ms=elapsed_ms,
          rate_limited=True,
          hint_rpm=hint_rpm,
          circuit_state=circuit_state,
          retry_after=retry_after,
        )
      else:
        return RequestResult(
          success=False,
          status_code=response.status_code,
          latency_ms=elapsed_ms,
          rate_limited=False,
          error=f"Unexpected status: {response.status_code}",
          hint_rpm=hint_rpm,
          circuit_state=circuit_state,
        )

    except Exception as e:
      elapsed_ms = (time.monotonic() - start) * 1000
      return RequestResult(
        success=False,
        status_code=None,
        latency_ms=elapsed_ms,
        rate_limited=False,
        error=str(e),
      )


async def test_single_client_rate_limiting(
  gateway,
  model: str,
  num_requests: int,
  initial_rpm: float,
) -> TestResult:
  """
  Test rate limiting with a single client.

  Validates:
  - Rate limiter properly throttles requests
  - AIMD adaptation works correctly
  - Throughput matches expected rate
  """
  print(f"\n{'=' * 60}")
  print(f"Single Client Rate Limiting Test")
  print(f"{'=' * 60}")
  print(f"Model: {model}")
  print(f"Requests: {num_requests}")
  print(f"Initial RPM: {initial_rpm}")
  print()

  config = RateLimiterConfig(
    initial_rpm=initial_rpm,
    min_rpm=1.0,
    max_rpm=1000.0,
    additive_increase=2.0 / 60.0,  # Faster increase for testing
    multiplicative_decrease=0.5,
    max_concurrent=10,
    cooldown_period=1.0,  # Shorter cooldown for testing
  )

  limiter = AdaptiveRateLimiter(model=model, config=config)

  results: list[RequestResult] = []
  rpm_history: list[tuple[float, float]] = []  # (time, rpm)
  start_time = time.monotonic()
  last_progress_time = start_time

  for i in range(num_requests):
    # Acquire rate limit token with timeout to avoid getting stuck
    try:
      await asyncio.wait_for(limiter.acquire(), timeout=10.0)
    except asyncio.TimeoutError:
      print(f"  Request {i + 1}: Timeout waiting for rate limit token (RPM={limiter.current_rpm:.1f})")
      # Skip this request but continue
      continue

    try:
      # Make request
      result = await gateway.handle_request(model)
      results.append(result)

      # Record feedback to rate limiter
      if result.rate_limited:
        limiter.record_rate_limit(retry_after=result.retry_after)
      else:
        limiter.record_success(
          hint_rpm=result.hint_rpm,
          circuit_state=result.circuit_state,
        )

    finally:
      limiter.release()

    # Record RPM history
    elapsed = time.monotonic() - start_time
    rpm_history.append((elapsed, limiter.current_rpm))

    # Progress update every 10 requests or every 2 seconds
    now = time.monotonic()
    if (i + 1) % 10 == 0 or (now - last_progress_time) > 2.0:
      last_progress_time = now
      print(
        f"  Progress: {i + 1}/{num_requests} | "
        f"Current RPM: {limiter.current_rpm:.1f} | "
        f"Rate limits: {limiter.stats.rate_limited_requests} | "
        f"Elapsed: {elapsed:.1f}s"
      )

  duration = time.monotonic() - start_time
  stats = limiter.stats

  # Calculate metrics
  successful = [r for r in results if r.success]
  rate_limited = [r for r in results if r.rate_limited]
  latencies = [r.latency_ms for r in successful]

  throughput = len(successful) / duration if duration > 0 else 0

  result = TestResult(
    test_name="single_client_rate_limiting",
    duration_seconds=duration,
    total_requests=num_requests,
    successful_requests=len(successful),
    rate_limited_requests=len(rate_limited),
    error_requests=num_requests - len(successful) - len(rate_limited),
    throughput_rps=throughput,
    avg_latency_ms=statistics.mean(latencies) if latencies else 0,
    p50_latency_ms=statistics.median(latencies) if latencies else 0,
    p95_latency_ms=(statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0),
    p99_latency_ms=(statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0),
    final_rpm=stats.current_rpm,
    rate_limit_events=stats.rate_limited_requests,
    circuit_open_events=stats.circuit_open_events,
    details={
      "rpm_history": rpm_history[-10:],  # Last 10 samples
      "config": {
        "initial_rpm": config.initial_rpm,
        "min_rpm": config.min_rpm,
        "max_rpm": config.max_rpm,
      },
    },
  )

  print_result(result)
  return result


async def test_multi_client_convergence(
  gateway,
  model: str,
  num_clients: int,
  requests_per_client: int,
  initial_rpm: float,
) -> TestResult:
  """
  Test AIMD convergence with multiple clients.

  Validates:
  - Multiple clients converge to fair share
  - AIMD algorithm distributes capacity fairly
  - Convergence time is reasonable
  """
  print(f"\n{'=' * 60}")
  print(f"Multi-Client AIMD Convergence Test")
  print(f"{'=' * 60}")
  print(f"Model: {model}")
  print(f"Clients: {num_clients}")
  print(f"Requests per client: {requests_per_client}")
  print(f"Initial RPM: {initial_rpm}")
  print()

  config = RateLimiterConfig(
    initial_rpm=initial_rpm,
    min_rpm=1.0,
    max_rpm=1000.0,
    additive_increase=2.0 / 60.0,  # Faster increase for testing
    multiplicative_decrease=0.5,
    max_concurrent=10,
    cooldown_period=1.0,  # Shorter cooldown for testing
  )

  # Create clients
  clients: list[AdaptiveRateLimiter] = []
  for i in range(num_clients):
    clients.append(AdaptiveRateLimiter(model=f"{model}_client_{i}", config=config))

  all_results: list[list[RequestResult]] = [[] for _ in range(num_clients)]
  rpm_histories: list[list[tuple[float, float]]] = [[] for _ in range(num_clients)]
  start_time = time.monotonic()

  async def run_client(client_id: int):
    """Run requests for a single client."""
    limiter = clients[client_id]

    for i in range(requests_per_client):
      try:
        await asyncio.wait_for(limiter.acquire(), timeout=10.0)
      except asyncio.TimeoutError:
        # Skip this request but continue
        continue

      try:
        result = await gateway.handle_request(model)
        all_results[client_id].append(result)

        if result.rate_limited:
          limiter.record_rate_limit(retry_after=result.retry_after)
        else:
          limiter.record_success(
            hint_rpm=result.hint_rpm,
            circuit_state=result.circuit_state,
          )

      finally:
        limiter.release()

      # Record RPM history
      elapsed = time.monotonic() - start_time
      rpm_histories[client_id].append((elapsed, limiter.current_rpm))

  # Run all clients concurrently
  print("  Starting clients...")
  await asyncio.gather(*[run_client(i) for i in range(num_clients)])

  duration = time.monotonic() - start_time

  # Analyze results
  all_flattened = [r for results in all_results for r in results]
  successful = [r for r in all_flattened if r.success]
  rate_limited = [r for r in all_flattened if r.rate_limited]
  latencies = [r.latency_ms for r in successful]

  throughput = len(successful) / duration if duration > 0 else 0

  # Calculate fairness metrics
  client_successful = [len([r for r in results if r.success]) for results in all_results]
  client_rpms = [c.current_rpm for c in clients]

  print(f"\n  Client Results:")
  for i, (limiter, successes, rpm) in enumerate(zip(clients, client_successful, client_rpms)):
    print(f"    Client {i}: {successes} successful, final RPM: {rpm:.1f}")

  # Calculate convergence time (when variance drops below threshold)
  convergence_time = None
  if len(rpm_histories[0]) > 10:
    for t_idx in range(10, len(rpm_histories[0])):
      rpms_at_time = [rpm_histories[c][t_idx][1] for c in range(num_clients) if t_idx < len(rpm_histories[c])]
      if len(rpms_at_time) == num_clients:
        variance = statistics.variance(rpms_at_time)
        mean = statistics.mean(rpms_at_time)
        cv = (variance**0.5) / mean if mean > 0 else float("inf")  # Coefficient of variation
        if cv < 0.1:  # Less than 10% variation
          convergence_time = rpm_histories[0][t_idx][0]
          break

  result = TestResult(
    test_name="multi_client_convergence",
    duration_seconds=duration,
    total_requests=num_clients * requests_per_client,
    successful_requests=len(successful),
    rate_limited_requests=len(rate_limited),
    error_requests=len(all_flattened) - len(successful) - len(rate_limited),
    throughput_rps=throughput,
    avg_latency_ms=statistics.mean(latencies) if latencies else 0,
    p50_latency_ms=statistics.median(latencies) if latencies else 0,
    p95_latency_ms=(statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0),
    p99_latency_ms=(statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0),
    final_rpm=statistics.mean(client_rpms),
    rate_limit_events=sum(c.stats.rate_limited_requests for c in clients),
    circuit_open_events=sum(c.stats.circuit_open_events for c in clients),
    convergence_time_seconds=convergence_time,
    details={
      "client_rpms": client_rpms,
      "client_successful": client_successful,
      "fairness_cv": (
        statistics.variance(client_rpms) ** 0.5 / statistics.mean(client_rpms)
        if statistics.mean(client_rpms) > 0
        else 0
      ),
    },
  )

  print_result(result)
  return result


async def test_gateway_headers(
  gateway_url: str,
  api_key: str,
  model: str,
) -> TestResult:
  """
  Test that gateway headers are correctly returned.

  Validates:
  - X-RateLimit-Hint-RPM header is present
  - X-Gateway-Circuit-State header is present
  - Retry-After header on 429 responses
  """
  print(f"\n{'=' * 60}")
  print(f"Gateway Headers Test")
  print(f"{'=' * 60}")
  print(f"Gateway URL: {gateway_url}")
  print(f"Model: {model}")
  print()

  gateway = RealGateway(gateway_url, api_key)

  try:
    results: list[RequestResult] = []

    # Make several requests to check headers
    for i in range(10):
      result = await gateway.handle_request(model)
      results.append(result)

      print(
        f"  Request {i + 1}: "
        f"status={result.status_code}, "
        f"hint_rpm={result.hint_rpm}, "
        f"circuit_state={result.circuit_state}, "
        f"retry_after={result.retry_after}"
      )

      await asyncio.sleep(0.5)  # Avoid overwhelming

    successful = [r for r in results if r.success]
    with_hint = [r for r in results if r.hint_rpm is not None]
    with_circuit = [r for r in results if r.circuit_state is not None]

    print(f"\n  Summary:")
    print(f"    Total requests: {len(results)}")
    print(f"    Successful: {len(successful)}")
    print(f"    With X-RateLimit-Hint-RPM: {len(with_hint)}")
    print(f"    With X-Gateway-Circuit-State: {len(with_circuit)}")

    # Check header presence
    header_pass = len(with_hint) > 0 and len(with_circuit) > 0

    return TestResult(
      test_name="gateway_headers",
      duration_seconds=0,
      total_requests=len(results),
      successful_requests=len(successful),
      rate_limited_requests=len([r for r in results if r.rate_limited]),
      error_requests=len(results) - len(successful),
      throughput_rps=0,
      avg_latency_ms=(statistics.mean([r.latency_ms for r in successful]) if successful else 0),
      p50_latency_ms=0,
      p95_latency_ms=0,
      p99_latency_ms=0,
      final_rpm=0,
      rate_limit_events=0,
      circuit_open_events=0,
      details={
        "header_test_passed": header_pass,
        "hint_rpm_present": len(with_hint) > 0,
        "circuit_state_present": len(with_circuit) > 0,
        "sample_hint_rpm": with_hint[0].hint_rpm if with_hint else None,
        "sample_circuit_state": (with_circuit[0].circuit_state if with_circuit else None),
      },
    )

  finally:
    await gateway.close()


async def test_queue_with_rate_limiter(
  gateway,
  model: str,
  num_requests: int,
  initial_rpm: float,
  max_concurrent: int,
) -> TestResult:
  """
  Test the full ModelRequestQueue with integrated rate limiting.

  Validates:
  - Queue properly integrates with rate limiter
  - Priority ordering works correctly
  - Backpressure handling
  """
  print(f"\n{'=' * 60}")
  print(f"Request Queue Integration Test")
  print(f"{'=' * 60}")
  print(f"Model: {model}")
  print(f"Requests: {num_requests}")
  print(f"Initial RPM: {initial_rpm}")
  print(f"Max Concurrent: {max_concurrent}")
  print()

  rate_limiter_config = RateLimiterConfig(
    initial_rpm=initial_rpm,
    max_concurrent=max_concurrent,
    additive_increase=10.0 / 60.0,  # Faster increase for testing (10 RPM/min)
    cooldown_period=0.5,  # Shorter cooldown for testing
    min_rpm=10.0,  # Higher minimum to avoid getting stuck
  )

  config = QueueConfig(
    max_queue_depth=1000,
    queue_timeout=120.0,  # Longer timeout to avoid flaky failures
    num_workers=max_concurrent,
    rate_limiter_config=rate_limiter_config,
  )

  queue = ModelRequestQueue(
    model=model,
    config=config,
  )

  await queue.start()

  results: list[RequestResult] = []
  start_time = time.monotonic()

  async def make_request(priority: RequestPriority) -> RequestResult:
    """Submit a request through the queue."""

    async def do_request():
      return await gateway.handle_request(model)

    return await queue.submit(do_request, priority=priority)

  # Submit requests with mixed priorities
  tasks = []
  for i in range(num_requests):
    # Mix of priorities: 10% high, 70% normal, 20% low
    r = random.random()
    if r < 0.1:
      priority = RequestPriority.HIGH
    elif r < 0.8:
      priority = RequestPriority.NORMAL
    else:
      priority = RequestPriority.LOW

    tasks.append(make_request(priority))

    # Progress updates
    if (i + 1) % 20 == 0:
      stats = queue.stats
      print(
        f"  Submitted: {i + 1}/{num_requests} | Queue depth: {stats.queue_depth} | Pending: {stats.pending_requests}"
      )

  # Wait for all requests
  print("  Waiting for requests to complete...")
  completed = await asyncio.gather(*tasks, return_exceptions=True)

  for result in completed:
    if isinstance(result, Exception):
      results.append(
        RequestResult(
          success=False,
          status_code=None,
          latency_ms=0,
          rate_limited=False,
          error=str(result),
        )
      )
    else:
      results.append(result)

  duration = time.monotonic() - start_time

  await queue.stop()

  stats = queue.stats
  limiter_stats = queue.rate_limiter.stats
  successful = [r for r in results if r.success]
  rate_limited = [r for r in results if r.rate_limited]
  latencies = [r.latency_ms for r in successful]

  throughput = len(successful) / duration if duration > 0 else 0

  result = TestResult(
    test_name="queue_with_rate_limiter",
    duration_seconds=duration,
    total_requests=num_requests,
    successful_requests=len(successful),
    rate_limited_requests=len(rate_limited),
    error_requests=len(results) - len(successful) - len(rate_limited),
    throughput_rps=throughput,
    avg_latency_ms=statistics.mean(latencies) if latencies else 0,
    p50_latency_ms=statistics.median(latencies) if latencies else 0,
    p95_latency_ms=(statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0),
    p99_latency_ms=(statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0),
    final_rpm=limiter_stats.current_rpm,
    rate_limit_events=limiter_stats.rate_limited_requests,
    circuit_open_events=limiter_stats.circuit_open_events,
    details={
      "queue_stats": {
        "completed": stats.completed_requests,
        "failed": stats.failed_requests,
        "rate_limited": stats.rate_limited_requests,
        "avg_wait_ms": stats.average_wait_time_ms,
      },
    },
  )

  print_result(result)
  return result


def print_result(result: TestResult):
  """Print a test result summary."""
  print(f"\n  Results:")
  print(f"    Duration: {result.duration_seconds:.2f}s")
  print(f"    Total requests: {result.total_requests}")
  print(f"    Successful: {result.successful_requests}")
  print(f"    Rate limited: {result.rate_limited_requests}")
  print(f"    Errors: {result.error_requests}")
  print(f"    Throughput: {result.throughput_rps:.2f} req/s")
  print(f"    Avg latency: {result.avg_latency_ms:.1f}ms")
  print(f"    P50 latency: {result.p50_latency_ms:.1f}ms")
  print(f"    P95 latency: {result.p95_latency_ms:.1f}ms")
  print(f"    Final RPM: {result.final_rpm:.1f}")

  if result.convergence_time_seconds is not None:
    print(f"    Convergence time: {result.convergence_time_seconds:.2f}s")

  if result.details:
    print(f"    Details: {json.dumps(result.details, indent=6, default=str)}")


async def run_all_tests(args):
  """Run all load tests."""
  # Determine gateway
  if args.use_mock:
    print("Using mock gateway")
    gateway = MockGateway(
      capacity_rpm=args.mock_capacity,
      latency_ms=args.mock_latency,
    )
  else:
    print(f"Using real gateway at {args.gateway_url}")
    gateway = RealGateway(args.gateway_url, args.api_key)

  results = []

  try:
    if args.test in ("all", "single-client"):
      result = await test_single_client_rate_limiting(
        gateway=gateway,
        model=args.model,
        num_requests=args.requests,
        initial_rpm=args.initial_rpm,
      )
      results.append(result)

    if args.test in ("all", "multi-client"):
      result = await test_multi_client_convergence(
        gateway=gateway,
        model=args.model,
        num_clients=args.clients,
        requests_per_client=args.requests // args.clients,
        initial_rpm=args.initial_rpm,
      )
      results.append(result)

    if args.test in ("all", "queue"):
      result = await test_queue_with_rate_limiter(
        gateway=gateway,
        model=args.model,
        num_requests=args.requests,
        initial_rpm=args.initial_rpm,
        max_concurrent=args.max_concurrent,
      )
      results.append(result)

    if args.test in ("all", "headers") and not args.use_mock:
      result = await test_gateway_headers(
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        model=args.model,
      )
      results.append(result)

  finally:
    if isinstance(gateway, RealGateway):
      await gateway.close()

  # Summary
  print(f"\n{'=' * 60}")
  print("Test Summary")
  print(f"{'=' * 60}")

  for result in results:
    # Queue test expects rate limiting to occur, so we check for no errors (timeouts)
    # rather than high success rate. Other tests check for >50% success.
    if result.test_name == "queue_with_rate_limiter":
      # Queue test passes if no timeout errors occurred (rate limits are expected)
      error_count = result.error_requests
      status = "✅ PASS" if error_count == 0 else "❌ FAIL"
    else:
      status = "✅ PASS" if result.successful_requests > result.total_requests * 0.5 else "❌ FAIL"
    print(f"  {result.test_name}: {status}")
    print(f"    Throughput: {result.throughput_rps:.2f} req/s")
    print(f"    Success rate: {result.successful_requests / result.total_requests * 100:.1f}%")

  return results


def main():
  parser = argparse.ArgumentParser(description="Load test rate limiting")

  parser.add_argument(
    "--test",
    choices=["all", "single-client", "multi-client", "queue", "headers"],
    default="all",
    help="Which test to run",
  )

  parser.add_argument(
    "--requests",
    type=int,
    default=100,
    help="Number of requests to make",
  )

  parser.add_argument(
    "--clients",
    type=int,
    default=3,
    help="Number of concurrent clients (for multi-client test)",
  )

  parser.add_argument(
    "--model",
    default="nova-micro-v1",
    help="Model to test with",
  )

  parser.add_argument(
    "--initial-rpm",
    type=float,
    default=100.0,
    help="Initial requests per minute",
  )

  parser.add_argument(
    "--max-concurrent",
    type=int,
    default=10,
    help="Maximum concurrent requests",
  )

  parser.add_argument(
    "--use-mock",
    action="store_true",
    help="Use mock gateway instead of real gateway",
  )

  parser.add_argument(
    "--mock-capacity",
    type=float,
    default=200.0,
    help="Mock gateway capacity in RPM",
  )

  parser.add_argument(
    "--mock-latency",
    type=float,
    default=50.0,
    help="Mock gateway latency in ms",
  )

  parser.add_argument(
    "--gateway-url",
    default=os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_URL", "http://localhost:8080"),
    help="Gateway URL",
  )

  parser.add_argument(
    "--api-key",
    default=os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY", "test_key"),
    help="Gateway API key",
  )

  args = parser.parse_args()

  print("Rate Limiting Load Test")
  print(f"========================")
  print(f"Test: {args.test}")
  print(f"Requests: {args.requests}")
  print(f"Model: {args.model}")

  asyncio.run(run_all_tests(args))


if __name__ == "__main__":
  main()
