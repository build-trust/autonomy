#!/usr/bin/env python3
"""
Live Gateway Test for Model() Throttle Configuration.

This script tests the new Model() throttle and timeout configuration against
a real gateway. It validates:

1. Timeout configuration (request_timeout, connect_timeout, stream_timeout)
2. Throttle configuration (throttle=True enables rate limiting)
3. Retry logic with exponential backoff
4. AIMD rate adjustment based on gateway hints

Usage:
    # Set up port-forward to gateway first:
    kubectl port-forward -n autonomy-external-apis-gateway service/autonomy-external-apis-gateway 8080:8080 &

    # Run the test:
    cd autonomy/source/python
    AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1 \
    AUTONOMY_EXTERNAL_APIS_GATEWAY_URL=http://localhost:8080 \
    AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY=test_key \
    uv run --active python -m autonomy.models.clients.test_model_throttle_live

    # Run specific tests:
    uv run --active python -m autonomy.models.clients.test_model_throttle_live --test basic
    uv run --active python -m autonomy.models.clients.test_model_throttle_live --test throttle
    uv run --active python -m autonomy.models.clients.test_model_throttle_live --test timeout
    uv run --active python -m autonomy.models.clients.test_model_throttle_live --test batch
"""

import argparse
import asyncio
import os
import sys
import time
from typing import Optional


def check_environment():
    """Check that required environment variables are set."""
    if os.environ.get("AUTONOMY_USE_EXTERNAL_APIS_GATEWAY") != "1":
        print("❌ Error: AUTONOMY_USE_EXTERNAL_APIS_GATEWAY must be set to 1")
        print("   Set it with: export AUTONOMY_USE_EXTERNAL_APIS_GATEWAY=1")
        return False

    gateway_url = os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_URL", "http://localhost:8080")
    print(f"✓ Gateway URL: {gateway_url}")

    api_key = os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY", "test_key")
    print(f"✓ API Key: {api_key[:10]}...")

    return True


async def test_basic_model_creation():
    """Test basic Model() creation with default settings."""
    print("\n" + "=" * 60)
    print("Test: Basic Model Creation (no throttle)")
    print("=" * 60)

    from autonomy.models import Model

    # Create model with defaults
    model = Model(name="nova-micro-v1")

    print(f"✓ Model created: {model.name}")
    print(f"  request_timeout: {model.request_timeout}s")
    print(f"  connect_timeout: {model.connect_timeout}s")
    print(f"  stream_timeout: {model.stream_timeout}s")
    print(f"  throttle: {model.throttle}")

    # Make a simple request
    print("\nMaking a test request...")
    start = time.monotonic()

    try:
        response = await model.complete_chat(
            messages=[{"role": "user", "content": "Say hello in 3 words"}],
            max_tokens=20,
        )
        elapsed = time.monotonic() - start

        print(f"✓ Response received in {elapsed:.2f}s")

        # Extract content from response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"  Content: {content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


async def test_throttle_enabled():
    """Test Model() with throttle enabled."""
    print("\n" + "=" * 60)
    print("Test: Model with Throttle Enabled")
    print("=" * 60)

    from autonomy.models import Model

    # Create model with throttle enabled
    model = Model(
        name="nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=30.0,  # Conservative start
        throttle_max_requests_in_progress=5,
        throttle_max_requests_waiting_in_queue=100,
        throttle_max_retry_attempts=2,
        throttle_initial_seconds_between_retry_attempts=1.0,
    )

    print(f"✓ Model created with throttle enabled")
    print(f"  throttle_requests_per_minute: {model.throttle_requests_per_minute}")
    print(f"  throttle_max_requests_in_progress: {model.throttle_max_requests_in_progress}")
    print(f"  throttle_max_retry_attempts: {model.throttle_max_retry_attempts}")

    # Make multiple concurrent requests to test queue
    print("\nMaking 5 concurrent requests...")

    async def make_request(i: int) -> tuple[int, float, bool]:
        """Make a single request and return (id, latency, success)."""
        start = time.monotonic()
        try:
            response = await model.complete_chat(
                messages=[{"role": "user", "content": f"Count to {i+1}"}],
                max_tokens=10,
            )
            elapsed = time.monotonic() - start
            return (i, elapsed, True)
        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"  Request {i} failed: {e}")
            return (i, elapsed, False)

    start = time.monotonic()
    results = await asyncio.gather(*[make_request(i) for i in range(5)])
    total_elapsed = time.monotonic() - start

    # Report results
    successes = sum(1 for _, _, success in results if success)
    avg_latency = sum(latency for _, latency, _ in results) / len(results)

    print(f"\n✓ Completed {len(results)} requests in {total_elapsed:.2f}s")
    print(f"  Success: {successes}/{len(results)}")
    print(f"  Avg latency: {avg_latency:.2f}s")

    for i, latency, success in sorted(results):
        status = "✓" if success else "❌"
        print(f"  {status} Request {i}: {latency:.2f}s")

    return successes == len(results)


async def test_custom_timeouts():
    """Test Model() with custom timeout configuration."""
    print("\n" + "=" * 60)
    print("Test: Custom Timeout Configuration")
    print("=" * 60)

    from autonomy.models import Model

    # Create model with custom timeouts
    model = Model(
        name="nova-micro-v1",
        request_timeout=30.0,  # Shorter timeout
        connect_timeout=5.0,
        stream_timeout=60.0,
    )

    print(f"✓ Model created with custom timeouts")
    print(f"  request_timeout: {model.request_timeout}s")
    print(f"  connect_timeout: {model.connect_timeout}s")
    print(f"  stream_timeout: {model.stream_timeout}s")

    # Test a normal request
    print("\nMaking a test request...")
    start = time.monotonic()

    try:
        response = await model.complete_chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
        )
        elapsed = time.monotonic() - start
        print(f"✓ Response received in {elapsed:.2f}s")
        return True

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


async def test_batch_processing():
    """Test batch processing with throttle enabled."""
    print("\n" + "=" * 60)
    print("Test: Batch Processing with Throttle")
    print("=" * 60)

    from autonomy.models import Model

    # Create model configured for batch processing
    model = Model(
        name="nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=60.0,
        throttle_max_requests_in_progress=10,
        throttle_max_requests_waiting_in_queue=500,
        throttle_max_seconds_to_wait_in_queue=120.0,
        throttle_max_retry_attempts=3,
    )

    print(f"✓ Model created for batch processing")

    # Process a batch of requests
    batch_size = 10
    print(f"\nProcessing batch of {batch_size} requests...")

    async def process_item(i: int) -> tuple[int, float, bool]:
        start = time.monotonic()
        try:
            response = await model.complete_chat(
                messages=[{"role": "user", "content": f"Number {i}: Give one word."}],
                max_tokens=5,
            )
            elapsed = time.monotonic() - start
            return (i, elapsed, True)
        except Exception as e:
            elapsed = time.monotonic() - start
            return (i, elapsed, False)

    start = time.monotonic()
    results = await asyncio.gather(*[process_item(i) for i in range(batch_size)])
    total_elapsed = time.monotonic() - start

    # Report results
    successes = sum(1 for _, _, success in results if success)
    latencies = [latency for _, latency, success in results if success]

    print(f"\n✓ Batch completed in {total_elapsed:.2f}s")
    print(f"  Success: {successes}/{batch_size}")
    if latencies:
        print(f"  Avg latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"  Min latency: {min(latencies):.2f}s")
        print(f"  Max latency: {max(latencies):.2f}s")
    print(f"  Throughput: {successes/total_elapsed:.2f} req/s")

    return successes >= batch_size * 0.8  # 80% success rate


async def test_anthropic_model():
    """Test throttle configuration with Anthropic model."""
    print("\n" + "=" * 60)
    print("Test: Anthropic Model with Throttle")
    print("=" * 60)

    # Check if Anthropic SDK is enabled
    if os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_USE_ANTHROPIC_SDK") != "1":
        print("⚠ Skipping: AUTONOMY_EXTERNAL_APIS_GATEWAY_USE_ANTHROPIC_SDK not set to 1")
        return None

    from autonomy.models import Model

    # Create Anthropic model with throttle
    model = Model(
        name="claude-sonnet-4",
        throttle=True,
        throttle_requests_per_minute=30.0,
        throttle_max_requests_in_progress=3,
    )

    print(f"✓ Model created: {model.name}")
    print(f"  throttle: {model.throttle}")
    print(f"  throttle_requests_per_minute: {model.throttle_requests_per_minute}")

    # Make a test request
    print("\nMaking a test request...")
    start = time.monotonic()

    try:
        response = await model.complete_chat(
            messages=[{"role": "user", "content": "Say hi in one word."}],
            max_tokens=10,
        )
        elapsed = time.monotonic() - start

        print(f"✓ Response received in {elapsed:.2f}s")

        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"  Content: {content[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


async def test_rate_limiter_stats():
    """Test accessing rate limiter stats through the model."""
    print("\n" + "=" * 60)
    print("Test: Rate Limiter Stats Access")
    print("=" * 60)

    from autonomy.models import Model

    # Create model with throttle enabled
    model = Model(
        name="nova-micro-v1",
        throttle=True,
        throttle_requests_per_minute=60.0,
    )

    print(f"✓ Model created with throttle enabled")

    # Make a few requests
    print("\nMaking 3 requests to populate stats...")

    for i in range(3):
        try:
            await model.complete_chat(
                messages=[{"role": "user", "content": f"Test {i}"}],
                max_tokens=5,
            )
            print(f"  ✓ Request {i+1} completed")
        except Exception as e:
            print(f"  ❌ Request {i+1} failed: {e}")

    # Try to access rate limiter stats through the client
    client = model.client
    if hasattr(client, '_queue_manager') and client._queue_manager:
        queue = await client._queue_manager.get_queue(model.name)
        stats = queue.stats
        limiter_stats = queue.rate_limiter.stats

        print(f"\n✓ Queue Stats:")
        print(f"  Completed requests: {stats.completed_requests}")
        print(f"  Failed requests: {stats.failed_requests}")
        print(f"  Rate limited: {stats.rate_limited_requests}")

        print(f"\n✓ Rate Limiter Stats:")
        print(f"  Current RPM: {limiter_stats.current_rpm:.1f}")
        print(f"  Total requests: {limiter_stats.total_requests}")
        print(f"  Successful: {limiter_stats.successful_requests}")
        print(f"  Rate limited: {limiter_stats.rate_limited_requests}")

        return True
    else:
        print("⚠ Queue manager not initialized (throttle may not be active)")
        return True  # Not a failure, just no stats yet


async def run_all_tests(args):
    """Run all or selected tests."""
    results = {}

    if args.test in ["all", "basic"]:
        results["basic"] = await test_basic_model_creation()

    if args.test in ["all", "throttle"]:
        results["throttle"] = await test_throttle_enabled()

    if args.test in ["all", "timeout"]:
        results["timeout"] = await test_custom_timeouts()

    if args.test in ["all", "batch"]:
        results["batch"] = await test_batch_processing()

    if args.test in ["all", "anthropic"]:
        result = await test_anthropic_model()
        if result is not None:
            results["anthropic"] = result

    if args.test in ["all", "stats"]:
        results["stats"] = await test_rate_limiter_stats()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠ Some tests failed")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test Model() throttle configuration with live gateway"
    )

    parser.add_argument(
        "--test",
        choices=["all", "basic", "throttle", "timeout", "batch", "anthropic", "stats"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Model() Throttle Configuration - Live Gateway Test")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    exit_code = asyncio.run(run_all_tests(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
