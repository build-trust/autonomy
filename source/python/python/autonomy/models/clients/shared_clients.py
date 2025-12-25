"""
Shared SDK clients for gateway communication.

This module provides singleton instances of SDK clients (AsyncOpenAI, AsyncAnthropic)
to avoid creating new clients per Model/Agent, which would cause memory leaks due to
unclosed httpx connection pools.

Usage:
    from autonomy.models.clients.shared_clients import (
        get_shared_openai_client,
        get_shared_anthropic_client,
    )

    # Get the shared client (creates on first call)
    client = await get_shared_openai_client()

    # Use it for requests
    response = await client.chat.completions.create(...)
"""

import asyncio
from typing import Optional

import httpx
from openai import AsyncOpenAI

from autonomy.models.clients.gateway_config import (
  get_gateway_url,
  get_gateway_api_key,
  get_client_metadata_headers,
)
from autonomy.logs import get_logger

logger = get_logger("shared_clients")

# Shared client instances
_shared_openai_client: Optional[AsyncOpenAI] = None
_shared_openai_lock = asyncio.Lock()

_shared_anthropic_client = None  # Type: Optional[AsyncAnthropic]
_shared_anthropic_lock = asyncio.Lock()

# Default timeout configuration
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 120.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 10.0

# Connection pool limits to prevent unbounded growth
MAX_CONNECTIONS = 100
MAX_KEEPALIVE_CONNECTIONS = 20


def _create_timeout() -> httpx.Timeout:
  """Create httpx timeout configuration."""
  return httpx.Timeout(
    connect=DEFAULT_CONNECT_TIMEOUT,
    read=DEFAULT_READ_TIMEOUT,
    write=DEFAULT_WRITE_TIMEOUT,
    pool=DEFAULT_POOL_TIMEOUT,
  )


def _create_limits() -> httpx.Limits:
  """Create httpx connection pool limits."""
  return httpx.Limits(
    max_connections=MAX_CONNECTIONS,
    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
  )


async def get_shared_openai_client() -> AsyncOpenAI:
  """
  Get the shared AsyncOpenAI client instance.

  This client is shared across all GatewayClient instances to avoid
  creating multiple httpx connection pools, which would cause memory leaks.

  The client is lazily initialized on first call and reused thereafter.

  Returns:
      AsyncOpenAI: Shared client instance configured for the gateway
  """
  global _shared_openai_client

  if _shared_openai_client is not None:
    return _shared_openai_client

  async with _shared_openai_lock:
    # Double-check after acquiring lock
    if _shared_openai_client is not None:
      return _shared_openai_client

    gateway_url = get_gateway_url()
    api_key = get_gateway_api_key()
    client_metadata = get_client_metadata_headers()

    timeout = _create_timeout()
    limits = _create_limits()

    # Create a custom httpx client with limits
    http_client = httpx.AsyncClient(
      timeout=timeout,
      limits=limits,
    )

    _shared_openai_client = AsyncOpenAI(
      api_key=api_key,
      base_url=f"{gateway_url}/v1",
      timeout=timeout,
      default_headers=client_metadata if client_metadata else None,
      http_client=http_client,
    )

    logger.debug(
      f"Initialized shared OpenAI client for gateway at {gateway_url} "
      f"(max_connections={MAX_CONNECTIONS}, max_keepalive={MAX_KEEPALIVE_CONNECTIONS})"
    )

    return _shared_openai_client


async def get_shared_anthropic_client():
  """
  Get the shared AsyncAnthropic client instance.

  This client is shared across all AnthropicGatewayClient instances to avoid
  creating multiple httpx connection pools, which would cause memory leaks.

  The client is lazily initialized on first call and reused thereafter.

  Returns:
      AsyncAnthropic: Shared client instance configured for the gateway
  """
  global _shared_anthropic_client

  if _shared_anthropic_client is not None:
    return _shared_anthropic_client

  async with _shared_anthropic_lock:
    # Double-check after acquiring lock
    if _shared_anthropic_client is not None:
      return _shared_anthropic_client

    # Import here to avoid circular imports and optional dependency issues
    from anthropic import AsyncAnthropic

    gateway_url = get_gateway_url()
    api_key = get_gateway_api_key()
    client_metadata = get_client_metadata_headers()

    timeout = _create_timeout()
    limits = _create_limits()

    # Create a custom httpx client with limits
    http_client = httpx.AsyncClient(
      timeout=timeout,
      limits=limits,
    )

    _shared_anthropic_client = AsyncAnthropic(
      api_key=api_key,
      base_url=gateway_url,  # Anthropic SDK adds /v1/messages
      timeout=timeout,
      default_headers=client_metadata if client_metadata else None,
      http_client=http_client,
    )

    logger.info(
      f"Initialized shared Anthropic client for gateway at {gateway_url} "
      f"(max_connections={MAX_CONNECTIONS}, max_keepalive={MAX_KEEPALIVE_CONNECTIONS})"
    )

    return _shared_anthropic_client


async def close_shared_clients() -> None:
  """
  Close all shared clients and release resources.

  This should be called during application shutdown to properly
  close httpx connection pools.
  """
  global _shared_openai_client, _shared_anthropic_client

  if _shared_openai_client is not None:
    try:
      await _shared_openai_client.close()
      logger.info("Closed shared OpenAI client")
    except Exception as e:
      logger.warning(f"Error closing shared OpenAI client: {e}")
    _shared_openai_client = None

  if _shared_anthropic_client is not None:
    try:
      await _shared_anthropic_client.close()
      logger.info("Closed shared Anthropic client")
    except Exception as e:
      logger.warning(f"Error closing shared Anthropic client: {e}")
    _shared_anthropic_client = None


def reset_shared_clients() -> None:
  """
  Reset shared clients (for testing purposes).

  This clears the cached clients so they will be recreated on next use.
  Note: This does NOT close the clients - use close_shared_clients() for that.
  """
  global _shared_openai_client, _shared_anthropic_client
  _shared_openai_client = None
  _shared_anthropic_client = None
