"""
Gateway configuration for Autonomy clients.

This module provides centralized configuration for connecting to the
Autonomy External APIs Gateway. It defines default values that work
out-of-the-box with the standard gateway configuration.

Environment Variables:
- AUTONOMY_EXTERNAL_APIS_GATEWAY_URL: Gateway base URL (default: http://localhost:8000)
- AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY: API key for authentication (direct value)
- AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE: Path to file containing API key (for K8s secrets)
- AUTONOMY_EXTERNAL_APIS_GATEWAY_USE_ANTHROPIC_SDK: Use Anthropic SDK for Claude models (default: 1)

Client Metadata Headers (for AWS usage tracking - Phase 8a):
- CLUSTER: Cluster identifier (injected by provisioner)
- ZONE: Zone identifier (injected by provisioner)

Note: Throttling configuration (rate limiting, queuing, retries) is now configured via the
Model() constructor parameters instead of environment variables. See Model() documentation
for throttle_* parameters.

Token Resolution Order:
1. AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY env var (direct value, highest priority)
2. AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE env var (path to file, for K8s mounted secrets)
3. DEFAULT_GATEWAY_API_KEY (fallback for local development)

File-Based Tokens:
When using AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE (typically set by the ai-zone-provisioner),
the token is read from a K8s secret mounted as a volume. The file is re-read
periodically to pick up refreshed tokens. This enables automatic token rotation
without container restarts.

Token Expiration Handling:
The client proactively checks JWT token expiration and re-reads from the file
when a token is about to expire. This helps avoid 401 errors due to Kubernetes
secret sync delays.
"""

import base64
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default gateway URL - typically runs locally during development
DEFAULT_GATEWAY_URL = "http://localhost:8000"

# Default API key for testing - requires GATEWAY_TEST_MODE=true on the gateway
# This key only works when the gateway has test mode enabled (for local dev/testing)
# In production, use zone JWT tokens or configure a proper client key
DEFAULT_GATEWAY_API_KEY = "test_key"

# Cache for file-based tokens to avoid reading file on every request
# Re-read interval is set to 5 minutes - well within the 1-hour refresh cycle
# and 6-hour token TTL, ensuring we always have a valid token
_FILE_CACHE_TTL_SECONDS = 300  # 5 minutes

# Proactively refresh token if it expires within this many seconds
# This gives buffer time for K8s secret sync delays
_TOKEN_EXPIRY_BUFFER_SECONDS = 600  # 10 minutes

_token_file_cache: dict = {
  "token": None,
  "read_at": 0.0,
  "file_path": None,
  "token_exp": None,  # Cached expiration timestamp from JWT
}


def _parse_jwt_expiration(token: str) -> Optional[int]:
  """
  Parse the expiration timestamp from a JWT token without full verification.

  This is a lightweight check to determine if a token is about to expire.
  Full verification is done by the gateway.

  :param token: JWT token string
  :return: Expiration timestamp (Unix epoch seconds) or None if parsing fails
  """
  try:
    # JWT format: header.payload.signature
    parts = token.split(".")
    if len(parts) != 3:
      return None

    # Decode payload (second part) - add padding if needed
    payload_b64 = parts[1]
    # Add padding for base64url decoding
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
      payload_b64 += "=" * padding

    # Use urlsafe base64 decoding
    payload_bytes = base64.urlsafe_b64decode(payload_b64)
    payload = json.loads(payload_bytes)

    return payload.get("exp")
  except (ValueError, KeyError, json.JSONDecodeError, Exception):
    # If parsing fails, return None - we'll rely on regular cache TTL
    return None


def _is_token_expiring_soon(token: str, buffer_seconds: int = _TOKEN_EXPIRY_BUFFER_SECONDS) -> bool:
  """
  Check if a JWT token is expiring soon (within buffer_seconds).

  :param token: JWT token string
  :param buffer_seconds: Number of seconds before expiry to consider "expiring soon"
  :return: True if token expires within buffer_seconds, False otherwise
  """
  exp = _parse_jwt_expiration(token)
  if exp is None:
    # Can't determine expiration, assume it's fine
    return False

  now = time.time()
  time_until_expiry = exp - now

  if time_until_expiry <= 0:
    logger.warning(f"Token has already expired (expired {-time_until_expiry:.0f}s ago)")
    return True

  if time_until_expiry <= buffer_seconds:
    logger.info(f"Token expiring soon (in {time_until_expiry:.0f}s), will refresh from file")
    return True

  return False


def _read_token_from_file(file_path: str, force_refresh: bool = False) -> Optional[str]:
  """
  Read token from file, using cache to avoid excessive file I/O.

  The cache is invalidated after _FILE_CACHE_TTL_SECONDS to pick up
  refreshed tokens from the TokenRefresher. Additionally, if the cached
  token is about to expire, the cache is invalidated early to pick up
  a refreshed token.

  :param file_path: Path to the token file
  :param force_refresh: If True, bypass cache and read from file
  :return: Token string or None if read fails
  """
  now = time.time()

  # Check if cache is valid (unless force_refresh is requested)
  if not force_refresh:
    if (
      _token_file_cache["token"] is not None
      and _token_file_cache["file_path"] == file_path
      and (now - _token_file_cache["read_at"]) < _FILE_CACHE_TTL_SECONDS
    ):
      # Cache hit - but check if token is expiring soon
      cached_token = _token_file_cache["token"]
      if not _is_token_expiring_soon(cached_token):
        return cached_token
      else:
        # Token expiring soon, force re-read from file
        logger.debug("Cached token expiring soon, forcing file re-read")

  # Cache miss, expired, or token expiring soon - read from file
  try:
    with open(file_path, "r") as f:
      token = f.read().strip()

    if token:
      # Check if the newly read token is also expiring soon
      if _is_token_expiring_soon(token):
        # Log warning but still use it - it's the best we have
        # The K8s secret might not have been synced yet
        logger.warning(
          f"Token from file is expiring soon. "
          f"This may indicate K8s secret sync delay. "
          f"Consider restarting the pod if 401 errors persist."
        )

      # Update cache
      _token_file_cache["token"] = token
      _token_file_cache["read_at"] = now
      _token_file_cache["file_path"] = file_path
      _token_file_cache["token_exp"] = _parse_jwt_expiration(token)
      logger.debug(f"Read token from file: {file_path}")
      return token
    else:
      logger.warning(f"Token file is empty: {file_path}")
      return None

  except FileNotFoundError:
    logger.warning(f"Token file not found: {file_path}")
    return None
  except PermissionError:
    logger.warning(f"Permission denied reading token file: {file_path}")
    return None
  except (IOError, OSError) as e:
    logger.warning(f"Failed to read token from {file_path}: {e}")
    return None


def get_gateway_url() -> str:
  """
  Get the gateway URL from environment or use default.

  Checks in order:
  1. AUTONOMY_EXTERNAL_APIS_GATEWAY_URL env var
  2. AUTONOMY_EXTERNAL_APIS_GATEWAY_URL_FILE env var (path to file)
  3. DEFAULT_GATEWAY_URL

  :return: Gateway base URL (without /v1 suffix)
  """
  # Direct value takes precedence
  if url := os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_URL"):
    return url

  # File-based URL (for K8s secrets)
  if url_file := os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_URL_FILE"):
    try:
      with open(url_file, "r") as f:
        url = f.read().strip()
        if url:
          return url
    except (IOError, OSError) as e:
      logger.warning(f"Failed to read gateway URL from {url_file}: {e}")

  return DEFAULT_GATEWAY_URL


def get_gateway_api_key() -> str:
  """
  Get the gateway API key from environment, file, or use default.

  Checks in order:
  1. AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY env var (direct value)
  2. AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE env var (path to file, for K8s secrets)
  3. DEFAULT_GATEWAY_API_KEY (fallback)

  When using file-based tokens (set by ai-zone-provisioner), the token is
  cached and re-read every 5 minutes to pick up refreshed tokens without
  requiring container restarts. Additionally, if the cached token is about
  to expire (within 10 minutes), the file is re-read immediately to pick
  up any refreshed token from the provisioner.

  :return: Gateway API key
  """
  # Direct value takes precedence (useful for local development/testing)
  if api_key := os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY"):
    return api_key

  # File-based token (for K8s secrets mounted by provisioner)
  if key_file := os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE"):
    token = _read_token_from_file(key_file)
    if token:
      return token
    # Fall through to default if file read fails

  return DEFAULT_GATEWAY_API_KEY


def use_anthropic_sdk() -> bool:
  """
  Check if Anthropic SDK should be used for Claude models.

  When True, Claude models use AnthropicGatewayClient for native
  Anthropic API experience. When False (default), all models use GatewayClient
  with OpenAI-compatible API.

  :return: True if Anthropic SDK should be used
  """
  return os.environ.get("AUTONOMY_EXTERNAL_APIS_GATEWAY_USE_ANTHROPIC_SDK", "0") == "1"


def get_client_metadata_headers() -> dict:
  """
  Get client identification headers for AWS usage tracking.

  These headers are sent to the gateway and used by the BedrockProvider
  to create inference profiles tagged with cluster/zone information.
  This enables per-client usage tracking in AWS Cost Explorer and
  CloudWatch without requiring AWS credentials in client containers.

  Environment Variables:
  - CLUSTER: Cluster identifier (e.g., "production", "staging")
  - ZONE: Zone identifier (e.g., "agent-zone-1", "worker-zone-2")

  :return: Dictionary of headers to include in gateway requests
  """
  headers = {}
  if cluster := os.environ.get("CLUSTER"):
    headers["X-Client-Cluster"] = cluster
  if zone := os.environ.get("ZONE"):
    headers["X-Client-Zone"] = zone
  if node := os.environ.get("NODE"):
    headers["X-Client-Node"] = node
  return headers


def clear_token_cache() -> None:
  """
  Clear the token file cache.

  This forces the next call to get_gateway_api_key() to re-read the token
  from the file. Useful for testing or when you know the token has been
  updated.
  """
  _token_file_cache["token"] = None
  _token_file_cache["read_at"] = 0.0
  _token_file_cache["file_path"] = None
  _token_file_cache["token_exp"] = None


def get_token_expiration() -> Optional[int]:
  """
  Get the expiration timestamp of the currently cached token.

  This can be useful for debugging or monitoring token lifecycle.

  :return: Unix timestamp of token expiration, or None if no token is cached
           or expiration couldn't be determined
  """
  if _token_file_cache["token"] is None:
    return None
  return _token_file_cache.get("token_exp")


def get_token_time_remaining() -> Optional[float]:
  """
  Get the number of seconds until the currently cached token expires.

  :return: Seconds until expiration (negative if expired), or None if
           no token is cached or expiration couldn't be determined
  """
  exp = get_token_expiration()
  if exp is None:
    return None
  return exp - time.time()


# Legacy environment variable functions have been removed.
# Throttling configuration is now done via Model() constructor parameters:
# - throttle=True to enable
# - throttle_requests_per_minute=60.0
# - throttle_max_requests_in_progress=10
# - throttle_max_requests_waiting_in_queue=1000
# - throttle_max_seconds_to_wait_in_queue=300.0
# - throttle_max_retry_attempts=3
# - throttle_initial_seconds_between_retry_attempts=5.0
# - throttle_max_seconds_between_retry_attempts=60.0
