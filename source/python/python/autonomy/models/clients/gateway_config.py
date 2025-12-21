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
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default gateway URL - typically runs locally during development
DEFAULT_GATEWAY_URL = "http://localhost:8000"

# Hardcoded default API key that matches the gateway's config.toml
# This allows all clients to work out-of-the-box until per-client keys are implemented
# The key "unlimited_client_key" is configured in the gateway with no model restrictions
DEFAULT_GATEWAY_API_KEY = "unlimited_client_key"

# Cache for file-based tokens to avoid reading file on every request
# Re-read interval is set to 5 minutes - well within the 1-hour refresh cycle
# and 6-hour token TTL, ensuring we always have a valid token
_FILE_CACHE_TTL_SECONDS = 300  # 5 minutes

_token_file_cache: dict = {
  "token": None,
  "read_at": 0.0,
  "file_path": None,
}


def _read_token_from_file(file_path: str) -> Optional[str]:
  """
  Read token from file, using cache to avoid excessive file I/O.

  The cache is invalidated after _FILE_CACHE_TTL_SECONDS to pick up
  refreshed tokens from the TokenRefresher.

  :param file_path: Path to the token file
  :return: Token string or None if read fails
  """
  now = time.time()

  # Check if cache is valid
  if (
    _token_file_cache["token"] is not None
    and _token_file_cache["file_path"] == file_path
    and (now - _token_file_cache["read_at"]) < _FILE_CACHE_TTL_SECONDS
  ):
    return _token_file_cache["token"]

  # Cache miss or expired - read from file
  try:
    with open(file_path, "r") as f:
      token = f.read().strip()

    if token:
      # Update cache
      _token_file_cache["token"] = token
      _token_file_cache["read_at"] = now
      _token_file_cache["file_path"] = file_path
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
  requiring container restarts.

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
