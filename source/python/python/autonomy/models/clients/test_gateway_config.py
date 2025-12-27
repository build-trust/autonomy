"""
Tests for gateway configuration module.
"""

import base64
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest

from .gateway_config import (
  get_gateway_url,
  get_gateway_api_key,
  use_anthropic_sdk,
  get_client_metadata_headers,
  clear_token_cache,
  get_token_expiration,
  get_token_time_remaining,
  _parse_jwt_expiration,
  _is_token_expiring_soon,
  DEFAULT_GATEWAY_URL,
  DEFAULT_GATEWAY_API_KEY,
  _FILE_CACHE_TTL_SECONDS,
  _TOKEN_EXPIRY_BUFFER_SECONDS,
)


class TestGatewayConfig:
  """Tests for gateway configuration functions."""

  def test_get_gateway_url_default(self):
    """Test default gateway URL when env var not set."""
    with patch.dict(os.environ, {}, clear=True):
      assert get_gateway_url() == DEFAULT_GATEWAY_URL

  def test_get_gateway_url_from_env(self):
    """Test gateway URL from environment variable."""
    custom_url = "http://custom-gateway:9000"
    with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_URL": custom_url}):
      assert get_gateway_url() == custom_url

  def test_get_gateway_url_from_file(self):
    """Test gateway URL from file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("http://file-based-gateway:8080")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_URL_FILE": f.name}, clear=True):
          assert get_gateway_url() == "http://file-based-gateway:8080"
      finally:
        os.unlink(f.name)

  def test_get_gateway_url_env_takes_precedence_over_file(self):
    """Test that direct env var takes precedence over file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("http://file-based-gateway:8080")
      f.flush()
      try:
        env = {
          "AUTONOMY_EXTERNAL_APIS_GATEWAY_URL": "http://env-gateway:9000",
          "AUTONOMY_EXTERNAL_APIS_GATEWAY_URL_FILE": f.name,
        }
        with patch.dict(os.environ, env, clear=True):
          assert get_gateway_url() == "http://env-gateway:9000"
      finally:
        os.unlink(f.name)

  def test_get_gateway_url_file_not_found(self):
    """Test fallback to default when URL file doesn't exist."""
    with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_URL_FILE": "/nonexistent/path"}, clear=True):
      assert get_gateway_url() == DEFAULT_GATEWAY_URL

  def test_get_gateway_api_key_default(self):
    """Test default gateway API key when env var not set."""
    clear_token_cache()
    with patch.dict(os.environ, {}, clear=True):
      assert get_gateway_api_key() == DEFAULT_GATEWAY_API_KEY

  def test_get_gateway_api_key_from_env(self):
    """Test gateway API key from environment variable."""
    custom_key = "custom-api-key-12345"
    with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY": custom_key}):
      assert get_gateway_api_key() == custom_key

  def test_use_anthropic_sdk_default(self):
    """Test default value for use_anthropic_sdk (should be False)."""
    with patch.dict(os.environ, {}, clear=True):
      assert use_anthropic_sdk() == False

  def test_use_anthropic_sdk_enabled(self):
    """Test use_anthropic_sdk when explicitly enabled."""
    with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_USE_ANTHROPIC_SDK": "1"}):
      assert use_anthropic_sdk() == True

  def test_use_anthropic_sdk_disabled(self):
    """Test use_anthropic_sdk when disabled."""
    with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_USE_ANTHROPIC_SDK": "0"}):
      assert use_anthropic_sdk() == False


class TestFileBasedApiKey:
  """Tests for file-based API key reading (for K8s secrets)."""

  def setup_method(self):
    """Clear token cache before each test."""
    clear_token_cache()

  def test_get_api_key_from_file(self):
    """Test reading API key from file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("file-based-token-12345")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          assert get_gateway_api_key() == "file-based-token-12345"
      finally:
        os.unlink(f.name)

  def test_get_api_key_from_file_with_whitespace(self):
    """Test reading API key from file strips whitespace."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("  token-with-whitespace  \n")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          assert get_gateway_api_key() == "token-with-whitespace"
      finally:
        os.unlink(f.name)

  def test_env_var_takes_precedence_over_file(self):
    """Test that direct env var takes precedence over file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("file-based-token")
      f.flush()
      try:
        env = {
          "AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY": "env-based-token",
          "AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name,
        }
        with patch.dict(os.environ, env, clear=True):
          assert get_gateway_api_key() == "env-based-token"
      finally:
        os.unlink(f.name)

  def test_file_not_found_falls_back_to_default(self):
    """Test fallback to default when file doesn't exist."""
    with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": "/nonexistent/path"}, clear=True):
      assert get_gateway_api_key() == DEFAULT_GATEWAY_API_KEY

  def test_empty_file_falls_back_to_default(self):
    """Test fallback to default when file is empty."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          assert get_gateway_api_key() == DEFAULT_GATEWAY_API_KEY
      finally:
        os.unlink(f.name)

  def test_token_caching(self):
    """Test that token is cached and not re-read on every call."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("initial-token")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          # First read
          assert get_gateway_api_key() == "initial-token"

          # Update file
          with open(f.name, "w") as update_f:
            update_f.write("updated-token")

          # Should still return cached value
          assert get_gateway_api_key() == "initial-token"
      finally:
        os.unlink(f.name)

  def test_cache_expiry(self):
    """Test that cache expires and token is re-read."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("initial-token")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          # First read
          assert get_gateway_api_key() == "initial-token"

          # Update file
          with open(f.name, "w") as update_f:
            update_f.write("updated-token")

          # Simulate cache expiry by manipulating the cache
          from . import gateway_config

          gateway_config._token_file_cache["read_at"] = time.time() - _FILE_CACHE_TTL_SECONDS - 1

          # Should now return updated value
          assert get_gateway_api_key() == "updated-token"
      finally:
        os.unlink(f.name)

  def test_clear_token_cache(self):
    """Test that clear_token_cache forces re-read."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write("initial-token")
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          # First read
          assert get_gateway_api_key() == "initial-token"

          # Update file
          with open(f.name, "w") as update_f:
            update_f.write("updated-token")

          # Clear cache
          clear_token_cache()

          # Should now return updated value
          assert get_gateway_api_key() == "updated-token"
      finally:
        os.unlink(f.name)

  def test_jwt_token_from_file(self):
    """Test reading a JWT-formatted token from file."""
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXpvbmUiLCJpYXQiOjE2MzA1MTY4MDAsImV4cCI6MTYzMDUzODQwMH0.signature"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write(jwt_token)
      f.flush()
      try:
        clear_token_cache()
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          assert get_gateway_api_key() == jwt_token
      finally:
        os.unlink(f.name)

  def test_different_file_paths_not_cached_together(self):
    """Test that different file paths maintain separate cache entries."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
      f1.write("token-from-file-1")
      f1.flush()
      with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
        f2.write("token-from-file-2")
        f2.flush()
        try:
          clear_token_cache()
          # Read from first file
          with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f1.name}, clear=True):
            assert get_gateway_api_key() == "token-from-file-1"

          # Reading from second file should get new value (cache invalidated by path change)
          with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f2.name}, clear=True):
            assert get_gateway_api_key() == "token-from-file-2"
        finally:
          os.unlink(f1.name)
          os.unlink(f2.name)


class TestClientMetadataHeaders:
  """Tests for get_client_metadata_headers function (Phase 8a)."""

  def test_empty_headers_when_no_env_vars(self):
    """Test that empty dict is returned when no metadata env vars are set."""
    with patch.dict(os.environ, {}, clear=True):
      headers = get_client_metadata_headers()
      assert headers == {}

  def test_cluster_header_only(self):
    """Test headers when only CLUSTER is set."""
    with patch.dict(os.environ, {"CLUSTER": "production"}, clear=True):
      headers = get_client_metadata_headers()
      assert headers == {"X-Client-Cluster": "production"}

  def test_zone_header_only(self):
    """Test headers when only ZONE is set."""
    with patch.dict(os.environ, {"ZONE": "agent-zone-1"}, clear=True):
      headers = get_client_metadata_headers()
      assert headers == {"X-Client-Zone": "agent-zone-1"}

  def test_node_header_only(self):
    """Test headers when only NODE is set."""
    with patch.dict(os.environ, {"NODE": "worker-node-5"}, clear=True):
      headers = get_client_metadata_headers()
      assert headers == {"X-Client-Node": "worker-node-5"}

  def test_all_metadata_headers(self):
    """Test headers when all metadata env vars are set."""
    env = {
      "CLUSTER": "production",
      "ZONE": "agent-zone-1",
      "NODE": "worker-node-5",
    }
    with patch.dict(os.environ, env, clear=True):
      headers = get_client_metadata_headers()
      assert headers == {
        "X-Client-Cluster": "production",
        "X-Client-Zone": "agent-zone-1",
        "X-Client-Node": "worker-node-5",
      }

  def test_cluster_and_zone_headers(self):
    """Test headers with cluster and zone (typical production setup)."""
    env = {
      "CLUSTER": "staging",
      "ZONE": "test-zone-2",
    }
    with patch.dict(os.environ, env, clear=True):
      headers = get_client_metadata_headers()
      assert headers == {
        "X-Client-Cluster": "staging",
        "X-Client-Zone": "test-zone-2",
      }

  def test_empty_string_values_not_included(self):
    """Test that empty string env vars are not included in headers."""
    env = {
      "CLUSTER": "",
      "ZONE": "valid-zone",
    }
    with patch.dict(os.environ, env, clear=True):
      headers = get_client_metadata_headers()
      # Empty string is falsy, so CLUSTER should not be included
      assert headers == {"X-Client-Zone": "valid-zone"}

  def test_headers_are_string_values(self):
    """Test that all header values are strings."""
    env = {
      "CLUSTER": "prod-123",
      "ZONE": "zone-456",
      "NODE": "node-789",
    }
    with patch.dict(os.environ, env, clear=True):
      headers = get_client_metadata_headers()
      for key, value in headers.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def _create_jwt_token(payload: dict, secret: str = "test-secret") -> str:
  """
  Create a simple JWT token for testing.
  Note: This is a simplified implementation for testing only.
  """
  header = {"alg": "HS256", "typ": "JWT"}
  header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
  payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
  # For testing, we don't need a valid signature
  signature = "test_signature"
  return f"{header_b64}.{payload_b64}.{signature}"


class TestJWTExpirationParsing:
  """Tests for JWT token expiration parsing functionality."""

  def test_parse_jwt_expiration_valid_token(self):
    """Test parsing expiration from a valid JWT token."""
    exp_time = int(time.time()) + 3600  # 1 hour from now
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time, "iat": int(time.time())})
    assert _parse_jwt_expiration(token) == exp_time

  def test_parse_jwt_expiration_no_exp_claim(self):
    """Test parsing JWT without exp claim returns None."""
    token = _create_jwt_token({"sub": "test-zone", "iat": int(time.time())})
    assert _parse_jwt_expiration(token) is None

  def test_parse_jwt_expiration_invalid_format(self):
    """Test parsing invalid token format returns None."""
    assert _parse_jwt_expiration("not-a-jwt") is None
    assert _parse_jwt_expiration("only.two") is None
    assert _parse_jwt_expiration("") is None

  def test_parse_jwt_expiration_invalid_base64(self):
    """Test parsing token with invalid base64 returns None."""
    assert _parse_jwt_expiration("header.!!!invalid!!!.signature") is None

  def test_parse_jwt_expiration_invalid_json(self):
    """Test parsing token with invalid JSON payload returns None."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
    invalid_payload = base64.urlsafe_b64encode(b"not-json").rstrip(b"=").decode()
    token = f"{header}.{invalid_payload}.signature"
    assert _parse_jwt_expiration(token) is None


class TestTokenExpiringCheck:
  """Tests for token expiring soon check functionality."""

  def test_token_not_expiring_soon(self):
    """Test token that expires in the future is not expiring soon."""
    exp_time = int(time.time()) + 3600  # 1 hour from now
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})
    assert _is_token_expiring_soon(token) is False

  def test_token_expiring_soon(self):
    """Test token that expires soon is detected."""
    exp_time = int(time.time()) + 300  # 5 minutes from now (less than 10 min buffer)
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})
    assert _is_token_expiring_soon(token) is True

  def test_token_already_expired(self):
    """Test token that has already expired is detected."""
    exp_time = int(time.time()) - 60  # 1 minute ago
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})
    assert _is_token_expiring_soon(token) is True

  def test_token_expiring_with_custom_buffer(self):
    """Test token expiring soon with custom buffer time."""
    exp_time = int(time.time()) + 120  # 2 minutes from now
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})
    # With 60 second buffer, should not be expiring soon
    assert _is_token_expiring_soon(token, buffer_seconds=60) is False
    # With 180 second buffer, should be expiring soon
    assert _is_token_expiring_soon(token, buffer_seconds=180) is True

  def test_token_without_exp_not_expiring_soon(self):
    """Test token without exp claim is not considered expiring soon."""
    token = _create_jwt_token({"sub": "test-zone", "iat": int(time.time())})
    assert _is_token_expiring_soon(token) is False

  def test_invalid_token_not_expiring_soon(self):
    """Test invalid token is not considered expiring soon."""
    assert _is_token_expiring_soon("not-a-jwt") is False


class TestTokenExpirationCacheRefresh:
  """Tests for automatic token cache refresh on expiring tokens."""

  def setup_method(self):
    """Clear token cache before each test."""
    clear_token_cache()

  def test_expiring_token_triggers_refresh(self):
    """Test that an expiring token triggers a file re-read."""
    # Create initial token that will expire soon
    exp_time_soon = int(time.time()) + 300  # 5 minutes (less than 10 min buffer)
    token_soon = _create_jwt_token({"sub": "test-zone", "exp": exp_time_soon})

    # Create updated token with longer expiry
    exp_time_long = int(time.time()) + 7200  # 2 hours
    token_long = _create_jwt_token({"sub": "test-zone", "exp": exp_time_long})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write(token_soon)
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          # First read - gets soon-expiring token
          result1 = get_gateway_api_key()
          assert result1 == token_soon

          # Update file with longer-lived token
          with open(f.name, "w") as update_f:
            update_f.write(token_long)

          # Second read - should trigger refresh because token is expiring soon
          result2 = get_gateway_api_key()
          assert result2 == token_long
      finally:
        os.unlink(f.name)

  def test_valid_token_uses_cache(self):
    """Test that a valid (not expiring) token uses cache."""
    exp_time = int(time.time()) + 7200  # 2 hours
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write(token)
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          # First read
          result1 = get_gateway_api_key()
          assert result1 == token

          # Update file
          new_token = _create_jwt_token({"sub": "test-zone-2", "exp": exp_time})
          with open(f.name, "w") as update_f:
            update_f.write(new_token)

          # Second read - should use cache since token is not expiring
          result2 = get_gateway_api_key()
          assert result2 == token  # Still cached
      finally:
        os.unlink(f.name)


class TestTokenExpirationHelpers:
  """Tests for token expiration helper functions."""

  def setup_method(self):
    """Clear token cache before each test."""
    clear_token_cache()

  def test_get_token_expiration_no_cache(self):
    """Test get_token_expiration returns None when no token cached."""
    assert get_token_expiration() is None

  def test_get_token_expiration_with_cache(self):
    """Test get_token_expiration returns exp when token is cached."""
    exp_time = int(time.time()) + 3600
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write(token)
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          get_gateway_api_key()  # Populate cache
          assert get_token_expiration() == exp_time
      finally:
        os.unlink(f.name)

  def test_get_token_time_remaining_no_cache(self):
    """Test get_token_time_remaining returns None when no token cached."""
    assert get_token_time_remaining() is None

  def test_get_token_time_remaining_with_cache(self):
    """Test get_token_time_remaining returns correct value."""
    exp_time = int(time.time()) + 3600  # 1 hour
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write(token)
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          get_gateway_api_key()  # Populate cache
          remaining = get_token_time_remaining()
          # Should be approximately 3600 seconds (allow some tolerance)
          assert remaining is not None
          assert 3590 <= remaining <= 3600
      finally:
        os.unlink(f.name)

  def test_get_token_time_remaining_expired(self):
    """Test get_token_time_remaining returns negative for expired token."""
    exp_time = int(time.time()) - 60  # Expired 1 minute ago
    token = _create_jwt_token({"sub": "test-zone", "exp": exp_time})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
      f.write(token)
      f.flush()
      try:
        with patch.dict(os.environ, {"AUTONOMY_EXTERNAL_APIS_GATEWAY_API_KEY_FILE": f.name}, clear=True):
          get_gateway_api_key()  # Populate cache
          remaining = get_token_time_remaining()
          # Should be negative since token is expired
          assert remaining is not None
          assert remaining < 0
      finally:
        os.unlink(f.name)
