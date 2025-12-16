"""
Tests for gateway configuration module.
"""

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
  DEFAULT_GATEWAY_URL,
  DEFAULT_GATEWAY_API_KEY,
  _FILE_CACHE_TTL_SECONDS,
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
    """Test default value for use_anthropic_sdk (should be True)."""
    with patch.dict(os.environ, {}, clear=True):
      assert use_anthropic_sdk() == True

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
