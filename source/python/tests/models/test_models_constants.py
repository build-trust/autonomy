import pytest

from autonomy.models.clients.litellm_client import (
  PROVIDER_ALIASES,
  ALL_PROVIDER_ALLOWED_FULL_NAMES,
  BEDROCK_INFERENCE_PROFILE_MAP,
)
from autonomy.models.clients.bedrock_client import (
  BEDROCK_MODELS,
  BEDROCK_INFERENCE_PROFILE_MAP as BEDROCK_INFERENCE_PROFILE_MAP_CLIENT,
  REQUIRES_INFERENCE_PROFILE,
)
from autonomy.models.model import MODEL_CLIENTS


class TestProviderAliases:
  """Test PROVIDER_ALIASES constant and structure."""

  def test_provider_aliases_is_dict(self):
    """Test that PROVIDER_ALIASES is a dictionary."""
    assert isinstance(PROVIDER_ALIASES, dict)
    assert len(PROVIDER_ALIASES) > 0

  def test_required_providers_present(self):
    """Test that all required providers are present."""
    required_providers = ["litellm_proxy", "bedrock", "ollama", "ollama_chat"]

    for provider in required_providers:
      assert provider in PROVIDER_ALIASES, f"Provider {provider} not found in PROVIDER_ALIASES"

  def test_provider_structure_consistency(self):
    """Test that each provider has consistent structure."""
    for provider_name, models in PROVIDER_ALIASES.items():
      assert isinstance(provider_name, str), f"Provider name {provider_name} is not a string"
      assert len(provider_name) > 0, "Provider name is empty"

      assert isinstance(models, dict), f"Models for provider {provider_name} is not a dict"
      assert len(models) > 0, f"Provider {provider_name} has no models"

      for model_alias, full_name in models.items():
        assert isinstance(model_alias, str), f"Model alias {model_alias} is not a string"
        assert isinstance(full_name, str), f"Full name {full_name} is not a string"
        assert len(model_alias) > 0, f"Model alias is empty for provider {provider_name}"
        assert len(full_name) > 0, f"Full name is empty for model {model_alias} in provider {provider_name}"

  def test_claude_models_consistency(self):
    """Test that Claude models are consistently available across providers."""
    claude_models = ["claude-3-5-haiku-v1", "claude-3-5-sonnet-v1", "claude-3-5-sonnet-v2", "claude-3-7-sonnet-v1"]

    providers_with_claude = ["litellm_proxy", "bedrock"]

    for model in claude_models:
      for provider in providers_with_claude:
        if provider in PROVIDER_ALIASES:
          assert model in PROVIDER_ALIASES[provider], f"Claude model {model} not found in provider {provider}"

  def test_llama_models_consistency(self):
    """Test that Llama models are consistently available across providers."""
    llama_models = ["llama3.2", "llama3.3"]
    providers_with_llama = ["litellm_proxy", "bedrock", "ollama", "ollama_chat"]

    for model in llama_models:
      found_providers = []
      for provider in providers_with_llama:
        if provider in PROVIDER_ALIASES and model in PROVIDER_ALIASES[provider]:
          found_providers.append(provider)

      assert len(found_providers) >= 2, (
        f"Llama model {model} found in only {len(found_providers)} providers: {found_providers}"
      )

  def test_embedding_models_present(self):
    """Test that embedding models are present."""
    embedding_models = [
      "embed-english-v3",
      "embed-multilingual-v3",
      "titan-embed-text-v1",
      "titan-embed-text-v2",
      "nomic-embed-text",
    ]

    for model in embedding_models:
      found = False
      for provider_models in PROVIDER_ALIASES.values():
        if model in provider_models:
          found = True
          break
      assert found, f"Embedding model {model} not found in any provider"

  def test_nova_models_present(self):
    """Test that Nova models are present."""
    nova_models = ["nova-lite-v1", "nova-micro-v1", "nova-pro-v1", "nova-premier-v1"]

    for model in nova_models:
      found = False
      for provider_models in PROVIDER_ALIASES.values():
        if model in provider_models:
          found = True
          break
      assert found, f"Nova model {model} not found in any provider"

  def test_deepseek_models_present(self):
    """Test that DeepSeek models are present."""
    deepseek_models = ["deepseek-r1"]

    for model in deepseek_models:
      found_providers = []
      for provider, provider_models in PROVIDER_ALIASES.items():
        if model in provider_models:
          found_providers.append(provider)

      assert len(found_providers) >= 2, (
        f"DeepSeek model {model} found in only {len(found_providers)} providers: {found_providers}"
      )

  def test_provider_specific_models(self):
    """Test that some models are provider-specific as expected."""
    # Some models should only be in specific providers
    ollama_only_models = []  # Add any Ollama-only models here if they exist
    bedrock_only_models = []  # Add any Bedrock-only models here if they exist

    for model in ollama_only_models:
      ollama_count = 0
      total_count = 0
      for provider, provider_models in PROVIDER_ALIASES.items():
        if model in provider_models:
          total_count += 1
          if "ollama" in provider:
            ollama_count += 1

      assert ollama_count == total_count, f"Model {model} should only be in Ollama providers but found in others"

  def test_model_name_formats(self):
    """Test that model names follow expected formats."""
    for provider_name, models in PROVIDER_ALIASES.items():
      for model_alias, full_name in models.items():
        # Full names should start with appropriate provider prefix
        expected_prefixes = {
          "litellm_proxy": ["litellm_proxy/"],
          "bedrock": ["bedrock/"],
          "ollama": ["ollama/", "ollama_chat/"],  # ollama can have both prefixes
          "ollama_chat": ["ollama_chat/"],
        }

        if provider_name in expected_prefixes:
          valid_prefixes = expected_prefixes[provider_name]
          assert any(full_name.startswith(prefix) for prefix in valid_prefixes), (
            f"Model {full_name} in provider {provider_name} doesn't start with any expected prefix: {valid_prefixes}"
          )

  def test_no_duplicate_aliases(self):
    """Test that there are no duplicate model aliases within providers."""
    for provider_name, models in PROVIDER_ALIASES.items():
      aliases = list(models.keys())
      unique_aliases = set(aliases)
      assert len(aliases) == len(unique_aliases), f"Provider {provider_name} has duplicate model aliases"

  def test_valid_model_identifiers(self):
    """Test that model identifiers are valid."""
    invalid_chars = [" ", "\t", "\n", "\r"]

    for provider_name, models in PROVIDER_ALIASES.items():
      for model_alias, full_name in models.items():
        # Check for invalid characters in aliases
        for char in invalid_chars:
          assert char not in model_alias, f"Model alias {model_alias} contains invalid character '{char}'"

        # Full names can have more flexible format but should not be empty
        assert full_name.strip() == full_name, f"Full name '{full_name}' has leading/trailing whitespace"


class TestAllProviderAllowedFullNames:
  """Test ALL_PROVIDER_ALLOWED_FULL_NAMES constant."""

  def test_is_set_type(self):
    """Test that ALL_PROVIDER_ALLOWED_FULL_NAMES is a set."""
    assert isinstance(ALL_PROVIDER_ALLOWED_FULL_NAMES, set)
    assert len(ALL_PROVIDER_ALLOWED_FULL_NAMES) > 0

  def test_contains_all_provider_full_names(self):
    """Test that it contains all full names from all providers."""
    expected_names = set()
    for provider_models in PROVIDER_ALIASES.values():
      expected_names.update(provider_models.values())

    assert ALL_PROVIDER_ALLOWED_FULL_NAMES == expected_names

  def test_no_duplicates(self):
    """Test that there are no duplicate entries."""
    all_names = []
    for provider_models in PROVIDER_ALIASES.values():
      all_names.extend(provider_models.values())

    # The set should have the same length as the unique list
    unique_names = list(set(all_names))
    assert len(ALL_PROVIDER_ALLOWED_FULL_NAMES) == len(unique_names)

  def test_full_names_format(self):
    """Test that full names have expected format."""
    for full_name in ALL_PROVIDER_ALLOWED_FULL_NAMES:
      assert isinstance(full_name, str)
      assert len(full_name) > 0
      assert "/" in full_name, f"Full name {full_name} should contain provider prefix with '/'"


class TestBedrockModels:
  """Test BEDROCK_MODELS constant."""

  def test_is_dict_type(self):
    """Test that BEDROCK_MODELS is a dictionary."""
    assert isinstance(BEDROCK_MODELS, dict)
    assert len(BEDROCK_MODELS) > 0

  def test_required_models_present(self):
    """Test that required Bedrock models are present."""
    required_models = ["claude-3-5-sonnet-v2", "llama3.3", "nova-pro-v1", "embed-english-v3", "deepseek-r1"]

    for model in required_models:
      assert model in BEDROCK_MODELS, f"Required Bedrock model {model} not found"

  def test_model_id_formats(self):
    """Test that Bedrock model IDs have expected formats."""
    for alias, model_id in BEDROCK_MODELS.items():
      assert isinstance(alias, str)
      assert isinstance(model_id, str)
      assert len(alias) > 0
      assert len(model_id) > 0

      # Model IDs should follow AWS Bedrock format
      if "anthropic" in model_id:
        assert model_id.startswith("anthropic."), f"Anthropic model {model_id} doesn't start with 'anthropic.'"
      elif "meta" in model_id:
        assert model_id.startswith("meta.") or model_id.startswith("us.meta."), (
          f"Meta model {model_id} doesn't start with 'meta.' or 'us.meta.'"
        )
      elif "amazon" in model_id:
        assert model_id.startswith("amazon."), f"Amazon model {model_id} doesn't start with 'amazon.'"
      elif "cohere" in model_id:
        assert model_id.startswith("cohere."), f"Cohere model {model_id} doesn't start with 'cohere.'"

  def test_claude_models_mapping(self):
    """Test Claude models mapping in Bedrock."""
    claude_models = {
      "claude-3-5-haiku-v1": "anthropic.claude-3-5-haiku-20241022-v1:0",
      "claude-3-5-sonnet-v1": "anthropic.claude-3-5-sonnet-20240620-v1:0",
      "claude-3-5-sonnet-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    }

    for alias, expected_id in claude_models.items():
      if alias in BEDROCK_MODELS:
        assert BEDROCK_MODELS[alias] == expected_id, (
          f"Claude model {alias} maps to {BEDROCK_MODELS[alias]} but expected {expected_id}"
        )

  def test_embedding_models_mapping(self):
    """Test embedding models mapping in Bedrock."""
    embedding_models = ["embed-english-v3", "embed-multilingual-v3", "titan-embed-text-v1", "titan-embed-text-v2"]

    for model in embedding_models:
      if model in BEDROCK_MODELS:
        model_id = BEDROCK_MODELS[model]
        assert "embed" in model_id or "titan" in model_id, f"Embedding model {model} has unexpected model ID {model_id}"

  def test_no_duplicate_model_ids(self):
    """Test that there are no duplicate model IDs."""
    model_ids = list(BEDROCK_MODELS.values())
    unique_ids = set(model_ids)

    if len(model_ids) != len(unique_ids):
      duplicates = []
      for model_id in unique_ids:
        if model_ids.count(model_id) > 1:
          duplicates.append(model_id)
      pytest.fail(f"Found duplicate model IDs: {duplicates}")


class TestBedrockInferenceProfiles:
  """Test Bedrock inference profile mappings."""

  def test_inference_profile_map_structure(self):
    """Test structure of inference profile mappings."""
    # Test both mappings are consistent
    for mapping in [BEDROCK_INFERENCE_PROFILE_MAP, BEDROCK_INFERENCE_PROFILE_MAP_CLIENT]:
      assert isinstance(mapping, dict)

      for original, profile in mapping.items():
        assert isinstance(original, str)
        assert isinstance(profile, str)
        assert len(original) > 0
        assert len(profile) > 0

  def test_mappings_consistency(self):
    """Test that both inference profile mappings are consistent."""
    assert BEDROCK_INFERENCE_PROFILE_MAP == BEDROCK_INFERENCE_PROFILE_MAP_CLIENT, (
      "Inference profile mappings are inconsistent between modules"
    )

  def test_us_prefix_mappings(self):
    """Test that mappings add 'us.' prefix as expected."""
    for original, profile in BEDROCK_INFERENCE_PROFILE_MAP.items():
      if not original.startswith("us."):
        assert profile.startswith("us."), f"Profile {profile} should start with 'us.' for model {original}"

  def test_required_inference_profiles(self):
    """Test that required inference profiles are mapped."""
    required_models = ["meta.llama3-3-70b-instruct-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"]

    for model in required_models:
      if model in BEDROCK_INFERENCE_PROFILE_MAP:
        profile = BEDROCK_INFERENCE_PROFILE_MAP[model]
        assert profile != model, f"Model {model} should map to a different profile"

  def test_requires_inference_profile_set(self):
    """Test REQUIRES_INFERENCE_PROFILE set."""
    assert isinstance(REQUIRES_INFERENCE_PROFILE, set)
    assert len(REQUIRES_INFERENCE_PROFILE) > 0

    for model_id in REQUIRES_INFERENCE_PROFILE:
      assert isinstance(model_id, str)
      assert len(model_id) > 0

  def test_required_profiles_have_mappings(self):
    """Test that models requiring inference profiles have mappings."""
    for model_id in REQUIRES_INFERENCE_PROFILE:
      # Not all models in REQUIRES_INFERENCE_PROFILE need to be in the mapping
      # Some might use the same ID, but if they're in the mapping, it should be valid
      if model_id in BEDROCK_INFERENCE_PROFILE_MAP:
        profile = BEDROCK_INFERENCE_PROFILE_MAP[model_id]
        assert isinstance(profile, str)
        assert len(profile) > 0


class TestModelClientsMapping:
  """Test MODEL_CLIENTS mapping."""

  def test_is_dict_type(self):
    """Test that MODEL_CLIENTS is a dictionary."""
    assert isinstance(MODEL_CLIENTS, dict)
    assert len(MODEL_CLIENTS) > 0

  def test_contains_all_aliases(self):
    """Test that MODEL_CLIENTS contains all model aliases."""
    expected_models = set()

    # Add all aliases from all providers
    for provider_models in PROVIDER_ALIASES.values():
      expected_models.update(provider_models.keys())

    # Add all full names
    expected_models.update(ALL_PROVIDER_ALLOWED_FULL_NAMES)

    # Add bedrock direct models
    for model_alias in BEDROCK_MODELS.keys():
      expected_models.add(f"bedrock-direct/{model_alias}")

    for model in expected_models:
      assert model in MODEL_CLIENTS, f"Model {model} not found in MODEL_CLIENTS"

  def test_client_type_mapping(self):
    """Test that models map to correct client types."""
    valid_clients = ["litellm", "bedrock_direct"]

    for model, client_type in MODEL_CLIENTS.items():
      assert client_type in valid_clients, f"Model {model} maps to invalid client type {client_type}"

  def test_litellm_mappings(self):
    """Test that most models map to litellm client."""
    litellm_count = 0
    bedrock_direct_count = 0

    for client_type in MODEL_CLIENTS.values():
      if client_type == "litellm":
        litellm_count += 1
      elif client_type == "bedrock_direct":
        bedrock_direct_count += 1

    assert litellm_count > 0, "No models mapped to litellm client"
    assert bedrock_direct_count > 0, "No models mapped to bedrock_direct client"

  def test_bedrock_direct_mappings(self):
    """Test bedrock-direct/ prefixed models."""
    bedrock_direct_models = [k for k in MODEL_CLIENTS.keys() if k.startswith("bedrock-direct/")]

    assert len(bedrock_direct_models) > 0, "No bedrock-direct models found"

    for model in bedrock_direct_models:
      assert MODEL_CLIENTS[model] == "bedrock_direct", (
        f"Bedrock direct model {model} should map to bedrock_direct client"
      )

      # Extract the base model name
      base_model = model.replace("bedrock-direct/", "")
      assert base_model in BEDROCK_MODELS, f"Bedrock direct model base {base_model} not found in BEDROCK_MODELS"

  def test_full_name_mappings(self):
    """Test that full model names map to litellm."""
    for full_name in ALL_PROVIDER_ALLOWED_FULL_NAMES:
      assert MODEL_CLIENTS[full_name] == "litellm", f"Full name {full_name} should map to litellm client"


class TestConstantsIntegration:
  """Test integration between different constants."""

  def test_provider_aliases_bedrock_consistency(self):
    """Test consistency between PROVIDER_ALIASES bedrock and BEDROCK_MODELS."""
    if "bedrock" in PROVIDER_ALIASES:
      bedrock_aliases = set(PROVIDER_ALIASES["bedrock"].keys())
      bedrock_models = set(BEDROCK_MODELS.keys())

      # Most models should be consistent, but some differences are acceptable
      common_models = bedrock_aliases.intersection(bedrock_models)
      assert len(common_models) > 0, "No common models between bedrock provider and BEDROCK_MODELS"

  def test_model_availability_across_constants(self):
    """Test that popular models are available across different constants."""
    popular_models = ["claude-3-5-sonnet-v2", "llama3.3"]

    for model in popular_models:
      # Should be in provider aliases
      found_in_provider = False
      for provider_models in PROVIDER_ALIASES.values():
        if model in provider_models:
          found_in_provider = True
          break
      assert found_in_provider, f"Popular model {model} not found in any provider"

      # Should be in MODEL_CLIENTS
      assert model in MODEL_CLIENTS, f"Popular model {model} not found in MODEL_CLIENTS"

      # Should be in BEDROCK_MODELS
      assert model in BEDROCK_MODELS, f"Popular model {model} not found in BEDROCK_MODELS"

  def test_embedding_model_consistency(self):
    """Test that embedding models are consistently defined."""
    embedding_models = ["embed-english-v3", "titan-embed-text-v1"]

    for model in embedding_models:
      # Should be in some provider
      found_in_provider = False
      for provider_models in PROVIDER_ALIASES.values():
        if model in provider_models:
          found_in_provider = True
          break

      if found_in_provider:
        assert model in MODEL_CLIENTS, f"Embedding model {model} not in MODEL_CLIENTS"

      # Should be in BEDROCK_MODELS if it's a Bedrock-supported model
      if any(
        "bedrock" in provider for provider in PROVIDER_ALIASES.keys() if model in PROVIDER_ALIASES.get(provider, {})
      ):
        assert model in BEDROCK_MODELS, f"Bedrock embedding model {model} not in BEDROCK_MODELS"

  def test_constants_immutability_safety(self):
    """Test that constants can be safely used without modification."""
    # These should be treated as immutable, test that they're proper containers
    assert isinstance(PROVIDER_ALIASES, dict)
    assert isinstance(ALL_PROVIDER_ALLOWED_FULL_NAMES, set)
    assert isinstance(BEDROCK_MODELS, dict)
    assert isinstance(BEDROCK_INFERENCE_PROFILE_MAP, dict)
    assert isinstance(MODEL_CLIENTS, dict)
    assert isinstance(REQUIRES_INFERENCE_PROFILE, set)

  def test_no_empty_values(self):
    """Test that there are no empty values in constants."""
    # PROVIDER_ALIASES
    for provider, models in PROVIDER_ALIASES.items():
      assert provider, "Empty provider name found"
      assert models, f"No models for provider {provider}"
      for alias, full_name in models.items():
        assert alias, f"Empty alias in provider {provider}"
        assert full_name, f"Empty full name for alias {alias} in provider {provider}"

    # BEDROCK_MODELS
    for alias, model_id in BEDROCK_MODELS.items():
      assert alias, "Empty alias in BEDROCK_MODELS"
      assert model_id, f"Empty model ID for alias {alias}"

    # MODEL_CLIENTS
    for model, client in MODEL_CLIENTS.items():
      assert model, "Empty model name in MODEL_CLIENTS"
      assert client, f"Empty client type for model {model}"


if __name__ == "__main__":
  pytest.main([__file__])
