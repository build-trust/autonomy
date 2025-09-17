import json
import logging
import os
import threading
from typing import List, Optional, Dict, Any, AsyncGenerator
from copy import deepcopy
import asyncio
import time

import boto3
from botocore.exceptions import ClientError

from ...logs import get_logger, InfoContext, DebugContext
from ...nodes.message import ConversationMessage


# Model mapping for Bedrock - maps friendly names to actual model IDs
BEDROCK_MODELS = {
  # Claude models
  "claude-3-5-haiku-v1": "anthropic.claude-3-5-haiku-20241022-v1:0",
  "claude-3-5-sonnet-v1": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "claude-3-5-sonnet-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "claude-3-7-sonnet-v1": "anthropic.claude-3-7-sonnet-20250219-v1:0",
  "claude-opus-4-v1": "anthropic.claude-opus-4-20250514-v1:0",
  "claude-sonnet-4-v1": "anthropic.claude-sonnet-4-20250514-v1:0",
  # Llama models
  "llama3.2": "meta.llama3-2-90b-instruct-v1:0",
  "llama3.3": "meta.llama3-3-70b-instruct-v1:0",
  "llama4-maverick": "meta.llama4-maverick-17b-instruct-v1:0",
  "llama4-scout": "meta.llama4-scout-17b-instruct-v1:0",
  # DeepSeek models
  "deepseek-r1": "us.deepseek.r1-v1:0",
  # Amazon Nova models
  "nova-lite-v1": "amazon.nova-lite-v1:0",
  "nova-micro-v1": "amazon.nova-micro-v1:0",
  "nova-pro-v1": "amazon.nova-pro-v1:0",
  "nova-premier-v1": "amazon.nova-premier-v1:0",
  # Amazon Titan models
  "titan-text-express-v1": "amazon.titan-text-express-v1",
  "titan-text-lite-v1": "amazon.titan-text-lite-v1",
  # Embedding models
  "embed-english-v3": "cohere.embed-english-v3",
  "embed-multilingual-v3": "cohere.embed-multilingual-v3",
  "titan-embed-text-v1": "amazon.titan-embed-text-v1",
  "titan-embed-text-v2": "amazon.titan-embed-text-v2:0",
  "titan-embed-image-v1": "amazon.titan-embed-image-v1",
}

# Cross-region inference profile mapping
BEDROCK_INFERENCE_PROFILE_MAP = {
  "amazon.nova-lite-v1:0": "us.amazon.nova-lite-v1:0",
  "amazon.nova-micro-v1:0": "us.amazon.nova-micro-v1:0",
  "amazon.nova-pro-v1:0": "us.amazon.nova-pro-v1:0",
  "amazon.nova-premier-v1:0": "us.amazon.nova-premier-v1:0",
  "anthropic.claude-3-7-sonnet-20250219-v1:0": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
  "anthropic.claude-opus-4-20250514-v1:0": "us.anthropic.claude-opus-4-20250514-v1:0",
  "anthropic.claude-sonnet-4-20250514-v1:0": "us.anthropic.claude-sonnet-4-20250514-v1:0",
  "meta.llama3-2-90b-instruct-v1:0": "us.meta.llama3-2-90b-instruct-v1:0",
  "meta.llama3-3-70b-instruct-v1:0": "us.meta.llama3-3-70b-instruct-v1:0",
  "meta.llama4-maverick-17b-instruct-v1:0": "us.meta.llama4-maverick-17b-instruct-v1:0",
  "meta.llama4-scout-17b-instruct-v1:0": "us.meta.llama4-scout-17b-instruct-v1:0",
  "us.deepseek.r1-v1:0": "us.deepseek.r1-v1:0",
}

# Models that REQUIRE inference profiles in certain regions
REQUIRES_INFERENCE_PROFILE = {
  "meta.llama3-3-70b-instruct-v1:0",
  "amazon.nova-lite-v1:0",
  "amazon.nova-micro-v1:0",
  "amazon.nova-pro-v1:0",
  "amazon.nova-premier-v1:0",
  "meta.llama4-maverick-17b-instruct-v1:0",
  "meta.llama4-scout-17b-instruct-v1:0",
}

# Global variables for AWS configuration
region = None
account_id = None
cluster_id = None
init_lock = threading.Lock()

_inference_profile_cache = {}
_cache_lock = threading.Lock()


def construct_bedrock_arn(model_identifier: str, original_name: str) -> Optional[str]:
  """Construct or retrieve a Bedrock inference profile ARN."""
  global region, account_id, cluster_id, init_lock

  with init_lock:
    if account_id is None:
      try:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
          session = boto3.Session()
          region = session.region_name

        sts_client = boto3.client("sts", region_name=region)
        account_id = sts_client.get_caller_identity()["Account"]

        cluster_id = os.environ.get("CLUSTER")
        if not cluster_id:
          logger = get_logger("model")
          logger.warning(
            "CLUSTER is not set. Cannot automatically manage inference profiles. Returning None."
          )
          return None

      except Exception as e:
        logger = get_logger("model")
        logger.warning(f"Could not construct Bedrock ARN: {e}")
        return None

  sanitized_model_name = original_name.replace(":", "_").replace(".", "_")
  cache_key = f"{cluster_id}_{sanitized_model_name}"

  with _cache_lock:
    if cache_key in _inference_profile_cache:
      return _inference_profile_cache[cache_key]
    else:
      inference_profile_name = f"{cluster_id}_{sanitized_model_name}"
      bedrock_client = boto3.client("bedrock", region_name=region)

      # Check if profile already exists
      try:
        paginator = bedrock_client.get_paginator("list_inference_profiles")
        for page in paginator.paginate(typeEquals="APPLICATION"):
          for profile in page.get("inferenceProfileSummaries", []):
            if profile["inferenceProfileName"] == inference_profile_name:
              arn = profile["inferenceProfileArn"]
              _inference_profile_cache[cache_key] = arn
              return arn
      except Exception as e:
        logger = get_logger("model")
        logger.warning(f"An error occurred while listing existing inference profiles: {e}")
        return None

      # Determine the source ARN for the new profile
      if model_identifier in BEDROCK_INFERENCE_PROFILE_MAP:
        source_profile_id = BEDROCK_INFERENCE_PROFILE_MAP[model_identifier]
        model_source_arn = (
          f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{source_profile_id}"
        )
      else:
        model_source_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_identifier}"

      # Create the profile
      try:
        response = bedrock_client.create_inference_profile(
          inferenceProfileName=inference_profile_name,
          modelSource={"copyFrom": model_source_arn},
          tags=[
            {"key": "ockam.ai/clusterID", "value": cluster_id},
            {"key": "ockam.ai/modelID", "value": model_identifier},
          ],
        )
        created_arn = response["inferenceProfileArn"]
        _inference_profile_cache[cache_key] = created_arn
        return created_arn
      except bedrock_client.exceptions.ConflictException:
        _inference_profile_cache[cache_key] = None
        return None
      except Exception:
        return None


class BedrockClient(InfoContext, DebugContext):
  """Direct AWS Bedrock client without LiteLLM dependency."""

  def __init__(self, name: str, max_input_tokens: Optional[int] = None, **kwargs):
    self.logger = get_logger("model")
    self.original_name = name
    self.max_input_tokens = max_input_tokens
    self.kwargs = kwargs

    # Resolve model name
    if name in BEDROCK_MODELS:
      self.model_id = BEDROCK_MODELS[name]
    else:
      # Assume it's already a full model ID
      self.model_id = name

    self.name = f"bedrock/{self.model_id}"

    # Set up AWS clients
    self.region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not self.region:
      session = boto3.Session()
      self.region = session.region_name

    self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region)
    self.bedrock = boto3.client("bedrock", region_name=self.region)

    # Handle inference profiles
    self.final_model_id = self.model_id
    inference_profile_arn = os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN")

    if inference_profile_arn:
      self.final_model_id = inference_profile_arn
    else:
      # Try to create/get inference profile
      arn = construct_bedrock_arn(self.model_id, name)
      if arn:
        self.final_model_id = arn
      elif self.model_id in REQUIRES_INFERENCE_PROFILE:
        self.logger.error(
          f"Model {self.model_id} requires an inference profile but none could be created. "
          f"Set CLUSTER environment variable or provide BEDROCK_INFERENCE_PROFILE_ARN."
        )
        raise ValueError(
          f"Model {self.model_id} requires an inference profile. "
          f"Set CLUSTER environment variable or provide BEDROCK_INFERENCE_PROFILE_ARN."
        )
      else:
        self.logger.warning(
          f"Failed to obtain/create inference profile ARN for model_identifier: {self.model_id}. Model will be called directly."
        )

    self.logger.debug(f"Initialized Bedrock client for model: {self.final_model_id}")

  def count_tokens(
    self, messages: List[dict] | List[ConversationMessage], is_thinking: bool = False, tools=None
  ) -> int:
    """Basic token counting - rough estimation."""
    normalized_messages = self._normalize_messages(messages, is_thinking)

    total_chars = 0
    for message in normalized_messages:
      content = message.get("content", "")
      if isinstance(content, str):
        total_chars += len(content)

    # Rough estimation: 1 token ≈ 4 characters for most models
    return max(1, total_chars // 4)

  def support_tools(self) -> bool:
    """Check if the model supports tool/function calling."""
    # Most Bedrock models support tools except some older ones
    return not ("deepseek" in self.model_id.lower())

  def support_forced_assistant_answer(self) -> bool:
    """Bedrock generally doesn't support forced assistant answers."""
    return False

  def _normalize_messages(
    self, messages: List[dict] | List[ConversationMessage], is_thinking: bool = False
  ) -> List[dict]:
    """Normalize messages to Bedrock format."""
    messages = deepcopy(messages)

    # Convert ConversationMessage objects to dict
    if messages and not isinstance(messages[0], dict):
      converted_messages = []
      for message in messages:
        msg_dict = {
          "role": message.role.value,
          "content": message.content.text
          if hasattr(message.content, "text")
          else str(message.content),
        }
        converted_messages.append(msg_dict)
      messages = converted_messages

    # Clean up messages
    cleaned_messages = []
    for message in messages:
      # Skip empty messages
      if not message.get("content"):
        continue

      # Bedrock uses specific role names
      role = message["role"]
      if role == "system":
        # System messages go in a separate parameter for some models
        cleaned_messages.append(message)
      elif role in ["user", "assistant"]:
        cleaned_messages.append(message)
      elif role == "tool":
        # Convert tool messages to assistant
        message["role"] = "assistant"
        cleaned_messages.append(message)

    # Handle thinking mode
    if is_thinking and cleaned_messages:
      last_message = cleaned_messages[-1]
      if last_message["role"] == "user":
        last_message["content"] = "<think>" + last_message["content"]

    return cleaned_messages

  def _prepare_bedrock_payload(
    self, messages: List[dict], stream: bool = False, **kwargs
  ) -> Dict[str, Any]:
    """Prepare the payload for Bedrock API call."""
    # Separate system messages from conversation
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    conversation_messages = [msg for msg in messages if msg["role"] != "system"]

    # Build the base payload - format depends on model family
    if "anthropic" in self.model_id:
      # Claude format
      payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": conversation_messages,
        "max_tokens": kwargs.get("max_tokens", 4000),
      }

      if system_messages:
        payload["system"] = system_messages[0]["content"]

    elif "meta.llama" in self.model_id:
      # Llama format
      payload = {
        "prompt": self._messages_to_llama_prompt(messages),
        "max_gen_len": kwargs.get("max_tokens", 4000),
      }

    elif "amazon" in self.model_id:
      # Amazon models format
      if "titan" in self.model_id and "embed" in self.model_id:
        # Embedding model
        payload = {
          "inputText": messages[-1]["content"] if messages else "",
        }
      else:
        # Text generation models
        payload = {
          "inputText": self._messages_to_prompt(messages),
          "textGenerationConfig": {
            "maxTokenCount": kwargs.get("max_tokens", 4000),
          },
        }
    else:
      # Generic format
      payload = {
        "messages": conversation_messages,
        "max_tokens": kwargs.get("max_tokens", 4000),
      }

      if system_messages:
        payload["system"] = system_messages[0]["content"]

    # Add common parameters
    if "temperature" in kwargs:
      if "anthropic" in self.model_id:
        payload["temperature"] = kwargs["temperature"]
      elif "meta.llama" in self.model_id:
        payload["temperature"] = kwargs["temperature"]
      elif "amazon" in self.model_id and "textGenerationConfig" in payload:
        payload["textGenerationConfig"]["temperature"] = kwargs["temperature"]

    if "top_p" in kwargs:
      if "anthropic" in self.model_id:
        payload["top_p"] = kwargs["top_p"]
      elif "meta.llama" in self.model_id:
        payload["top_p"] = kwargs["top_p"]

    return payload

  def _messages_to_llama_prompt(self, messages: List[dict]) -> str:
    """Convert messages to Llama chat format."""
    prompt_parts = []

    for message in messages:
      role = message["role"]
      content = message["content"]

      if role == "system":
        prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
      elif role == "user":
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
      elif role == "assistant":
        prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")

    # Add assistant start for completion
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")

    return "".join(prompt_parts)

  def _messages_to_prompt(self, messages: List[dict]) -> str:
    """Convert messages to a simple prompt format."""
    prompt_parts = []

    for message in messages:
      role = message["role"].title()
      content = message["content"]
      prompt_parts.append(f"{role}: {content}")

    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)

  def complete_chat(
    self,
    messages: List[dict] | List[ConversationMessage],
    stream: bool = False,
    is_thinking: bool = False,
    **kwargs,
  ):
    """Complete chat using direct Bedrock API."""
    self.logger.info(
      f"Processing {len(messages)} messages with Bedrock model '{self.original_name}'"
    )

    normalized_messages = self._normalize_messages(messages, is_thinking)
    payload = self._prepare_bedrock_payload(normalized_messages, stream, **kwargs)

    if stream:
      return self._complete_chat_stream(payload, is_thinking, **kwargs)
    else:
      async def _complete():
        return await self._complete_chat_non_stream(payload, **kwargs)
      return _complete()

  async def _complete_chat_non_stream(self, payload: Dict[str, Any], **kwargs):
    """Non-streaming chat completion."""
    try:
      response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: self.bedrock_runtime.invoke_model(
          modelId=self.final_model_id,
          body=json.dumps(payload),
          contentType="application/json",
        ),
      )

      response_body = json.loads(response["body"].read())

      # Parse response based on model type
      content = self._extract_content_from_response(response_body)

      # Create response object similar to OpenAI/LiteLLM format
      class Choice:
        def __init__(self, content: str):
          self.message = type(
            "Message", (), {"content": content, "role": "assistant", "reasoning_content": None}
          )()
          self.finish_reason = "stop"

      class Response:
        def __init__(self, content: str):
          self.choices = [Choice(content)]

      # Handle thinking content
      final_content = content
      thinking_content = None

      if "</think>" in content:
        end_think_tag = content.find("</think>")
        thinking_content = content[:end_think_tag].replace("<think>", "")
        final_content = content[end_think_tag + len("</think>") :].strip()

      response_obj = Response(final_content)
      if thinking_content:
        response_obj.choices[0].message.reasoning_content = thinking_content

      self.logger.info(f"Finished processing messages with Bedrock model '{self.original_name}'")
      return response_obj

    except ClientError as e:
      error_code = e.response["Error"]["Code"]
      error_message = e.response["Error"]["Message"]

      # Provide helpful error messages for common issues
      if "inference profile" in error_message.lower():
        helpful_message = (
          f"Model {self.model_id} requires an inference profile. "
          f"Set CLUSTER environment variable to enable automatic profile creation, "
          f"or provide BEDROCK_INFERENCE_PROFILE_ARN with an existing profile."
        )
        self.logger.error(f"Inference profile required: {helpful_message}")
        raise ValueError(helpful_message)
      else:
        self.logger.error(f"Bedrock API error: {error_code} - {error_message}")
        raise Exception(f"Bedrock API error: {error_code} - {error_message}")
    except Exception as e:
      self.logger.error(f"Unexpected error in Bedrock completion: {e}")
      raise

  async def _complete_chat_stream(self, payload: Dict[str, Any], is_thinking: bool, **kwargs):
    """Streaming chat completion."""
    try:
      self.logger.debug(f"Starting Bedrock streaming for model {self.model_id}")
      # Add streaming configuration
      if "anthropic" in self.model_id:
        # Claude streaming uses invoke_model_with_response_stream
        response = await asyncio.get_event_loop().run_in_executor(
          None,
          lambda: self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=self.final_model_id,
            body=json.dumps(payload),
            contentType="application/json",
          ),
        )

        async for chunk in self._process_anthropic_stream(response["body"], is_thinking):
          self.logger.debug(f"Yielding Bedrock chunk: {type(chunk)}")
          yield chunk
      else:
        # For non-Claude models that don't support streaming, fall back to non-streaming
        result = await self._complete_chat_non_stream(payload, **kwargs)

        # Simulate streaming by yielding the complete response as chunks
        content = result.choices[0].message.content
        chunk_size = max(1, len(content) // 10)  # Split into ~10 chunks

        for i in range(0, len(content), chunk_size):
          chunk_content = content[i : i + chunk_size]

          class Delta:
            def __init__(self, content: str):
              self.content = content
              self.reasoning_content = None

          class Choice:
            def __init__(self, content: str, is_final: bool = False):
              self.delta = Delta(content)
              self.finish_reason = "stop" if is_final else None

          class Chunk:
            def __init__(self, content: str, is_final: bool):
              self.choices = [Choice(content, is_final)]

          is_final = i + chunk_size >= len(content)
          yield Chunk(chunk_content, is_final)
          await asyncio.sleep(0.01)  # Small delay to simulate streaming

    except Exception as e:
      self.logger.error(f"Streaming error: {e}")
      raise

  async def _process_anthropic_stream(self, stream, is_thinking: bool):
    """Process Anthropic/Claude streaming response."""
    current_thinking = is_thinking
    chunk_count = 0

    for event in stream:
      if "chunk" in event:
        chunk = event["chunk"]
        if "bytes" in chunk:
          chunk_data = json.loads(chunk["bytes"].decode())
          chunk_count += 1
          self.logger.debug(f"Processing Bedrock event {chunk_count}: {chunk_data.get('type', 'unknown')}")

          if chunk_data.get("type") == "content_block_delta":
            delta = chunk_data.get("delta", {})
            content = delta.get("text", "")

            # Handle thinking tags
            if "<think>" in content:
              current_thinking = True
              content = content.replace("<think>", "")

            if "</think>" in content:
              current_thinking = False
              content = content.replace("</think>", "")

            # Create chunk object
            class Delta:
              def __init__(self, content: str):
                if current_thinking:
                  self.reasoning_content = content
                  self.content = None
                else:
                  self.content = content
                  self.reasoning_content = None

            class Choice:
              def __init__(self, content: str):
                self.delta = Delta(content)
                self.finish_reason = None

            class Chunk:
              def __init__(self, content: str):
                self.choices = [Choice(content)]

            if content:  # Only yield if there's actual content
              self.logger.debug(f"Yielding content chunk: {repr(content[:50])}")
              yield Chunk(content)

          elif chunk_data.get("type") == "message_stop":
            # Final chunk - send empty content with finish_reason to signal completion
            self.logger.debug("Processing message_stop - yielding finish chunk")

            class Delta:
              def __init__(self):
                self.content = ""  # Empty string instead of None
                self.reasoning_content = None

            class Choice:
              def __init__(self):
                self.delta = Delta()
                self.finish_reason = "stop"

            class Chunk:
              def __init__(self):
                self.choices = [Choice()]

            yield Chunk()

  def _extract_content_from_response(self, response_body: Dict[str, Any]) -> str:
    """Extract text content from Bedrock response based on model type."""
    if "anthropic" in self.model_id:
      # Claude response format
      content = response_body.get("content", [])
      if content and isinstance(content, list):
        return content[0].get("text", "")
      return response_body.get("completion", "")

    elif "meta.llama" in self.model_id:
      # Llama response format
      return response_body.get("generation", "")

    elif "amazon" in self.model_id:
      # Amazon models response format
      if "results" in response_body:
        results = response_body["results"]
        if results and isinstance(results, list):
          return results[0].get("outputText", "")
      return response_body.get("completion", "")

    else:
      # Try common response fields
      return (
        response_body.get("completion")
        or response_body.get("text")
        or response_body.get("output")
        or str(response_body)
      )

  async def embeddings(self, text: List[str], **kwargs) -> List[List[float]]:
    """Generate embeddings using Bedrock embedding models."""
    if "embed" not in self.model_id:
      raise ValueError(f"Model {self.model_id} is not an embedding model")

    embeddings = []

    for text_item in text:
      try:
        if "cohere" in self.model_id:
          # Cohere embedding format
          payload = {"texts": [text_item], "input_type": "search_document"}
        elif "amazon.titan" in self.model_id:
          # Titan embedding format
          payload = {"inputText": text_item}
        else:
          # Generic format
          payload = {"inputText": text_item}

        response = await asyncio.get_event_loop().run_in_executor(
          None,
          lambda: self.bedrock_runtime.invoke_model(
            modelId=self.final_model_id,
            body=json.dumps(payload),
            contentType="application/json",
          ),
        )

        response_body = json.loads(response["body"].read())

        # Extract embedding based on model type
        if "cohere" in self.model_id:
          embedding = response_body.get("embeddings", [{}])[0].get("embedding", [])
        elif "amazon.titan" in self.model_id:
          embedding = response_body.get("embedding", [])
        else:
          embedding = response_body.get("embedding", [])

        embeddings.append(embedding)

      except Exception as e:
        self.logger.error(f"Error generating embedding for text: {e}")
        # Return zero vector as fallback
        embeddings.append([0.0] * 1024)

    return embeddings

  @staticmethod
  def list_available_models() -> List[str]:
    """List all available Bedrock models."""
    return list(BEDROCK_MODELS.keys())
