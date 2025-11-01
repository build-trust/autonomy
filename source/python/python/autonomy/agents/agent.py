import asyncio
import copy
import json
import secrets
import time
import traceback
import uuid

from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional

from ..autonomy_in_rust_for_python import warn
from ..knowledge import KnowledgeProvider
from ..knowledge.mem0 import Mem0Knowledge
from ..knowledge.noop import NoopKnowledge
from ..logs import get_logger
from ..memory.memory import Memory
from ..models.model import Model
from ..nodes.message import (
  AgentReference,
  AssistantMessage,
  ConversationMessage,
  ConversationSnippet,
  ConversationRole,
  Error,
  GetConversationsRequest,
  GetConversationsResponse,
  GetIdentifierRequest,
  GetIdentifierResponse,
  MessageConverter,
  Phase,
  StreamedConversationSnippet,
  ToolCall,
  ToolCallResponseMessage,
)
from ..nodes.node import Node
from ..planning import Planner
from ..tools.protocol import InvokableTool
from ..helpers.validate_address import validate_address

logger = get_logger("agent")


class AgentState(Enum):
  """
  States for the agent conversation state machine.

  - INIT: Initial state where we set up the conversation and initialize planning if available
  - PLANNING: Create the next step of the plan
  - MODEL_CALLING: Call the model to generate a response
  - TOOL_CALLING: Execute tool calls from the model response
  - FINISHED: End of the conversation
  """

  INIT = "init"
  PLANNING = "planning"
  MODEL_CALLING = "model_calling"
  TOOL_CALLING = "tool_calling"
  FINISHED = "finished"


#     ┌─────────┐
#     │         │
#     │  INIT   │
#     │         │
#     └───┬─────┘
#         │
#         ├─────────────────────────┐
#         │                         │
#         │ [planner available]     │ [planner not available]
#         ▼                         ▼
#     ┌─────────┐               ┌──────────┐
#     │         │               │          │
#     │PLANNING │◀──────────────┤ MODEL    │◀─────┐
#     │         ├──[execute]───▶│ CALLING  │      │
#     └───┬─────┘               └┬────┬────┘      │
#         │                      │    │           │
#         │                      │    │           │
#         │ [plan completed]     │    │           │
#         │                      │    │           │
#         │                      │    │           │
#         │                      │    ▼           │
#         │                      │  ┌────────┐    │
#         │                      │  │        │    │
#         │                      │  │  TOOL  │────┘
#         │                      │  │CALLING │
#         │                      │  └────────┘
#         │                      │
#         │                      │ [response complete && no tool calls && planner available]
#         │                      │
#         ▼                      │
#     ┌─────────┐                │ [response complete && no tool calls && no planner]
#     │         │                │ [error conditions]
#     │FINISHED │◀───────────────┘ [max iterations reached]
#     │         │
#     └─────────┘
#
# State Transition Conditions:
#
# MODEL_CALLING transitions:
# - → TOOL_CALLING: [response contains tool calls]
# - → PLANNING: [response complete && no tool calls && planner available]
# - → FINISHED: [response complete && no tool calls && no planner] OR [error conditions] OR [max iterations reached]
#
# PLANNING transitions:
# - → MODEL_CALLING: [execute next step]
# - → FINISHED: [plan completed] OR [error conditions] OR [max planning transitions]
#
# TOOL_CALLING transitions:
# - → MODEL_CALLING: [after processing all tool calls] OR [error conditions] OR [max tool calling transitions]
#
# Error conditions include: model completion errors, timeout exceeded, transition limits exceeded
#


class AgentStateMachine:
  """
  State machine for managing agent conversation flow.

  This class implements a state machine that manages the flow of an agent conversation.
  It handles the transitions between different states and executes the appropriate actions
  for each state. The state machine is designed to be used by the `handle__conversation_snippet`
  method of the `Agent` class.
  """

  def __init__(self, agent, scope, conversation, stream, response, contextual_knowledge=None):
    self.agent = agent
    self.scope = scope
    self.conversation = conversation
    self.stream = stream
    self.streaming_response = response
    self.contextual_knowledge = contextual_knowledge
    self.state = AgentState.INIT
    self.plan = None
    self.iteration = 0
    self.tool_calls = []
    self.whole_response = ConversationSnippet(scope, conversation, [])
    self.final_content_sent = False

    # Enhanced loop protection
    self.total_transitions = 0
    self.planning_transitions = 0
    self.tool_calling_transitions = 0
    self.start_time = time.time()

    # Configurable limits with sensible defaults
    self.max_total_transitions = getattr(agent, "max_total_transitions", 500)
    self.max_planning_transitions = getattr(agent, "max_planning_transitions", 50)
    self.max_tool_calling_transitions = getattr(agent, "max_tool_calling_transitions", 100)
    self.max_execution_time = getattr(agent, "max_execution_time", 300.0)  # 5 minutes

  async def transition(self):
    """Execute the current state and determine the next state."""

    # Comprehensive loop protection
    self.total_transitions += 1
    current_time = time.time()
    elapsed_time = current_time - self.start_time

    # Check total transition limit
    if self.total_transitions > self.max_total_transitions:
      logger.error(f"State machine exceeded maximum total transitions ({self.max_total_transitions})")
      yield Error(f"Agent exceeded maximum state transitions ({self.max_total_transitions})")
      self.state = AgentState.FINISHED
      return

    # Check total execution time limit
    if elapsed_time > self.max_execution_time:
      logger.error(f"State machine exceeded maximum execution time ({self.max_execution_time}s)")
      yield Error(f"Agent exceeded maximum execution time ({self.max_execution_time:.1f}s)")
      self.state = AgentState.FINISHED
      return

    # Log transition for debugging
    logger.debug(
      f"State transition #{self.total_transitions}: {self.state.name} "
      f"(planning: {self.planning_transitions}, tool_calling: {self.tool_calling_transitions}, "
      f"iteration: {self.iteration}, elapsed: {elapsed_time:.1f}s)"
    )

    match self.state:
      case AgentState.INIT:
        if self.agent.planner is not None:
          self.state = AgentState.PLANNING
        else:
          self.state = AgentState.MODEL_CALLING
      case AgentState.PLANNING:
        async for result in self._handle_planning_state():
          yield result
      case AgentState.MODEL_CALLING:
        async for result in self._handle_model_calling_state():
          yield result
      case AgentState.TOOL_CALLING:
        async for result in self._handle_tool_calling_state():
          yield result

  async def _handle_planning_state(self):
    """Handle the PLANNING state: Execute the next steps from the plan."""

    # Track planning transitions and check limits
    self.planning_transitions += 1

    if self.planning_transitions > self.max_planning_transitions:
      logger.error(f"Planning exceeded maximum transitions ({self.max_planning_transitions})")
      yield Error(f"Planning phase exceeded maximum transitions ({self.max_planning_transitions})")
      self.state = AgentState.FINISHED
      return

    plan_completed = True
    next_steps = self.plan.next_step(
      await self.agent.get_messages_only(self.conversation, self.scope), self.contextual_knowledge
    )

    async for next_step in next_steps:
      if next_step is None:
        break
      plan_completed = False
      if next_step.phase == Phase.PLANNING:
        await self.agent.remember(self.scope, self.conversation, next_step)
      if self.stream:
        yield await self.streaming_response.make_snippet(next_step)
      else:
        self.whole_response.messages.append(next_step)

    if plan_completed:
      # We executed the whole plan
      self.state = AgentState.FINISHED
    else:
      # The model will execute the next plan step
      self.state = AgentState.MODEL_CALLING

  async def _handle_model_calling_state(self):
    """Handle the MODEL_CALLING state: Call the model to generate a response."""

    self.iteration += 1
    if self.iteration >= self.agent.maximum_iterations:
      yield Error(str(RuntimeError("Reached maximum_iterations")))
      self.state = AgentState.FINISHED
      return

    self.tool_calls = []
    response_received = False
    finish_signal_received = False
    chunks_processed = 0

    logger.debug(f"Starting model completion, iteration {self.iteration}")

    try:
      async for err, finished, model_response in self.agent.complete_chat(
        self.scope, self.conversation, self.contextual_knowledge, stream=self.stream
      ):
        chunks_processed += 1
        logger.debug(
          f"Received chunk {chunks_processed}: err={err}, finished={finished}, response_type={type(model_response)}"
        )

        if err:
          logger.error(f"Model completion error: {err}")
          yield err
          self.state = AgentState.FINISHED
          return

        response_received = True

        # Remember the model response
        await self.agent.remember(self.scope, self.conversation, model_response)

        # Handle tool calls
        if len(model_response.tool_calls) > 0:
          if self.stream:
            yield await self.streaming_response.make_snippet(model_response)
          else:
            self.whole_response.messages.append(model_response)
          self.tool_calls.extend(model_response.tool_calls)
          self.state = AgentState.TOOL_CALLING
          if finished:
            finish_signal_received = True
            return

        # Handle finished responses
        if finished:
          finish_signal_received = True
          logger.debug(f"Processing finished response: stream={self.stream}")

          if self.stream:
            if model_response.content or model_response.tool_calls:
              yield await self.streaming_response.make_snippet(model_response, finished=True)
            self.final_content_sent = True
          else:
            self.whole_response.messages.append(model_response)

          # Determine next state
          if self.state == AgentState.TOOL_CALLING:
            logger.debug("Transitioning to TOOL_CALLING")
            return
          elif self.agent.planner is None:
            logger.debug("Transitioning to FINISHED (no planner)")
            self.state = AgentState.FINISHED
          else:
            logger.debug("Transitioning to PLANNING")
            self.state = AgentState.PLANNING
          return
        else:
          # Intermediate streaming chunk
          if self.stream and (model_response.content or model_response.tool_calls):
            yield await self.streaming_response.make_snippet(model_response)
          elif not self.stream:
            self.whole_response.messages.append(model_response)

          # Prevent infinite streaming - max 1000 chunks
          if chunks_processed >= 1000:
            logger.warning(f"Streaming exceeded max chunks ({chunks_processed}), forcing finish")
            if self.stream and not self.final_content_sent:
              # Send empty finish chunk if needed
              from ..nodes.message import AssistantMessage

              empty_response = AssistantMessage(content="", phase=Phase.EXECUTING)
              yield await self.streaming_response.make_snippet(empty_response, finished=True)
              self.final_content_sent = True
            break

    except Exception as e:
      logger.error(f"Exception in model calling: {e}")
      yield Error(f"Model completion failed: {str(e)}")
      self.state = AgentState.FINISHED
      return

    # Post-processing: Handle incomplete streaming
    logger.debug(
      f"Stream complete: chunks={chunks_processed}, response_received={response_received}, finish_signal={finish_signal_received}"
    )

    if not response_received:
      logger.error("No response received from model")
      yield Error("No response received from model")
      self.state = AgentState.FINISHED
      return

    # Handle incomplete streaming (no finish signal received)
    if response_received and not finish_signal_received:
      logger.warning(f"Stream incomplete - no finish signal after {chunks_processed} chunks, forcing completion")

      if self.stream and not self.final_content_sent:
        # Send empty finish chunk to complete the stream
        from ..nodes.message import AssistantMessage

        empty_response = AssistantMessage(content="", phase=Phase.EXECUTING)
        yield await self.streaming_response.make_snippet(empty_response, finished=True)
        self.final_content_sent = True

      # Force state transition
      if self.state == AgentState.MODEL_CALLING:
        if self.agent.planner is None:
          self.state = AgentState.FINISHED
        else:
          self.state = AgentState.PLANNING

    # Final safety check
    if self.state == AgentState.MODEL_CALLING:
      logger.warning(f"Forcing FINISHED state after {chunks_processed} chunks")
      self.state = AgentState.FINISHED

  async def _handle_tool_calling_state(self):
    """Handle the TOOL_CALLING state: Process tool calls from the model response."""

    # Track tool calling transitions and check limits
    self.tool_calling_transitions += 1

    if self.tool_calling_transitions > self.max_tool_calling_transitions:
      logger.error(f"Tool calling exceeded maximum transitions ({self.max_tool_calling_transitions})")
      yield Error(f"Tool calling phase exceeded maximum transitions ({self.max_tool_calling_transitions})")
      self.state = AgentState.FINISHED
      return

    for tool_call in self.tool_calls:
      error, tool_call_response = await self.agent.call_tool(tool_call)
      if error:
        warn(f"MCP call failed: {error}")

      # the tool_call_response contains the error message if the tool call failed
      await self.agent.remember(self.scope, self.conversation, tool_call_response)
      if self.stream:
        yield await self.streaming_response.make_snippet(tool_call_response)
      else:
        self.whole_response.messages.append(tool_call_response)

    # after processing tool calls, we need another model call
    self.state = AgentState.MODEL_CALLING

  async def _handle_finished_state(self):
    """Handle the FINISHED state: End the conversation and return the final response."""

    if self.stream:
      # Only send finished snippet if we haven't already sent final content
      if not self.final_content_sent:
        yield await self.streaming_response.make_finished_snippet()
    else:
      yield self.whole_response

  async def initialize_plan(self, messages):
    """Initialize the plan if a planner is available and remember messages."""

    if self.agent.planner is None:
      for message in messages:
        await self.agent.remember(self.scope, self.conversation, message)
    else:
      # When a planner is provided, the input messages will be handled solely by the planner
      self.plan = await self.agent.planner.plan(messages, self.contextual_knowledge, self.stream)

  async def run(self):
    """Run the state machine until completion."""

    try:
      # Add overall timeout protection for the entire state machine run
      async with asyncio.timeout(self.max_execution_time):
        while self.state != AgentState.FINISHED:
          # Additional safety check to prevent infinite loops
          if self.total_transitions > self.max_total_transitions:
            logger.error("State machine hit transition limit in run loop")
            break

          async for result in self.transition():
            yield result

        async for result in self._handle_finished_state():
          yield result

    except asyncio.TimeoutError:
      logger.error(f"State machine run timed out after {self.max_execution_time}s")
      yield Error(f"Agent execution timed out after {self.max_execution_time:.1f} seconds")

    # Log final statistics
    elapsed_time = time.time() - self.start_time
    logger.info(
      f"State machine completed: {self.total_transitions} total transitions "
      f"(planning: {self.planning_transitions}, tool_calling: {self.tool_calling_transitions}) "
      f"in {elapsed_time:.1f}s"
    )


class Agent:
  tools: Dict[str, InvokableTool]
  memory_knowledge: Optional["AgentMemoryKnowledge"]

  def __init__(
    self,
    node: Node,
    name: str,
    instructions: str,
    model: Model,
    memory_model: Optional[Model],
    memory_embeddings_model: Optional[Model],
    tool_specs: List[dict],
    tools: Dict[str, InvokableTool],
    planner: Optional[Planner],
    memory: Memory,
    knowledge: KnowledgeProvider,
    maximum_iterations: int,
    max_total_transitions: int = 500,
    max_planning_transitions: int = 50,
    max_tool_calling_transitions: int = 100,
    max_execution_time: float = 300.0,
  ):
    logger.info(f"Starting agent '{name}'")
    self.node = node

    self.tools = tools
    self.tool_specs = tool_specs
    logger.info(f"Started agent '{name}'")

    self.name = name
    self.model = model
    self.memory_model = memory_model
    self.memory_embeddings_model = memory_embeddings_model
    self.memory_knowledge = None

    self.memory = memory
    memory.set_instructions(system_message(instructions))

    self.planner = planner
    self.maximum_iterations = maximum_iterations

    # Enhanced loop protection settings
    self.max_total_transitions = max_total_transitions
    self.max_planning_transitions = max_planning_transitions
    self.max_tool_calling_transitions = max_tool_calling_transitions
    self.max_execution_time = max_execution_time

    self.knowledge = knowledge

    self.converter = MessageConverter.create(node)

    # Add async lock for memory synchronization
    self._memory_lock = asyncio.Lock()

    # Add tool call ID tracking to prevent collisions
    self._tool_call_ids = OrderedDict()
    self._tool_call_id_lock = asyncio.Lock()

  async def handle_message(self, context, message):
    try:
      logger.debug(f"Agent '{self.name}' received: {message}")

      message = self.converter.message_from_json(message)
      handlers = {
        ConversationSnippet: self.handle__conversation_snippet,
        StreamedConversationSnippet: self.handle__conversation_snippet,
        GetIdentifierRequest: self.handle__get_identifier_request,
        GetConversationsRequest: self.handle__get_conversations_request,
      }

      handler = handlers.get(type(message))
      if type(message) is ConversationSnippet or type(message) is StreamedConversationSnippet:
        async for reply in handler(message):
          if reply is not None:
            await context.reply(self.converter.message_to_json(reply))
      else:
        if handler is not None:
          reply = await handler(message)
        else:
          logger.error(f"Unexpected message: {message}")
          reply = Error(f"Unexpected Message: {message}")

        if reply is not None:
          await context.reply(self.converter.message_to_json(reply))
    except Exception as e:
      traceback.print_exc()
      error = Error(str(e))
      await context.reply(self.converter.message_to_json(error))

  async def handle__get_identifier_request(self, message: GetIdentifierRequest) -> GetIdentifierResponse:
    name_snake_case = self.name.lower().replace(" ", "_")
    return GetIdentifierResponse(message.scope, message.conversation, name_snake_case)

  async def handle__get_conversations_request(self, message: GetConversationsRequest) -> GetConversationsResponse:
    async with self._memory_lock:
      return GetConversationsResponse(self.memory.get_messages_only(message.scope, message.conversation))

  async def handle__conversation_snippet(self, message):
    """Handle conversation snippets from users or other agents."""

    stream = type(message) is StreamedConversationSnippet
    scope = message.scope if type(message) is ConversationSnippet else message.snippet.scope
    conversation = message.conversation if type(message) is ConversationSnippet else message.snippet.conversation
    messages = message.messages if type(message) is ConversationSnippet else message.snippet.messages

    # create the streaming response handler
    response = ConversationResponse(scope, conversation, stream)

    # if we have memory/knowledge capabilities, search for relevant information
    contextual_knowledge = None
    if self.memory_knowledge is None and self.memory_model is not None and self.memory_embeddings_model is not None:
      self.memory_knowledge = await AgentMemoryKnowledge.create(self.memory_model, self.memory_embeddings_model)

    if self.memory_knowledge is not None and len(messages) > 0:
      user_message = messages[-1]
      if user_message.role == ConversationRole.USER and hasattr(user_message.content, "text"):
        contextual_knowledge = await self.memory_knowledge.search(scope, conversation, user_message.content.text)

    # create the state machine to handle the conversation
    machine = AgentStateMachine(self, scope, conversation, stream, response, contextual_knowledge)

    # initialize the plan with input messages
    await machine.initialize_plan(messages)

    # run the state machine until completion
    async for result in machine.run():
      yield result

  async def get_messages_only(self, conversation, scope) -> list[ConversationMessage]:
    # Synchronize memory reads to ensure consistency
    async with self._memory_lock:
      messages: list[dict] = self.memory.get_messages_only(scope, conversation)
      return [self.converter.conversation_message_from_dict(m) for m in messages]

  async def remember(self, scope: str, conversation: str, message: ConversationMessage):
    # TODO: at some point memory becomes bigger than context window and we need to start pushing memories
    # into Knowledge
    # self.history_knowledge.add(messages=[model_response], user_id=scope)

    # Use async lock to ensure atomic operations across concurrent conversations
    async with self._memory_lock:
      # Convert message to dict format if needed
      if not isinstance(message, dict):
        message = self.converter.message_to_dict(message)

      # Add to memory atomically
      self.memory.add_message(scope, conversation, message)

  async def message_history(self, scope: str, conversation: str) -> list[dict]:
    # Synchronize memory reads to ensure consistency with writes
    async with self._memory_lock:
      return self.memory.get_messages(scope, conversation)

  async def determine_input_context(self, scope: str, conversation: str):
    """Determine the input context for the agent based on memory and knowledge."""

    # Get the message history as ConversationMessage objects (not raw dicts)
    async with self._memory_lock:
      raw_messages = self.memory.get_messages(scope, conversation)
      messages = [self.converter.conversation_message_from_dict(m) for m in raw_messages]

    # Add contextual information from knowledge if available
    if self.knowledge is not None and len(messages) > 0:
      # Use the last user message to search for relevant knowledge
      last_user_message = None
      for msg in reversed(messages):
        if msg.role == ConversationRole.USER:
          last_user_message = msg
          break

      if last_user_message and hasattr(last_user_message.content, "text"):
        contextual_knowledge = await self.knowledge.search_knowledge(
          scope, conversation, last_user_message.content.text
        )
        if contextual_knowledge:
          # Add contextual knowledge as a system message
          from ..nodes.message import SystemMessage

          context_message = SystemMessage(f"Relevant context: {contextual_knowledge}")
          messages.append(context_message)

    return messages

  async def complete_chat(
    self, scope: str, conversation: str, contextual_knowledge: Optional[str] = None, stream: bool = False
  ):
    """Complete a chat conversation using the agent's model."""

    try:
      messages = await self.determine_input_context(scope, conversation)

      # Add contextual knowledge if provided
      if contextual_knowledge:
        from ..nodes.message import SystemMessage

        context_message = SystemMessage(f"Relevant context: {contextual_knowledge}")
        # Insert before the last message, or append if messages is empty or None
        if messages:
          messages.insert(-1, context_message)  # Insert before the last message (usually user input)
        else:
          # Handle case where messages is None or empty
          if messages is None:
            messages = []
          messages.append(context_message)  # Append if messages is empty

      # Call the model to generate a response
      if stream:
        # Enhanced streaming with timeout and better error handling
        import asyncio

        chunk_count = 0
        has_content = False
        last_finished = False
        max_chunks = 1000  # Prevent infinite streams
        timeout_seconds = 300  # 5 minutes max per stream

        logger.debug(f"Starting streaming model call with {len(messages)} messages")

        try:
          # Execute streaming with timeout protection
          async with asyncio.timeout(timeout_seconds):
            async for chunk in self.model.complete_chat(messages, stream=True, tools=self.tool_specs):
              chunk_count += 1
              logger.debug(f"Processing streaming chunk {chunk_count}")

              # Prevent runaway streams
              if chunk_count > max_chunks:
                logger.error(f"Stream exceeded maximum chunks ({max_chunks}), terminating")
                break

              # Process streaming response chunks
              if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                finished = choice.finish_reason is not None if hasattr(choice, "finish_reason") else False
                last_finished = finished

                # Handle content chunks
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                  content = choice.delta.content
                  logger.debug(f"Delta content: {repr(content)}, finished: {finished}")

                  if content:  # Non-empty content
                    has_content = True
                    response = AssistantMessage(content=content, phase=Phase.EXECUTING)
                    logger.debug(f"Yielding content chunk: finished={finished}")
                    yield None, finished, response
                  elif finished and (content == "" or content is None):
                    # Empty content with finish_reason - this is a termination chunk
                    logger.debug("Received empty finish chunk - streaming complete")
                    has_content = True
                    # Send empty response to signal completion
                    response = AssistantMessage(content="", phase=Phase.EXECUTING)
                    yield None, True, response
                    break

                # Handle tool call chunks
                elif hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                  tool_calls = []
                  for tc in choice.delta.tool_calls:
                    # Generate unique ID with collision detection
                    call_id = tc.id if hasattr(tc, "id") and tc.id else None
                    if not call_id:
                      call_id = await self._generate_unique_tool_call_id()

                    tool_call = ToolCall(
                      id=call_id,
                      function=tc.function if hasattr(tc, "function") else None,
                      type="function",
                    )
                    tool_calls.append(tool_call)
                  response = AssistantMessage(tool_calls=tool_calls, phase=Phase.EXECUTING)
                  yield None, finished, response
                  has_content = True

                # Handle legacy format - message instead of delta
                elif finished and hasattr(choice, "message") and choice.message.content:
                  response = AssistantMessage(content=choice.message.content, phase=Phase.EXECUTING)
                  yield None, True, response
                  has_content = True

                # Break on finish signal
                if finished:
                  logger.debug("Breaking due to finish_reason")
                  break
              else:
                logger.warning(f"Unexpected chunk format in streaming response: {chunk}")

        except asyncio.TimeoutError:
          logger.error(f"Streaming timeout after {timeout_seconds} seconds")
          response = AssistantMessage(
            content="I apologize, but the response took too long to generate.", phase=Phase.EXECUTING
          )
          yield None, True, response
          return
        except Exception as stream_error:
          logger.error(f"Streaming error: {stream_error}")
          response = AssistantMessage(
            content="I apologize, but I encountered an error generating the response.", phase=Phase.EXECUTING
          )
          yield None, True, response
          return

        # Post-stream validation and cleanup
        logger.debug(
          f"Streaming complete: chunks={chunk_count}, has_content={has_content}, last_finished={last_finished}"
        )

        if chunk_count == 0:
          # No chunks received at all
          logger.warning("No streaming chunks received")
          response = AssistantMessage(
            content="I apologize, but I encountered an issue generating a response.", phase=Phase.EXECUTING
          )
          yield None, True, response
        elif not has_content and not last_finished:
          # We got chunks but no actual content and no finish signal
          logger.warning(f"Streaming incomplete: {chunk_count} chunks but no content or finish signal")
          response = AssistantMessage(content="I apologize, but the response was incomplete.", phase=Phase.EXECUTING)
          yield None, True, response
        elif has_content and not last_finished:
          # We got content but no finish signal - force completion
          logger.warning("Streaming had content but no finish signal, forcing completion")
          response = AssistantMessage(content="", phase=Phase.EXECUTING)
          yield None, True, response
      else:
        # Non-streaming response
        response = await self.model.complete_chat(messages, stream=False, tools=self.tool_specs)
        if hasattr(response, "choices") and len(response.choices) > 0:
          message = response.choices[0].message
          if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
              # Generate unique ID with collision detection
              call_id = tc.id if hasattr(tc, "id") and tc.id else None
              if not call_id:
                call_id = await self._generate_unique_tool_call_id()

              tool_call = ToolCall(
                id=call_id,
                function=tc.function if hasattr(tc, "function") else None,
                type="function",
              )
              tool_calls.append(tool_call)
            assistant_message = AssistantMessage(
              content=message.content or "", tool_calls=tool_calls, phase=Phase.EXECUTING
            )
          else:
            assistant_message = AssistantMessage(content=message.content or "", phase=Phase.EXECUTING)
          yield None, True, assistant_message

    except Exception as e:
      logger.error(f"Error in complete_chat: {e}")
      error = Error(f"Model completion failed: {str(e)}")
      yield error, True, None

  async def send_response(self, response, scope: str, conversation: str, stream: bool = False):
    """Send a response back to the conversation."""

    if stream:
      # For streaming responses, yield each part
      for message in response.messages:
        snippet = StreamedConversationSnippet(
          ConversationSnippet(scope, conversation, [message]),
          1,  # part number
          True,  # finished
        )
        yield snippet
    else:
      # For non-streaming, return the complete response
      yield ConversationSnippet(scope, conversation, response.messages)

  async def call_tool(self, tool_call: ToolCall):
    """Execute a tool call and return the result."""

    tool_name = tool_call.function.name
    tool_args = tool_call.function.arguments

    try:
      if tool_name in self.tools:
        tool = self.tools[tool_name]
        result = await tool.invoke(tool_args)

        response = ToolCallResponseMessage(
          tool_call_id=tool_call.id, name=tool_name, content=str(result), phase=Phase.EXECUTING
        )
        return None, response
      else:
        error_msg = f"Tool '{tool_name}' not found"
        response = ToolCallResponseMessage(
          tool_call_id=tool_call.id, name=tool_name, content=error_msg, phase=Phase.EXECUTING
        )
        return error_msg, response

    except Exception as e:
      error_msg = f"Tool execution failed: {str(e)}"
      response = ToolCallResponseMessage(
        tool_call_id=tool_call.id, name=tool_name, content=error_msg, phase=Phase.EXECUTING
      )
      return error_msg, response

  async def _generate_unique_tool_call_id(self):
    """Generate a unique tool call ID with collision detection."""
    async with self._tool_call_id_lock:
      max_attempts = 100
      attempt = 0

      while attempt < max_attempts:
        # Generate UUID-based ID
        call_id = f"call_{uuid.uuid4().hex}"

        # Check for collision
        if call_id not in self._tool_call_ids:
          self._tool_call_ids[call_id] = True

          # Clean up old IDs to prevent memory growth (keep last 1000)
          if len(self._tool_call_ids) > 1000:
            # Remove oldest 100 IDs (using OrderedDict preserves insertion order)
            old_ids = list(self._tool_call_ids.keys())[:100]
            for old_id in old_ids:
              del self._tool_call_ids[old_id]

          return call_id

        attempt += 1

      # Fallback if somehow all UUIDs collide (extremely unlikely)
      import time

      fallback_id = f"call_{uuid.uuid4().hex}_{int(time.time() * 1000000)}"
      self._tool_call_ids[fallback_id] = True
      return fallback_id

  @staticmethod
  async def start(
    node: Node,
    instructions: str,
    name: Optional[str] = None,
    model: Optional[Model] = None,
    memory_model: Optional[Model] = None,
    memory_embeddings_model: Optional[Model] = None,
    tools: Optional[List[InvokableTool]] = None,
    planner: Optional[Planner] = None,
    knowledge: Optional[KnowledgeProvider] = None,
    max_iterations: int = 14,
    max_total_transitions: int = 500,
    max_planning_transitions: int = 50,
    max_tool_calling_transitions: int = 100,
    max_execution_time: float = 300.0,
    exposed_as: Optional[str] = None,
  ):
    """Start an agent on a node."""

    if name is None:
      name = secrets.token_hex(12)

    validate_address(name)

    if model is None:
      model = Model(name="claude-3-5-sonnet-v2")

    if knowledge is None:
      knowledge = NoopKnowledge()

    return await Agent.start_agent_impl(
      node,
      name,
      instructions,
      model,
      memory_model,
      memory_embeddings_model,
      tools,
      planner,
      knowledge,
      max_iterations,
      max_total_transitions,
      max_planning_transitions,
      max_tool_calling_transitions,
      max_execution_time,
      exposed_as,
    )

  @staticmethod
  async def stop(node: Node, name: str):
    """Stop an agent on a node."""
    await node.stop_worker(name)

  @staticmethod
  async def start_many(
    node: Node,
    instructions: str,
    number_of_agents: int,
    model: Optional[Model] = None,
    memory_model: Optional[Model] = None,
    memory_embeddings_model: Optional[Model] = None,
    tools: Optional[List[InvokableTool]] = None,
    planner: Optional[Planner] = None,
    knowledge: Optional[KnowledgeProvider] = None,
    max_iterations: int = 14,
    max_total_transitions: int = 500,
    max_planning_transitions: int = 50,
    max_tool_calling_transitions: int = 100,
    max_execution_time: float = 300.0,
  ):
    """Start multiple agents on a node."""

    if model is None:
      model = Model(name="claude-3-5-sonnet-v2")

    if knowledge is None:
      knowledge = NoopKnowledge()

    agents = []
    for i in range(number_of_agents):
      name = secrets.token_hex(12)
      agent_ref = await Agent.start_agent_impl(
        node,
        name,
        instructions,
        model,
        memory_model,
        memory_embeddings_model,
        tools,
        copy.deepcopy(planner),
        copy.deepcopy(knowledge),
        max_iterations,
        max_total_transitions,
        max_planning_transitions,
        max_tool_calling_transitions,
        max_execution_time,
        None,
      )
      agents.append(agent_ref)

    return agents

  @staticmethod
  async def start_agent_impl(
    node: Node,
    name: str,
    instructions: str,
    model: Model,
    memory_model: Optional[Model],
    memory_embeddings_model: Optional[Model],
    tools: Optional[List[InvokableTool]],
    planner: Optional[Planner],
    knowledge: KnowledgeProvider,
    max_iterations: int,
    max_total_transitions: int,
    max_planning_transitions: int,
    max_tool_calling_transitions: int,
    max_execution_time: float,
    exposed_as: Optional[str],
  ):
    """Create and start an agent implementation."""

    validate_address(name)

    if tools is None:
      tools = []

    # Prepare tools
    tool_specs, tools_dict = await prepare_tools(tools, node)

    # Create shared memory for the agent
    from ..memory.memory import Memory

    memory = Memory()

    # Create the agent
    def agent_creator():
      return Agent(
        node=node,
        name=name,
        instructions=instructions,
        model=model,
        memory_model=memory_model,
        memory_embeddings_model=memory_embeddings_model,
        tool_specs=tool_specs,
        tools=tools_dict,
        planner=planner,
        memory=memory,
        knowledge=knowledge,
        maximum_iterations=max_iterations,
        max_total_transitions=max_total_transitions,
        max_planning_transitions=max_planning_transitions,
        max_tool_calling_transitions=max_tool_calling_transitions,
        max_execution_time=max_execution_time,
      )

    await node.start_spawner(name, agent_creator, key_extractor, None, exposed_as)

    logger.debug(f"Successfully started agent {name}")

    return AgentReference(name, node, exposed_as)


def key_extractor(message):
  """Extract the key from a message for routing."""
  message = json.loads(message)
  scope = message.get("scope")
  conversation = message.get("conversation")
  if conversation:
    if scope:
      return f"{scope}/{conversation}"
  return None


async def prepare_tools(tools: List[InvokableTool], node: Node):
  """Prepare tools for use by the agent."""

  tool_specs = []
  tools_dict = {}

  if tools is None:
    tools = []

  for tool in tools:
    # Set the node reference for tools that need it
    if hasattr(tool, "node"):
      tool.node = node

    # Get the tool specification
    spec = await tool.spec()
    tool_specs.append(spec)

    # Add to the tools dictionary using tool name
    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    tools_dict[tool_name] = tool

  return tool_specs, tools_dict


def system_message(message: str) -> dict:
  """Create a system message dictionary."""
  return {"role": "system", "content": {"text": message, "type": "text"}, "phase": "system"}


class ConversationResponse:
  def __init__(self, scope: str, conversation: str, stream: bool = False):
    self.scope = scope
    self.conversation = conversation
    self.stream = stream
    self.counter = 0
    # Async lock for counter protection in async context
    self._counter_lock = asyncio.Lock()

  async def make_snippet(self, message: ConversationMessage, finished: bool = False):
    if self.stream:
      # Async-safe counter increment
      async with self._counter_lock:
        self.counter += 1
        current_counter = self.counter
      return StreamedConversationSnippet(
        ConversationSnippet(self.scope, self.conversation, [message]), current_counter, finished
      )
    else:
      return ConversationSnippet(self.scope, self.conversation, [message])

  async def make_finished_snippet(self):
    # Async-safe counter increment
    async with self._counter_lock:
      self.counter += 1
      current_counter = self.counter
    return StreamedConversationSnippet(ConversationSnippet(self.scope, self.conversation, []), current_counter, True)


class AgentMemoryKnowledge:
  def __init__(self, mem0_knowledge: Mem0Knowledge):
    self.mem0_knowledge = mem0_knowledge
    self.added_messages = []

  @staticmethod
  async def create(memory_model: Model, memory_embeddings_model: Model):
    mem0_knowledge = await Mem0Knowledge.create(memory_model, memory_embeddings_model)

    return AgentMemoryKnowledge(mem0_knowledge)

  async def add(self, scope: Optional[str], conversation: Optional[str], messages):
    new_messages = messages

    # for message in messages:
    #     # FIXME: Far from perfect, messages should be assigned unique ids
    #     if {"scope": scope, "conversation": conversation, "message": message} not in self.added_messages:
    #         new_messages.append(message)

    if not new_messages:
      return

    await self.mem0_knowledge.add(scope=scope, conversation=conversation, messages=new_messages)

    for message in new_messages:
      self.added_messages.append({"scope": scope, "conversation": conversation, "message": message})

  async def search(self, scope: Optional[str], conversation: Optional[str], query: str) -> Optional[str]:
    return await self.mem0_knowledge.search_knowledge(scope=scope, conversation=conversation, query=query)
