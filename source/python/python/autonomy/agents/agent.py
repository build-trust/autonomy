"""
Agent implementation for the Autonomy framework.

Agents are intelligent actors that autonomously accomplish goals. Apps, running on the
Autonomy Computer, can create millions of parallel collaborating agents in seconds.

Each agent:
- Has a unique identity.
- Uses a large language model to understand, reason, and make autonomous decisions.
- Has memory, so it can retain and recall information over time, to make better autonomous decisions.
- Invokes tools to gather new information and take actions.
- Makes plans that break down complex tasks into small iterative steps.
- Can retrieve knowledge beyond the training data of its language model.
- Collaborates with and can delegate work to other agents.

Autonomy Agents are built using the actor model and have all the properties of actors.

Key Components:
- Agent: The main agent implementation (persistent identity + configuration)
- StateMachine: Orchestrates individual conversations (ephemeral per-request)
- State: Enum defining conversation flow states (READY/THINKING/ACTING/DONE)
- ConversationResponse: Helper for generating streaming/non-streaming responses

Architecture Pattern:
One Agent instance handles many concurrent conversations, each with its own
StateMachine. This enables resource efficiency while maintaining isolation.

For practical guides:
- Agent deployment: docs/_for-coding-agents/create-a-new-autonomy-app.mdx
- Memory and scope: docs/_for-coding-agents/memory.mdx
- Tools: docs/_for-coding-agents/tools.mdx
"""

import asyncio
import json
import secrets
import time
import traceback
import uuid

from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional

from ..autonomy_in_rust_for_python import warn
from ..helpers.validate_address import validate_address
from ..logs import get_logger
from ..memory.memory import Memory
from ..models.model import Model
from ..nodes.node import Node
from ..tools.protocol import InvokableTool

from ..nodes.message import (
  AgentReference,
  AssistantMessage,
  ConversationMessage,
  ConversationSnippet,
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

logger = get_logger("agent")


# =============================================================================
# CONSTANTS - EXECUTION LIMITS AND SAFETY BACKSTOPS
# =============================================================================

# Prevent infinite loops and runaway execution
MAX_ITERATIONS_DEFAULT = 1000
MAX_EXECUTION_TIME_DEFAULT = 600.0  # 10 minutes

# Streaming safety limits
MAX_STREAMING_CHUNKS = 1000
STREAMING_TIMEOUT_SECONDS = 300  # 5 minutes

# Tool call ID management - prevent UUID collisions
MAX_TOOL_ID_GENERATION_ATTEMPTS = 100
MAX_TOOL_CALL_IDS_TO_RETAIN = 1000
TOOL_ID_CLEANUP_BATCH_SIZE = 100

# Memory and context limits
MAX_ACTIVE_MESSAGES_DEFAULT = 100
MAX_ACTIVE_TOKENS_DEFAULT = 8000


# =============================================================================
# STATE MACHINE
# =============================================================================


class State(Enum):
  """
  Conversation flow states for agent's autonomous decision-making.

  States:
    - READY: Prepare conversation context before model interaction
    - THINKING: Agent uses LLM to understand, reason, and pick next step
    - ACTING: Agent invokes tools to gather information and take actions
    - WAITING_FOR_INPUT: Agent has asked user for input and is paused
    - DONE: Agent finalizes and returns results

  Transitions are controlled by agent's autonomous decisions and tool execution results.
  See ASCII diagram below for complete transition graph.
  """

  READY = "ready"
  THINKING = "thinking"
  ACTING = "acting"
  WAITING_FOR_INPUT = "waiting_for_input"
  DONE = "done"


#     ┌─────────┐
#     │         │
#     │  READY  │
#     │         │
#     └───┬─────┘
#         │
#         ▼
#     ┌──────────┐
#     │          │
#     │ THINKING │◀─────┐
#     │          │      │
#     └┬────┬────┘      │
#      │    │           │
#      │    ▼           │
#      │  ┌────────┐    │
#      │  │        │    │
#      │  │ ACTING │────┤
#      │  │        │    │
#      │  └───┬────┘    │
#      │      │         │
#      │      ▼         │
#      │  ┌──────────────────┐
#      │  │                  │
#      │  │ WAITING_FOR_     │ [paused, waits for user]
#      │  │     INPUT        │
#      │  │                  │
#      │  └──────────────────┘
#      │        │
#      │        │ [user responds]
#      │        └──────────────┐
#      │                       │
#      │ [response complete && no tool calls]
#      │ [error conditions]
#      │ [max iterations reached]
#      ▼
#  ┌─────────┐
#  │         │
#  │  DONE   │
#  │         │
#  └─────────┘
#
# State Transition Conditions:
#
# READY transitions:
# - → THINKING: [always]
#
# THINKING transitions:
# - → ACTING: [response contains tool calls]
# - → DONE: [response complete && no tool calls] OR [error conditions] OR [max iterations reached]
#
# ACTING transitions:
# - → THINKING: [tool calls executed successfully]
# - → WAITING_FOR_INPUT: [ask_user_for_input tool called]
#
# WAITING_FOR_INPUT transitions:
# - → THINKING: [user responds with new message] (resume triggered by Agent)
# - → THINKING: [after processing all tool calls] OR [error conditions] OR [max tool calling transitions]
#
# Error conditions include: model completion errors, timeout exceeded, transition limits exceeded
#


class StateMachine:
  """
  Manage agent's autonomous decision-making flow through discrete state transitions.

  Manages a single conversation where the agent autonomously accomplishes goals by:
    1. Using its LLM to understand, reason, and decide (THINKING state)
    2. Invoking tools to gather information and take actions (ACTING state)
    3. Making plans that break complex tasks into iterative steps
    4. Repeating until goal accomplished (DONE state)

  Capabilities:
    - Autonomous decisions about when and how to use tools
    - Break complex tasks into manageable steps
    - Retrieve knowledge beyond LLM training data
    - Retain and recall information for better decisions

  Architecture (Actor Model Pattern):
    - Agent: Persistent actor (identity, memory, model, tools) across conversations
    - StateMachine: Ephemeral orchestrator per conversation (created per request)

    This separation enables one Agent to handle multiple concurrent conversations,
    each with independent StateMachine state.

  Safety Mechanisms:
    - Iteration limits → prevent infinite tool-calling loops
    - Execution time limits → catch stuck model calls or slow tools
    - Streaming limits → prevent memory exhaustion from runaway responses

  Attributes:
    agent: The Agent instance this state machine operates for
    scope: User/tenant identifier for memory isolation
    conversation: Conversation thread identifier
    stream: Whether to stream responses incrementally
    streaming_response: Helper for generating streaming chunks
    state: Current state (READY/THINKING/ACTING/DONE)
    iteration: State machine cycle count (increments each THINKING entry)
    tool_calls: Tool calls from agent's autonomous decisions
    whole_response: Accumulated response for non-streaming mode
    final_content_sent: Prevents duplicate finish signals in streaming
    start_time: Execution time limit enforcement timestamp
    max_execution_time: Maximum execution time in seconds
  """

  def __init__(self, agent, scope, conversation, stream, response):
    self.agent = agent
    self.scope = scope
    self.conversation = conversation
    self.stream = stream
    self.streaming_response = response

    # Initialize state machine at READY state - will transition to THINKING on first cycle
    self.state = State.READY
    self.iteration = 0
    self.tool_calls = []

    # For non-streaming mode, accumulate all messages before returning
    self.whole_response = ConversationSnippet(scope, conversation, [])

    # Track whether we've sent the final streaming chunk to prevent duplicates
    # Edge cases (timeouts, incomplete streams) could trigger multiple finish signals
    self.final_content_sent = False

    # Start execution timer for safety backstop
    self.start_time = time.time()
    self.max_execution_time = agent.max_execution_time

    # Pause/resume support for human-in-the-loop functionality
    self.is_paused = False  # True when waiting for user input
    self.waiting_prompt = None  # The question shown to user
    self.interrupted = False  # True when new message interrupts active state machine
    self.pending_tool_call_id = None  # Track ask_user_for_input tool call ID for resume

  async def transition(self):
    """
    Execute current state and determine next state.

    Checks safety limits, then delegates to state-specific handlers.
    """

    # Check execution time limit
    current_time = time.time()
    elapsed_time = current_time - self.start_time

    if elapsed_time > self.max_execution_time:
      logger.error(f"State machine exceeded maximum execution time ({self.max_execution_time}s)")
      yield Error(f"Agent exceeded maximum execution time ({self.max_execution_time:.1f}s)")
      self.state = State.DONE
      return

    logger.debug(f"State transition: {self.state.name} (iteration: {self.iteration}, elapsed: {elapsed_time:.1f}s)")

    match self.state:
      case State.READY:
        self.state = State.THINKING
      case State.THINKING:
        async for result in self._handle_thinking_state():
          yield result
      case State.ACTING:
        async for result in self._handle_acting_state():
          yield result
      case State.WAITING_FOR_INPUT:
        async for result in self._handle_waiting_for_input_state():
          yield result

  async def _handle_thinking_state(self):
    """
    Handle THINKING state: Call model to generate response.

    OUTCOMES:
    ┌────────────────────┬─────────────┬────────────────────┐
    │ Model Response     │ Has Tools?  │ Next State         │
    ├────────────────────┼─────────────┼────────────────────┤
    │ Finished           │ Yes         │ ACTING             │
    │ Finished           │ No          │ DONE               │
    │ Error              │ -           │ DONE               │
    │ Max iterations     │ -           │ DONE               │
    └────────────────────┴─────────────┴────────────────────┘

    Response types:
      - Final text response → DONE
      - Tool call request → ACTING
      - Streaming chunks → Process incrementally

    Streaming vs non-streaming: Streaming provides better UX but requires careful
    handling of incomplete responses, finish signals, and timeout edge cases.
    """

    # Check for interruption
    if self.interrupted:
      logger.info("State machine interrupted during THINKING")
      yield Error("Agent was interrupted by new message")
      self.state = State.DONE
      return

    self.iteration += 1
    if self.iteration >= self.agent.max_iterations:
      yield Error(str(RuntimeError("Reached max_iterations")))
      self.state = State.DONE
      return

    self.tool_calls = []
    response_received = False
    finish_signal_received = False
    chunks_processed = 0

    logger.debug(f"Starting model completion, iteration {self.iteration}")

    try:
      # Call the model and process response chunks
      async for err, finished, model_response in self.agent.complete_chat(
        self.scope, self.conversation, stream=self.stream
      ):
        chunks_processed += 1
        logger.debug(
          f"Received chunk {chunks_processed}: err={err}, finished={finished}, response_type={type(model_response)}"
        )

        if err:  # Propagate model error and terminate state machine
          logger.error(f"Model completion error: {err}")
          yield err
          self.state = State.DONE
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
          self.state = State.ACTING

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
            # Accumulate final message
            self.whole_response.messages.append(model_response)

          if self.state == State.ACTING:
            logger.debug("Transitioning to ACTING")
            return
          else:
            logger.debug("Transitioning to DONE")
            self.state = State.DONE
          return
        else:
          # Intermediate streaming chunk
          if self.stream and (model_response.content or model_response.tool_calls):
            yield await self.streaming_response.make_snippet(model_response)
          elif not self.stream:
            self.whole_response.messages.append(model_response)

          # Prevent infinite streaming from malformed model responses
          if chunks_processed >= MAX_STREAMING_CHUNKS:
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
      self.state = State.DONE
      return

    logger.debug(
      f"Stream complete: chunks={chunks_processed}, response_received={response_received}, "
      f"finish_signal={finish_signal_received}"
    )

    if not response_received:
      logger.error("No response received from model")
      yield Error("No response received from model")
      self.state = State.DONE
      return

    # Incomplete streaming: Network issues, model errors, or chunking problems
    if response_received and not finish_signal_received:
      logger.warning(f"Stream incomplete - no finish signal after {chunks_processed} chunks, forcing completion")

      if self.stream and not self.final_content_sent:
        # Send empty finish chunk to complete the stream
        from ..nodes.message import AssistantMessage

        empty_response = AssistantMessage(content="", phase=Phase.EXECUTING)
        yield await self.streaming_response.make_snippet(empty_response, finished=True)
        self.final_content_sent = True

      if self.state == State.THINKING:
        self.state = State.DONE

    # Final safety check to prevent hangs in edge cases
    if self.state == State.THINKING:
      logger.warning(f"Forcing DONE state after {chunks_processed} chunks")
      self.state = State.DONE

  async def _handle_acting_state(self):
    """
    Handle ACTING state: Execute tool calls from model response.

    Flow:
      1. Process all tool calls from model sequentially
      2. Execute each tool and capture result or error
      3. Store results in memory
      4. Transition to THINKING for model to process results

    Sequential execution ensures predictable ordering when tool results depend
    on each other.

    Special handling for human-in-the-loop:
      - Detects ask_user_for_input tool call
      - Transitions to WAITING_FOR_INPUT state
      - Pauses execution until user responds
    """

    # Check for interruption
    if self.interrupted:
      logger.info("State machine interrupted during ACTING")
      yield Error("Agent was interrupted by new message")
      self.state = State.DONE
      return

    for tool_call in self.tool_calls:
      error, tool_call_response, raw_result = await self.agent.call_tool(tool_call)

      if error:
        # Continue so model can see error and potentially recover
        warn(f"MCP call failed: {error}")

      # Check if this is ask_user_for_input with waiting marker
      if (not error and
          isinstance(raw_result, dict) and
          raw_result.get("_waiting_for_input")):

        # Extract prompt for user
        self.waiting_prompt = raw_result.get("prompt", "Waiting for input...")
        self.pending_tool_call_id = tool_call.id

        logger.info(f"Agent requesting user input: {self.waiting_prompt}")

        # Transition to waiting state
        self.state = State.WAITING_FOR_INPUT

        # Send the prompt to user as assistant message
        from ..nodes.message import AssistantMessage
        waiting_message = AssistantMessage(
          content=self.waiting_prompt,
          phase=Phase.WAITING_FOR_INPUT
        )
        await self.agent.remember(self.scope, self.conversation, waiting_message)

        if self.stream:
          yield await self.streaming_response.make_snippet(waiting_message)
        else:
          self.whole_response.messages.append(waiting_message)

        # Mark as paused - run() will detect this
        self.is_paused = True
        return

      # Normal tool call - store result and continue
      # the tool_call_response contains the error message if the tool call failed
      await self.agent.remember(self.scope, self.conversation, tool_call_response)

      if self.stream:
        yield await self.streaming_response.make_snippet(tool_call_response)
      else:
        self.whole_response.messages.append(tool_call_response)

    self.state = State.THINKING

  async def _handle_waiting_for_input_state(self):
    """
    Handle WAITING_FOR_INPUT state.

    This state indicates the agent has asked the user for input and is
    waiting for a response. The state machine is paused here and will
    not transition until resume is triggered by a new message.

    Note: This handler is called but doesn't change state - it just
    ensures is_paused flag is set. The actual resume happens when
    handle__conversation_snippet receives a new message.
    """
    logger.debug("State machine in WAITING_FOR_INPUT state (paused)")

    # Ensure paused flag is set
    if not self.is_paused:
      self.is_paused = True

    # Do not transition state - wait for external resume

  async def _handle_done_state(self):
    """
    Handle DONE state: End conversation and return final response.

    Streaming mode: Send final finish signal (if not already sent)
    Non-streaming mode: Return complete accumulated response
    """

    if self.stream:
      if not self.final_content_sent:
        yield await self.streaming_response.make_finished_snippet()
    else:
      yield self.whole_response

  async def initialize_plan(self, messages):
    """
    Store input messages in memory to establish conversation context.

    Stores user's initial messages before starting state machine, providing
    context for model's first response.

    Args:
      messages: List of ConversationMessage objects to store
    """

    for message in messages:
      await self.agent.remember(self.scope, self.conversation, message)

  async def run(self):
    """
    Run state machine until completion or pause.

    Modified to support pause/resume:
      - Checks is_paused flag after each transition
      - Stops iterating when paused (doesn't go to DONE)
      - Can be resumed by setting is_paused = False and calling run() again

    Executes main loop, transitioning between states until DONE. Includes
    timeout protection and statistics logging.

    asyncio.timeout provides additional safety beyond per-iteration checks,
    ensuring entire run completes in time.
    """

    try:
      # Add overall timeout protection for the entire state machine run
      async with asyncio.timeout(self.max_execution_time):
        while self.state != State.DONE:
          # Check for interruption
          if self.interrupted:
            logger.info("State machine interrupted, terminating")
            yield Error("Agent was interrupted by new message")
            self.state = State.DONE
            break

          async for result in self.transition():
            yield result

          # Check if we're paused
          if self.is_paused:
            logger.info(f"State machine paused in {self.state.name} state")
            # Don't transition to DONE - just stop iterating
            # Agent will keep this state machine and resume later
            return

        # Only send done state if not paused
        if not self.is_paused:
          async for result in self._handle_done_state():
            yield result

    except asyncio.TimeoutError:
      logger.error(f"State machine run timed out after {self.max_execution_time}s")
      yield Error(f"Agent execution timed out after {self.max_execution_time:.1f} seconds")

    elapsed_time = time.time() - self.start_time
    logger.info(f"State machine completed: {self.iteration} iterations in {elapsed_time:.1f}s")


# =============================================================================
# AGENT
# =============================================================================


class Agent:
  """
  Intelligent actor that autonomously accomplishes goals.

  Apps on Autonomy Computer can create millions of parallel collaborating agents in seconds.

  Each agent:
    - Has a unique identity
    - Uses LLMs to understand, reason, and make autonomous decisions
    - Has memory to retain and recall information over time
    - Invokes tools to gather information and take actions
    - Makes plans that break complex tasks into iterative steps
    - Can retrieve knowledge beyond its LLM's training data
    - Collaborates with and delegates work to other agents

  Built using the actor model with all actor properties.

  MODELS
    Agents use large language models for understanding, reasoning, and autonomous
    decision-making. The model processes natural language, understands context, and
    determines the best course of action to accomplish goals.

  MEMORY
    Organized by scope ID and conversation ID for context retention across interactions.
    Enables agents to build context, learn from tool results, and maintain coherent plans.

    Three-level hierarchy: {agent_name}/{user_scope}/{conversation}
      - agent_name: Unique identity (e.g., "assistant-bot")
      - user_scope: User/tenant isolation (e.g., "user-alice")
      - conversation: Thread separation (e.g., "chat-2024-01-15")

    Example: Agent "henry" serving user "alice" in conversation "chat1"
             → scope "henry/alice/chat1" for all memory operations

  TOOLS
    External functions (Python or MCP servers) that agents invoke to take actions
    and gather information. Agent autonomously decides when to invoke tools based
    on reasoning. Tool results become memory, informing subsequent decisions.

  ACTOR MODEL PROPERTIES
    - Unique identity: Distinct name and isolated state
    - Message passing: Asynchronous message communication
    - Concurrent execution: Multiple agents run in parallel without interference
    - State encapsulation: Private memory and configuration
    - Location transparency: Collaboration regardless of physical location

  PLANNING AND ITERATION
    Agents break complex tasks into iterative steps through state machine pattern:
      - Analyze goal and determine required steps
      - Execute actions iteratively, learning from each result
      - Adjust plans based on new information from tools
      - Continue until goal accomplished or limits reached

  SAFETY AND LIMITS
    - max_iterations: Prevent infinite planning loops
    - max_execution_time: Prevent stuck/slow agents
    - Streaming timeouts: Prevent hanging on slow responses
    - Memory isolation: Prevent cross-user and cross-agent data leakage

  USAGE
    # Single agent
    agent = await Agent.start(
      node=node,
      name="customer-support",
      instructions="Help users with their questions",
      model=Model(name="claude-sonnet-4-v1"),
      tools=[search_tool, database_tool]
    )

    # Multiple collaborating agents
    agents = await Agent.start_many(
      node=node,
      instructions="Analyze customer sentiment",
      number_of_agents=10,
      model=Model(name="claude-sonnet-4-v1")
    )

  Attributes:
    node: Node this agent actor runs on
    name: Unique identity
    model: LLM for understanding, reasoning, and decisions
    tools: Dict of tool names → InvokableTool instances
    tool_specs: Tool specifications in model-compatible format
    memory: Memory for retaining and recalling information
    max_iterations: Maximum planning iterations before timeout
    max_execution_time: Maximum execution time before timeout
    converter: MessageConverter for serialization/deserialization
    _tool_call_ids: OrderedDict tracking recent IDs for collision detection
    _tool_call_id_lock: Async lock protecting concurrent ID generation
  """

  tools: Dict[str, InvokableTool]

  def __init__(
    self,
    node: Node,
    name: str,
    instructions: str,
    model: Model,
    tool_specs: List[dict],
    tools: Dict[str, InvokableTool],
    memory: Memory,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
    max_execution_time: float = MAX_EXECUTION_TIME_DEFAULT,
  ):
    logger.info(f"Starting agent '{name}'")
    self.node = node
    self.tools = tools
    self.tool_specs = tool_specs
    self.name = name
    self.model = model

    self.memory = memory
    memory.set_instructions(system_message(instructions))

    self.max_iterations = max_iterations
    self.max_execution_time = max_execution_time

    self.converter = MessageConverter.create(node)

    # Tool call ID tracking prevents collisions (OrderedDict enables efficient FIFO cleanup)
    self._tool_call_ids = OrderedDict()
    self._tool_call_id_lock = asyncio.Lock()

    # State machine tracking for pause/resume support
    # Since each Agent instance handles one (scope, conversation) pair,
    # we only need to track one active state machine at a time
    self._active_state_machine: Optional[StateMachine] = None

    logger.info(f"Started agent '{name}'")

  def _memory_scope(self, scope: str) -> str:
    """
    Prevent different agents from seeing each other's memories by prepending agent name.

    Format: "{agent_name}/{user_scope}" → "henry/alice" for agent "henry" serving user "alice"

    This scoping strategy ensures memory isolation between different agents even when
    they serve the same user. Without this, two different agent instances could
    accidentally read or corrupt each other's conversation histories.

    Args:
      scope: The user-provided scope, typically in format "user_id" or "user_id/conversation_id"

    Returns:
      Agent-scoped memory key in format "agent_name/scope"

    Examples:
      Agent "henry" with scope "alice" → "henry/alice"
      Agent "support-bot" with scope "user123/chat1" → "support-bot/user123/chat1"
    """
    return f"{self.name}/{scope}"

  async def handle_message(self, context, message):
    """
    Handle incoming messages by routing to appropriate handler methods.

    Main entry point for messages sent to this agent worker.

    Flow:
    1. Deserialize JSON message → typed message object
    2. Route to handler based on message type
    3. Serialize and send reply back to sender

    Args:
      context: Message context containing reply method and metadata
      message: JSON-serialized message to handle
    """

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

      # Conversation snippets yield multiple streaming replies
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
    """
    Return the agent's unique identity for collaboration and message routing.

    Args:
      message: Request containing scope and conversation identifiers

    Returns:
      Response containing the agent's snake_case identifier
    """
    name_snake_case = self.name.lower().replace(" ", "_")
    return GetIdentifierResponse(message.scope, message.conversation, name_snake_case)

  async def handle__get_conversations_request(self, message: GetConversationsRequest) -> GetConversationsResponse:
    """
    Retrieve all messages from a specific conversation.

    Args:
      message: Request containing scope and conversation identifiers

    Returns:
      Response containing list of conversation messages
    """
    messages = await self.memory.get_messages_only(self._memory_scope(message.scope), message.conversation)
    return GetConversationsResponse(messages)

  async def handle__conversation_snippet(self, message):
    """
    Handle conversation snippets with support for pause/resume and interruption.

    Flow:
    1. Check if there's a paused state machine → resume it
    2. Check if there's an active state machine → interrupt it
    3. Otherwise → create new state machine

    Concurrency model:
    - This Agent instance handles ONE (scope, conversation) pair
    - Messages are serialized by Ockam's worker model
    - No concurrent calls to this method possible
    - No locks needed!

    Args:
      message: ConversationSnippet or StreamedConversationSnippet

    Yields:
      ConversationSnippet or StreamedConversationSnippet responses
    """

    # Extract parameters from message
    stream = type(message) is StreamedConversationSnippet
    scope = message.scope if type(message) is ConversationSnippet else message.snippet.scope
    conversation = message.conversation if type(message) is ConversationSnippet else message.snippet.conversation
    messages = message.messages if type(message) is ConversationSnippet else message.snippet.messages

    # CASE 1: Resume a paused state machine
    if self._active_state_machine and self._active_state_machine.is_paused:
      logger.info(f"Resuming paused state machine for conversation '{conversation}'")

      machine = self._active_state_machine

      # Add new user messages to memory
      for msg in messages:
        await self.remember(scope, conversation, msg)

      # Create tool result with user's response
      user_response = "\n".join([
        msg.content for msg in messages
        if hasattr(msg, 'content') and msg.content
      ])

      # Complete the pending ask_user_for_input tool call
      from ..nodes.message import ToolCallResponseMessage
      tool_result = ToolCallResponseMessage(
        tool_call_id=machine.pending_tool_call_id or "unknown",
        result=user_response
      )
      await self.remember(scope, conversation, tool_result)

      if stream:
        yield await machine.streaming_response.make_snippet(tool_result)
      else:
        machine.whole_response.messages.append(tool_result)

      # Resume the state machine
      machine.is_paused = False
      machine.state = State.THINKING  # Continue from thinking

      logger.info("State machine resumed, continuing execution")

      # Continue running the existing machine
      async for result in machine.run():
        yield result

      # Clean up when done
      if machine.state == State.DONE:
        logger.info("Resumed state machine completed")
        self._active_state_machine = None

      return

    # CASE 2: Interrupt an active (non-paused) state machine
    if self._active_state_machine and not self._active_state_machine.is_paused:
      logger.info(f"Interrupting active state machine for conversation '{conversation}'")

      # Signal interruption
      self._active_state_machine.interrupted = True

      # Give the state machine a moment to notice the flag
      # (it will check on next iteration)
      await asyncio.sleep(0.05)

      # Clear the reference
      self._active_state_machine = None

      logger.info("State machine interrupted, starting new conversation")

    # CASE 3: Start new state machine (normal flow or after interrupt)
    logger.debug(f"Creating new state machine for conversation '{conversation}'")

    # Create streaming response helper
    response = ConversationResponse(scope, conversation, stream)

    # Create new state machine
    machine = StateMachine(self, scope, conversation, stream, response)

    # Initialize with input messages
    await machine.initialize_plan(messages)

    # Store as active
    self._active_state_machine = machine

    # Run until completion or pause
    async for result in machine.run():
      yield result

    # Clean up if done (not paused)
    if machine.state == State.DONE:
      logger.info("State machine completed")
      self._active_state_machine = None
    elif machine.is_paused:
      logger.info(f"State machine paused, waiting for user input: '{machine.waiting_prompt}'")
      # Keep the reference so we can resume later

  async def get_messages_only(self, conversation, scope) -> list[ConversationMessage]:
    """
    Retrieve conversation messages as typed ConversationMessage objects.

    Args:
      conversation: Conversation identifier
      scope: User/tenant scope identifier

    Returns:
      List of ConversationMessage objects
    """
    messages: list[dict] = await self.memory.get_messages_only(self._memory_scope(scope), conversation)
    return [self.converter.conversation_message_from_dict(m) for m in messages]

  async def remember(self, scope: str, conversation: str, message: ConversationMessage):
    """
    Store a message in conversation memory.

    Args:
      scope: User/tenant scope identifier
      conversation: Conversation identifier
      message: Message to store (ConversationMessage or dict)
    """

    if not isinstance(message, dict):
      message = self.converter.message_to_dict(message)

    await self.memory.add_message(self._memory_scope(scope), conversation, message)

  async def message_history(self, scope: str, conversation: str) -> list[dict]:
    """
    Retrieve complete message history (system, user, assistant, and tool messages).

    Args:
      scope: User/tenant scope identifier
      conversation: Conversation identifier

    Returns:
      List of message dicts in chronological order
    """
    return await self.memory.get_messages(self._memory_scope(scope), conversation)

  async def determine_input_context(self, scope: str, conversation: str):
    """
    Retrieve message history as typed ConversationMessage objects for model input.

    Args:
      scope: User/tenant scope identifier
      conversation: Conversation identifier

    Returns:
      List of ConversationMessage objects providing context for the model
    """

    raw_messages = await self.memory.get_messages(self._memory_scope(scope), conversation)
    messages = [self.converter.conversation_message_from_dict(m) for m in raw_messages]

    return messages

  async def complete_chat(self, scope: str, conversation: str, stream: bool = False):
    """
    Complete a chat conversation using the agent's model.

    Flow:
      1. Retrieve conversation history from memory
      2. Choose streaming (incremental) vs non-streaming (atomic) path
      3. Call model.complete_chat() with context + tools
      4. Process response chunks or complete response
      5. Handle errors, timeouts, and edge cases

    Streaming vs Non-streaming:
      ┌─────────────┐
      │   stream?   │
      └──────┬──────┘
             ├── True  → Streaming: Better UX, lower latency to first token
             └── False → Non-streaming: Simpler errors, atomic responses

    Args:
      scope: User/tenant identifier for memory isolation
      conversation: Conversation thread identifier
      stream: Whether to stream responses incrementally

    Yields:
      Tuples of (error, finished, message):
        - error: Error object if model call failed, None otherwise
        - finished: Boolean indicating if this is the final message
        - message: AssistantMessage containing model response

    Safety:
      - Streaming timeout prevents hung connections
      - Chunk limit prevents memory exhaustion
      - Incomplete streams: Forced completion with proper finish signals
    """

    try:
      messages = await self.determine_input_context(scope, conversation)

      # =========================================================================
      # STREAMING PATH: Incremental response delivery
      # =========================================================================
      if stream:
        # Initialize streaming state tracking
        chunk_count = 0
        has_content = False
        last_finished = False
        max_chunks = MAX_STREAMING_CHUNKS
        timeout_seconds = STREAMING_TIMEOUT_SECONDS

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

                # --- Handle content chunks (text responses) ---
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                  content = choice.delta.content
                  logger.debug(f"Delta content: {repr(content)}, finished: {finished}")

                  if content:  # Non-empty content chunk
                    has_content = True
                    response = AssistantMessage(content=content, phase=Phase.EXECUTING)
                    logger.debug(f"Yielding content chunk: finished={finished}")
                    yield None, finished, response

                  elif finished and (content == "" or content is None):
                    # Empty content with finish_reason - termination chunk
                    logger.debug("Received empty finish chunk - streaming complete")
                    has_content = True
                    # Send empty response to signal completion
                    response = AssistantMessage(content="", phase=Phase.EXECUTING)
                    yield None, True, response
                    break

                # --- Handle tool call chunks ---
                elif hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                  tool_calls = []
                  for tc in choice.delta.tool_calls:
                    # Generate unique ID with collision detection (some models don't provide IDs)
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

                # --- Handle legacy format (message instead of delta) ---
                elif finished and hasattr(choice, "message") and choice.message.content:
                  # Some older model APIs use 'message' instead of 'delta'
                  response = AssistantMessage(content=choice.message.content, phase=Phase.EXECUTING)
                  yield None, True, response
                  has_content = True

                # Break on finish signal - no more chunks expected
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

        # --- Post-stream validation and cleanup ---
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

      # =========================================================================
      # NON-STREAMING PATH: Complete response delivery
      # =========================================================================
      else:
        response = await self.model.complete_chat(messages, stream=False, tools=self.tool_specs)

        if hasattr(response, "choices") and len(response.choices) > 0:
          message = response.choices[0].message

          if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
              # Generate unique ID with collision detection (some models don't provide IDs)
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
    """
    Send a response formatted for streaming or non-streaming mode.

    Args:
      response: Response object containing messages to send
      scope: User/tenant scope identifier
      conversation: Conversation identifier
      stream: Whether to use streaming format

    Yields:
      StreamedConversationSnippet or ConversationSnippet
    """

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
    """
    Execute a tool call and return the result or error.

    Flow:
      1. Look up tool by name in self.tools dict
      2. Invoke tool with provided arguments
      3. Capture result or error
      4. Return as ToolCallResponseMessage

    Args:
      tool_call: ToolCall object containing tool name, ID, and arguments

    Returns:
      Tuple of (error_string, ToolCallResponseMessage, raw_result):
        - error_string: Error description if failed, None if successful
        - ToolCallResponseMessage: Response containing tool result or error
        - raw_result: The raw result from tool.invoke() before string conversion
    """

    tool_name = tool_call.function.name
    tool_args = tool_call.function.arguments

    try:
      if tool_name in self.tools:
        tool = self.tools[tool_name]
        result = await tool.invoke(tool_args)

        response = ToolCallResponseMessage(
          tool_call_id=tool_call.id, name=tool_name, content=str(result), phase=Phase.EXECUTING
        )
        return None, response, result
      else:
        error_msg = f"Tool '{tool_name}' not found"
        response = ToolCallResponseMessage(
          tool_call_id=tool_call.id, name=tool_name, content=error_msg, phase=Phase.EXECUTING
        )
        return error_msg, response, None

    except Exception as e:
      error_msg = f"Tool execution failed: {str(e)}"
      response = ToolCallResponseMessage(
        tool_call_id=tool_call.id, name=tool_name, content=error_msg, phase=Phase.EXECUTING
      )
      return error_msg, response, None

  async def _generate_unique_tool_call_id(self):
    """
    Generate a unique tool call ID with collision detection.

    Why detect collisions when UUID4 has ~10^-36 collision probability?
      - UUID4 collision probability: ~10^-36 per pair
      - With concurrent calls/sec: expect first collision after 10^26 years
      - BUT: Collision consequences are severe (wrong tool results → wrong calls)
      - Detection cost is negligible vs debugging misrouted responses

    Flow:
      1. Generate UUID-based ID: "call_{uuid}"
      2. Check against recent IDs for collisions
      3. Cleanup old IDs when limit reached (prevents unbounded growth)
      4. Fallback to timestamped ID if all retries fail (virtually impossible)

    Memory management:
      - Retain recent IDs in OrderedDict (FIFO cleanup)
      - Covers typical conversation workloads
      - Batch cleanup reduces lock contention

    Returns:
      Unique tool call ID: "call_{uuid}" or "call_{uuid}_{timestamp}" (fallback)
    """
    async with self._tool_call_id_lock:
      max_attempts = MAX_TOOL_ID_GENERATION_ATTEMPTS
      attempt = 0

      while attempt < max_attempts:
        call_id = f"call_{uuid.uuid4().hex}"

        if call_id not in self._tool_call_ids:
          self._tool_call_ids[call_id] = True

          if len(self._tool_call_ids) > MAX_TOOL_CALL_IDS_TO_RETAIN:
            old_ids = list(self._tool_call_ids.keys())[:TOOL_ID_CLEANUP_BATCH_SIZE]
            for old_id in old_ids:
              del self._tool_call_ids[old_id]

          return call_id

        attempt += 1

      # Fallback: Add timestamp for guaranteed uniqueness (virtually unreachable)
      import time

      fallback_id = f"call_{uuid.uuid4().hex}_{int(time.time() * 1000000)}"
      self._tool_call_ids[fallback_id] = True
      return fallback_id

  @staticmethod
  async def stop(node: Node, name: str):
    """
    Stop an agent worker on a node.

    Args:
      node: Node instance where the agent is running
      name: Name of the agent to stop
    """
    await node.stop_worker(name)

  @staticmethod
  async def start(
    node: Node,
    instructions: str,
    name: Optional[str] = None,
    model: Optional[Model] = None,
    tools: Optional[List[InvokableTool]] = None,
    max_iterations: Optional[int] = None,
    max_execution_time: Optional[float] = None,
    exposed_as: Optional[str] = None,
    max_active_messages: int = 100,
    max_active_tokens: Optional[int] = 8000,
  ):
    """
    Start a single agent on a node.

    Factory method that handles initialization, memory setup, and registration.

    Args:
      node: Node instance where the agent will run
      instructions: System instructions defining agent behavior
      name: Unique agent name (auto-generated if None)
      model: LLM to use (defaults to claude-sonnet-4-v1)
      tools: List of tools available to the agent
      max_iterations: Maximum state machine iterations
      max_execution_time: Maximum execution time in seconds
      exposed_as: Optional external name for HTTP exposure
      max_active_messages: Maximum messages in active memory
      max_active_tokens: Maximum tokens in active memory

    Returns:
      AgentReference for interacting with the started agent
    """

    if name is None:
      name = secrets.token_hex(12)

    validate_address(name)

    if model is None:
      model = Model(name="claude-sonnet-4-v1")

    return await Agent.start_agent_impl(
      node,
      name,
      instructions,
      model,
      tools,
      max_iterations,
      max_execution_time,
      exposed_as,
      max_active_messages,
      max_active_tokens,
    )

  @staticmethod
  async def start_many(
    node: Node,
    instructions: str,
    number_of_agents: int,
    model: Optional[Model] = None,
    tools: Optional[List[InvokableTool]] = None,
    max_iterations: Optional[int] = None,
    max_execution_time: Optional[float] = None,
    max_active_messages: int = 100,
    max_active_tokens: Optional[int] = 8000,
  ):
    """
    Start multiple agents on a node for load distribution.

    Creates multiple agents with identical configuration to enable concurrent
    processing of different conversations for better throughput.

    Args:
      node: Node instance where the agents will run
      instructions: System instructions defining agent behavior
      number_of_agents: Number of agent instances to create
      model: LLM to use (defaults to claude-3-5-sonnet-v2)
      tools: List of tools available to all agents
      max_iterations: Maximum state machine iterations
      max_execution_time: Maximum execution time in seconds
      max_active_messages: Maximum messages in active memory
      max_active_tokens: Maximum tokens in active memory

    Returns:
      List of AgentReference objects for the started agents
    """

    if model is None:
      model = Model(name="claude-3-5-sonnet-v2")

    agents = []
    for i in range(number_of_agents):
      name = secrets.token_hex(12)

      agent_ref = await Agent.start_agent_impl(
        node,
        name,
        instructions,
        model,
        tools,
        max_iterations,
        max_execution_time,
        None,  # exposed_as not supported for multiple agents
        max_active_messages,
        max_active_tokens,
      )
      agents.append(agent_ref)

    return agents

  @staticmethod
  async def start_agent_impl(
    node: Node,
    name: str,
    instructions: str,
    model: Model,
    tools: Optional[List[InvokableTool]],
    max_iterations: Optional[int],
    max_execution_time: Optional[float],
    exposed_as: Optional[str],
    max_active_messages: int = 100,
    max_active_tokens: Optional[int] = 8000,
  ):
    """
    Internal implementation for agent creation and startup.

    Shared by start() and start_many() to avoid code duplication. Handles
    agent creation, memory initialization, and worker registration.

    Args:
      node: Node instance where the agent will run
      name: Unique agent name
      instructions: System instructions defining agent behavior
      model: LLM to use for reasoning
      tools: List of tools available to the agent
      max_iterations: Maximum state machine iterations
      max_execution_time: Maximum execution time in seconds
      exposed_as: Optional external name for HTTP exposure
      max_active_messages: Maximum messages in active memory
      max_active_tokens: Maximum tokens in active memory

    Returns:
      AgentReference for interacting with the started agent
    """

    validate_address(name)

    if tools is None:
      tools = []

    tool_specs, tools_dict = await prepare_tools(tools, node)

    from ..memory.memory import Memory

    memory = Memory(
      max_active_messages=max_active_messages,
      max_active_tokens=max_active_tokens,
      model=model,
    )
    await memory.initialize_database()

    def agent_creator():
      kwargs = {
        "node": node,
        "name": name,
        "instructions": instructions,
        "model": model,
        "tool_specs": tool_specs,
        "tools": tools_dict,
        "memory": memory,
      }
      if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
      if max_execution_time is not None:
        kwargs["max_execution_time"] = max_execution_time
      return Agent(**kwargs)

    await node.start_spawner(name, agent_creator, key_extractor, None, exposed_as)
    logger.debug(f"Successfully started agent {name}")

    return AgentReference(name, node, exposed_as)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def key_extractor(message):
  """
  Extract routing key for conversation-based worker partitioning.

  The spawner pattern allows one logical agent to have multiple worker instances
  for parallel processing. This function routes messages to workers by extracting
  "{scope}/{conversation}" keys.

  Routing guarantees:
    - Same conversation → same worker (maintains context)
    - Different conversations → parallel workers (better throughput)
    - Memory isolation preserved (workers handle distinct scope+conversation pairs)

  Args:
    message: JSON string with 'scope' and 'conversation' fields

  Returns:
    Routing key "{scope}/{conversation}" or None if fields missing

  Example:
    {"scope": "user123", "conversation": "chat1"} → "user123/chat1"
  """
  message = json.loads(message)
  scope = message.get("scope")
  conversation = message.get("conversation")
  if conversation:
    if scope:
      return f"{scope}/{conversation}"
  return None


async def prepare_tools(tools: List[InvokableTool], node: Node):
  """
  Prepare tools for agent use by configuring node references and extracting specs.

  Transforms tool list into two data structures:
    1. tool_specs: Model-compatible JSON schemas (for model to see available tools)
    2. tools_dict: Name→tool mapping (for fast lookup during tool execution)

  Automatically includes built-in tools:
    - ask_user_for_input: Request input from user (human-in-the-loop)

  Args:
    tools: List of InvokableTool instances to prepare
    node: Node instance to inject into tools that need it

  Returns:
    Tuple of (tool_specs, tools_dict):
      - tool_specs: List of JSON schema dicts for model consumption
      - tools_dict: Dict mapping tool names to InvokableTool instances
  """
  from .builtin_tools import AskUserForInputTool

  # Add built-in tools
  builtin_tools = [AskUserForInputTool()]

  tool_specs = []
  tools_dict = {}

  if tools is None:
    tools = []

  # Combine built-in tools with user-provided tools
  all_tools = builtin_tools + tools

  for tool in all_tools:
    if hasattr(tool, "node"):
      tool.node = node

    spec = await tool.spec()
    tool_specs.append(spec)

    tool_name = tool.name if hasattr(tool, "name") else str(tool)
    tools_dict[tool_name] = tool

  return tool_specs, tools_dict


def system_message(message: str) -> dict:
  """
  Create a system message dict for the memory system.

  System messages establish agent instructions and behavior. They:
    - Appear first in conversation history
    - Set behavioral context for the agent
    - Are not typically shown to end users

  Args:
    message: Instruction text (e.g., "You are a helpful assistant")

  Returns:
    Dict with 'role', 'content', 'phase' in memory-compatible format
  """
  return {"role": "system", "content": {"text": message, "type": "text"}, "phase": "system"}


class ConversationResponse:
  """
  Helper for generating properly sequenced conversation response snippets.

  Streaming responses are delivered as numbered chunks. This class:
    - Maintains thread-safe counter for chunk sequencing
    - Generates StreamedConversationSnippet or ConversationSnippet objects
    - Prevents duplicate finish signals

  Thread safety: Counter protected by async lock prevents race conditions when
  multiple coroutines concurrently process chunks for the same conversation.

  Attributes:
    scope: User/tenant identifier
    conversation: Conversation thread identifier
    stream: Whether to generate streaming or non-streaming snippets
    counter: Incrementing chunk sequence number
    _counter_lock: Async lock protecting counter increments
  """

  def __init__(self, scope: str, conversation: str, stream: bool = False):
    self.scope = scope
    self.conversation = conversation
    self.stream = stream
    self.counter = 0
    self._counter_lock = asyncio.Lock()

  async def make_snippet(self, message: ConversationMessage, finished: bool = False):
    """
    Create a conversation snippet for a single message.

    Args:
      message: The message to wrap in a snippet
      finished: Whether this is the final message in the sequence

    Returns:
      StreamedConversationSnippet (streaming) or ConversationSnippet (non-streaming)
    """
    if self.stream:
      async with self._counter_lock:
        self.counter += 1
        current_counter = self.counter

      return StreamedConversationSnippet(
        ConversationSnippet(self.scope, self.conversation, [message]), current_counter, finished
      )
    else:
      return ConversationSnippet(self.scope, self.conversation, [message])

  async def make_finished_snippet(self):
    """
    Create an empty finish snippet to signal stream completion.

    Used when stream completes without a final message (e.g., after tool execution).
    Empty snippet with finished=True signals client that stream is done.

    Returns:
      StreamedConversationSnippet with empty message list and finished=True
    """
    async with self._counter_lock:
      self.counter += 1
      current_counter = self.counter

    return StreamedConversationSnippet(ConversationSnippet(self.scope, self.conversation, []), current_counter, True)
