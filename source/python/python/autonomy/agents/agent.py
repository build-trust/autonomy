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
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
import uuid

from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, TypedDict, NotRequired

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
  GetConversationStateRequest,
  GetConversationStateResponse,
  GetIdentifierRequest,
  GetIdentifierResponse,
  MessageConverter,
  Phase,
  StreamedConversationSnippet,
  ToolCall,
  ToolCallResponseMessage,
)

from .context import (
  ContextTemplate,
  ContextSection,
  SystemInstructionsSection,
  ConversationHistorySection,
  AdditionalContextSection,
  FrameworkInstructionsSection,
  create_default_template,
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
MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT = 1000
MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT = 100000

# Filesystem defaults
FILESYSTEM_VISIBILITY_DEFAULT = "conversation"


# =============================================================================
# CONFIGURATION TYPE DEFINITIONS
# =============================================================================


class SubagentConfig(TypedDict):
  """Configuration for a subagent that can be started by a parent agent.

  Only 'role' and 'instructions' are required. All other fields are optional.

  Example:
    subagent_config = {
      "role": "researcher",
      "instructions": "You are a research specialist.",
      "model": Model(name="claude-sonnet-4-v1"),
      "auto_start": False,
      "runner_filter": "runner",
    }
  """

  role: str
  instructions: str
  model: NotRequired[Model]
  tools: NotRequired[List]
  max_iterations: NotRequired[int]
  max_execution_time: NotRequired[float]
  max_messages_in_short_term_memory: NotRequired[int]
  max_tokens_in_short_term_memory: NotRequired[int]
  enable_long_term_memory: NotRequired[bool]
  auto_start: NotRequired[bool]
  runner_filter: NotRequired[str]


class AgentConfig(TypedDict):
  """Configuration for starting a single agent.

  Only 'instructions' is required. All other fields are optional
  and will use sensible defaults if not provided.

  Example:
    config = {
      "instructions": "Help users with their questions",
      "name": "support-bot",
      "model": Model(name="claude-sonnet-4-v1"),
      "tools": [search_tool, database_tool],
      "max_iterations": 500,
      "subagents": {
        "researcher": {
          "role": "researcher",
          "instructions": "Research specialist",
        }
      }
    }
    agent = await Agent.start_from_config(node=node, config=config)
  """

  instructions: str
  name: NotRequired[str]
  model: NotRequired[Model]
  tools: NotRequired[List]
  context_template: NotRequired[ContextTemplate]
  max_iterations: NotRequired[int]
  max_execution_time: NotRequired[float]
  max_messages_in_short_term_memory: NotRequired[int]
  max_tokens_in_short_term_memory: NotRequired[int]
  enable_long_term_memory: NotRequired[bool]
  enable_ask_for_user_input: NotRequired[bool]
  enable_filesystem: NotRequired[bool]
  filesystem_visibility: NotRequired[str]
  subagents: NotRequired[Dict[str, SubagentConfig]]
  subagent_runner_filter: NotRequired[str]
  exposed_as: NotRequired[str]


class AgentManyConfig(TypedDict):
  """Configuration for starting multiple agents.

  Both 'instructions' and 'number_of_agents' are required.
  Note: 'name' and 'exposed_as' are not supported for multiple agents
  (names are auto-generated per agent).

  Example:
    config = {
      "instructions": "Process customer requests",
      "number_of_agents": 10,
      "model": Model(name="claude-sonnet-4-v1"),
      "max_iterations": 500,
    }
    agents = await Agent.start_many_from_config(node=node, config=config)
  """

  instructions: str
  number_of_agents: int
  model: NotRequired[Model]
  tools: NotRequired[List]
  context_template: NotRequired[ContextTemplate]
  max_iterations: NotRequired[int]
  max_execution_time: NotRequired[float]
  max_messages_in_short_term_memory: NotRequired[int]
  max_tokens_in_short_term_memory: NotRequired[int]
  enable_long_term_memory: NotRequired[bool]
  enable_ask_for_user_input: NotRequired[bool]


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
#         │ [start]
#         ▼
#     ┌──────────┐
#     │          │◀────────────────────────────┐
#     │ THINKING │◀────────────────┐           │
#     │          │                 │           │
#     └┬────┬────┘                 │           │
#      │    │                      ▲           ▲
#      │    │ [has tool calls]     │           │
#      │    ▼                      │           │
#      │  ┌────────┐               │           │
#      │  │        │               │           │
#      │  │ ACTING │───────────────┘           │
#      │  │        │                           │
#      │  └───┬────┘                           │
#      │      │                                │
#      │      │ [ask_user_for_input called]    │
#      │      ▼                                │
#      │  ╔══════════════════════╗             │
#      │  ║                      ║             │
#      │  ║  WAITING_FOR_INPUT   ║             │
#      │  ║                      ║             │
#      │  ╚══════╤═══════════════╝             ▲
#      │         │                             │
#      │         │                             │
#      │         │  USER RESPONDS              │
#      │         ▼                             │
#      │         └─────────────────────────────┘
#      │
#      │
#      │ [no tool calls OR error OR max iterations]
#      │
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
    self.paused_at = None  # Track when we paused to exclude from execution time
    self.total_paused_time = 0.0  # Cumulative time spent paused

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

    # Check execution time limit (excluding paused time)
    current_time = time.time()
    elapsed_time = current_time - self.start_time - self.total_paused_time

    if elapsed_time > self.max_execution_time:
      logger.error(f"[STATE_MACHINE] Exceeded maximum execution time ({self.max_execution_time}s)")
      yield Error(f"Agent exceeded maximum execution time ({self.max_execution_time:.1f}s)")
      self.state = State.DONE
      return

    logger.debug(f"[STATE_MACHINE→{self.state.name}] iteration={self.iteration}, elapsed={elapsed_time:.1f}s")

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
      logger.info("[STATE:THINKING] State machine interrupted")
      yield Error("Agent was interrupted by new message")
      self.state = State.DONE
      return

    self.iteration += 1
    if self.iteration >= self.agent.max_iterations:
      logger.warning(f"[STATE:THINKING] Reached max_iterations ({self.agent.max_iterations})")
      yield Error(str(RuntimeError("Reached max_iterations")))
      self.state = State.DONE
      return

    self.tool_calls = []
    response_received = False
    finish_signal_received = False
    chunks_processed = 0

    logger.debug(f"[STATE:THINKING] Starting model completion, iteration {self.iteration}")

    try:
      # Call the model and process response chunks
      async for err, finished, model_response in self.agent.complete_chat(
        self.scope, self.conversation, stream=self.stream
      ):
        chunks_processed += 1
        logger.debug(
          f"[STATE:THINKING] Received chunk {chunks_processed}: err={err}, finished={finished}, response_type={type(model_response)}"
        )

        if err:  # Propagate model error and terminate state machine
          logger.error(f"[STATE:THINKING] Model completion error: {err}")
          yield err
          self.state = State.DONE
          return

        response_received = True

        # Remember the model response
        logger.debug(f"[STATE:THINKING] Remembering model response")
        await self.agent.remember(self.scope, self.conversation, model_response)

        # Handle tool calls
        if len(model_response.tool_calls) > 0:
          logger.debug(
            f"[STATE:THINKING] Model requested {len(model_response.tool_calls)} tool calls, transitioning to ACTING"
          )
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
          logger.debug(f"[STATE:THINKING] Processing finished response: stream={self.stream}")

          if self.stream:
            if model_response.content or model_response.tool_calls:
              yield await self.streaming_response.make_snippet(model_response, finished=True)
            self.final_content_sent = True
          else:
            # Accumulate final message
            self.whole_response.messages.append(model_response)

          if self.state == State.ACTING:
            logger.debug("[STATE:THINKING] Transitioning to ACTING")
            return
          else:
            logger.debug("[STATE:THINKING] Transitioning to DONE")
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
            logger.warning(f"[STATE:THINKING] Streaming exceeded max chunks ({chunks_processed}), forcing finish")
            if self.stream and not self.final_content_sent:
              # Send empty finish chunk if needed
              from ..nodes.message import AssistantMessage

              empty_response = AssistantMessage(content="", phase=Phase.EXECUTING)
              yield await self.streaming_response.make_snippet(empty_response, finished=True)
              self.final_content_sent = True
            break

    except Exception as e:
      logger.error(f"[STATE:THINKING] Exception in model calling: {e}")
      logger.debug(f"[STATE:THINKING] Traceback: {traceback.format_exc()}")
      yield Error(f"Model completion failed: {str(e)}")
      self.state = State.DONE
      return

    logger.debug(
      f"[STATE:THINKING] Stream complete: chunks={chunks_processed}, response_received={response_received}, "
      f"finish_signal={finish_signal_received}"
    )

    if not response_received:
      logger.error("[STATE:THINKING] No response received from model")
      yield Error("No response received from model")
      self.state = State.DONE
      return

    # Incomplete streaming: Network issues, model errors, or chunking problems
    if response_received and not finish_signal_received:
      logger.warning(
        f"[STATE:THINKING] Stream incomplete - no finish signal after {chunks_processed} chunks, forcing completion"
      )

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
      logger.warning(f"[STATE:THINKING] Forcing DONE state after {chunks_processed} chunks")
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
      logger.info("[STATE:ACTING] State machine interrupted")
      yield Error("Agent was interrupted by new message")
      self.state = State.DONE
      return

    logger.debug(f"[STATE:ACTING] Processing {len(self.tool_calls)} tool calls")

    for tool_call in self.tool_calls:
      error, tool_call_response, raw_result = await self.agent.call_tool(tool_call)

      if error:
        # Continue so model can see error and potentially recover
        logger.error(f"[STATE:ACTING] Tool call failed: {error}")
        warn(f"MCP call failed: {error}")

      # Check if this is ask_user_for_input with waiting marker
      if not error and isinstance(raw_result, dict) and raw_result.get("_waiting_for_input"):
        # Extract prompt for user
        self.waiting_prompt = raw_result.get("prompt", "Waiting for input...")
        self.pending_tool_call_id = tool_call.id

        logger.info(f"[STATE:ACTING] Agent requesting user input: {self.waiting_prompt}")

        # Transition to waiting state
        self.state = State.WAITING_FOR_INPUT

        # Send the prompt to user as assistant message
        from ..nodes.message import AssistantMessage

        waiting_message = AssistantMessage(content=self.waiting_prompt, phase=Phase.WAITING_FOR_INPUT)
        await self.agent.remember(self.scope, self.conversation, waiting_message)

        if self.stream:
          yield await self.streaming_response.make_snippet(waiting_message, finished=True)
        else:
          self.whole_response.messages.append(waiting_message)

        # Mark as paused - run() will detect this
        self.is_paused = True
        self.paused_at = time.time()  # Record when we paused
        logger.debug(f"[STATE_MACHINE] Paused at {self.paused_at}, total paused time so far: {self.total_paused_time}s")
        return

      # Normal tool call - store result and continue
      # the tool_call_response contains the error message if the tool call failed
      logger.debug(f"[STATE:ACTING] Remembering tool response")
      await self.agent.remember(self.scope, self.conversation, tool_call_response)

      if self.stream:
        yield await self.streaming_response.make_snippet(tool_call_response)
      else:
        self.whole_response.messages.append(tool_call_response)

    logger.debug(f"[STATE:ACTING] All tool calls complete, transitioning to THINKING")
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

    logger.debug(
      f"[STATE_MACHINE] Starting run: scope={self.scope}, conversation={self.conversation}, stream={self.stream}"
    )

    try:
      # Add overall timeout protection for the entire state machine run
      # Note: This timeout includes paused time, but transition() checks exclude paused time
      async with asyncio.timeout(self.max_execution_time):
        while self.state != State.DONE:
          # Check for interruption
          if self.interrupted:
            logger.info("[STATE_MACHINE] Interrupted, terminating")
            yield Error("Agent was interrupted by new message")
            self.state = State.DONE
            break

          async for result in self.transition():
            yield result

          # Check if we're paused
          if self.is_paused:
            logger.info(f"[STATE_MACHINE] Paused in {self.state.name} state")
            # For non-streaming mode, yield accumulated response before pausing
            if not self.stream:
              yield self.whole_response
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

    elapsed_time = time.time() - self.start_time - self.total_paused_time
    logger.info(
      f"State machine completed: {self.iteration} iterations in {elapsed_time:.1f}s (paused for {self.total_paused_time:.1f}s)"
    )


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
    # Single agent - direct parameters
    agent = await Agent.start(
      node=node,
      name="customer-support",
      instructions="Help users with their questions",
      model=Model(name="claude-sonnet-4-v1"),
      tools=[search_tool, database_tool]
    )

    # Single agent - from config dict
    config = {
      "name": "customer-support",
      "instructions": "Help users with their questions",
      "model": Model(name="claude-sonnet-4-v1"),
      "tools": [search_tool, database_tool],
    }
    agent = await Agent.start_from_config(node=node, config=config)

    # Multiple agents - direct parameters
    agents = await Agent.start_many(
      node=node,
      instructions="Analyze customer sentiment",
      number_of_agents=10,
      model=Model(name="claude-sonnet-4-v1")
    )

    # Multiple agents - from config dict
    config = {
      "instructions": "Analyze customer sentiment",
      "number_of_agents": 10,
      "model": Model(name="claude-sonnet-4-v1"),
    }
    agents = await Agent.start_many_from_config(node=node, config=config)

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
    subagent_configs: Optional[Dict[str, SubagentConfig]] = None,
    parent_agent_name: Optional[str] = None,
    context_template: Optional[ContextTemplate] = None,
    enable_ask_for_user_input: bool = False,
    enable_filesystem: bool = False,
    filesystem_visibility: str = FILESYSTEM_VISIBILITY_DEFAULT,
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

    # Subagent management
    self.subagent_configs = subagent_configs or {}
    self.parent_agent_name = parent_agent_name
    self.subagents = None  # Will be initialized if subagents configured
    self.subagent_runner_filter = None  # Will be set if provided in config

    # State machine tracking for pause/resume support
    # Since each Agent instance handles one (scope, conversation) pair,
    # we only need to track one active state machine at a time
    self._active_state_machine: Optional[StateMachine] = None

    # Context template for organizing model input context into sections
    # If no custom template is provided, use the default template with framework instructions
    if context_template is None:
      self.context_template = create_default_template(
        memory,
        memory.instructions,
        enable_ask_for_user_input=enable_ask_for_user_input,
        enable_filesystem=enable_filesystem,
        filesystem_visibility=filesystem_visibility,
        subagent_configs=subagent_configs,
      )
    else:
      self.context_template = context_template

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
        GetConversationStateRequest: self.handle__get_conversation_state_request,
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

  async def handle__get_conversation_state_request(
    self, message: GetConversationStateRequest
  ) -> GetConversationStateResponse:
    """
    Get the current state of a conversation.

    Args:
      message: Request containing scope and conversation identifiers

    Returns:
      Response containing conversation state information:
      - is_paused: True if waiting for user input
      - phase: Current state machine phase
      - message_count: Number of messages in conversation
    """
    # Check if there's an active state machine
    state_machine = self._active_state_machine

    # Get messages from memory
    messages = await self.memory.get_messages_only(self._memory_scope(message.scope), message.conversation)

    if state_machine is None:
      # No active state machine - conversation is either new or completed
      return GetConversationStateResponse(
        is_paused=False,
        phase="done",
        message_count=len(messages),
        scope=message.scope,
        conversation=message.conversation,
      )

    # Active state machine exists - check if it matches the requested conversation
    # The active state machine stores the raw scope (not prefixed), so compare with message.scope directly
    if state_machine.scope == message.scope and state_machine.conversation == message.conversation:
      # This is the active conversation
      is_paused = state_machine.state == State.WAITING_FOR_INPUT
      phase = state_machine.state.value  # Get string value from enum

      return GetConversationStateResponse(
        is_paused=is_paused,
        phase=phase,
        message_count=len(messages),
        scope=message.scope,
        conversation=message.conversation,
      )
    else:
      # Different conversation is active, so this one is completed or not started
      return GetConversationStateResponse(
        is_paused=False,
        phase="done",
        message_count=len(messages),
        scope=message.scope,
        conversation=message.conversation,
      )

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

      # Create tool result with user's response
      # Note: We don't add user messages to memory here because the response
      # should only appear as a tool result, not as a separate user message
      user_response = "\n".join(
        [
          msg.content.text if hasattr(msg.content, "text") else str(msg.content)
          for msg in messages
          if hasattr(msg, "content") and msg.content
        ]
      )

      # Complete the pending ask_user_for_input tool call
      from ..nodes.message import ToolCallResponseMessage

      tool_result = ToolCallResponseMessage(
        tool_call_id=machine.pending_tool_call_id or "unknown", name="ask_user_for_input", content=user_response
      )
      await self.remember(scope, conversation, tool_result)

      # Create NEW response helper for resume (resets part_nb counter)
      # This is critical - the client creates a new request/mailbox and expects part_nb to start from 1
      resume_response = ConversationResponse(scope, conversation, stream)
      machine.streaming_response = resume_response
      # Note: whole_response stays as ConversationSnippet for non-streaming mode

      # Reset final_content_sent to allow finish fallbacks in resumed stream
      machine.final_content_sent = False

      if stream:
        yield await machine.streaming_response.make_snippet(tool_result)
      else:
        machine.whole_response.messages.append(tool_result)

      # Resume the state machine
      machine.is_paused = False

      # Update total paused time
      if machine.paused_at is not None:
        paused_duration = time.time() - machine.paused_at
        machine.total_paused_time += paused_duration
        logger.debug(
          f"[STATE_MACHINE] Resuming after {paused_duration}s pause, total paused time: {machine.total_paused_time}s"
        )
        machine.paused_at = None

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

    logger.debug(
      f"[MEMORY→WRITE] scope={scope}, conversation={conversation}, message_type={message.get('role', 'unknown')}"
    )
    if logger.isEnabledFor(10):  # DEBUG level
      logger.debug(f"[MEMORY→WRITE] Full message: {json.dumps(message, indent=2)}")

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
    messages = await self.memory.get_messages(self._memory_scope(scope), conversation)
    logger.debug(f"[MEMORY→READ] scope={scope}, conversation={conversation}, message_count={len(messages)}")
    return messages

  async def determine_input_context(self, scope: str, conversation: str):
    """
    Build structured model input context from multiple organized sections.

    Uses the agent's context template to construct model input by combining:
    - System instructions (agent behavior and capabilities)
    - Conversation history (user messages, assistant responses, tool results)
    - Additional context (knowledge retrieval, external data)
    - Framework instructions (tool schemas, formatting guidelines)

    The context template organizes these sections in optimal order for model
    reasoning while respecting memory limits and token constraints.

    Flow:
      1. Apply context template to agent-scoped memory
      2. Retrieve raw message dicts from memory system
      3. Convert to typed ConversationMessage objects
      4. Return structured context ready for model consumption

    Args:
      scope: User/tenant scope identifier for memory isolation
      conversation: Conversation thread identifier

    Returns:
      List of ConversationMessage objects providing complete context for model,
      organized by template sections and ready for LLM processing

    Example context structure:
      [SystemMessage(instructions), UserMessage(query), AssistantMessage(response),
       ToolCallResponseMessage(result), FrameworkMessage(tool_schemas)]
    """

    # Build context using the template (applies to agent-scoped memory)
    raw_messages = await self.context_template.build_context(self._memory_scope(scope), conversation)
    logger.debug(f"[MEMORY→READ] Retrieved {len(raw_messages)} messages for model context")

    if logger.isEnabledFor(10):  # DEBUG level
      for idx, msg in enumerate(raw_messages):
        logger.debug(f"[MEMORY→READ] Message {idx}: role={msg.get('role', 'unknown')}")

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
      └─────┬───────┘
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

      logger.debug(
        f"[MODEL→CALL] scope={scope}, conversation={conversation}, stream={stream}, message_count={len(messages)}, tools_count={len(self.tool_specs)}"
      )
      if logger.isEnabledFor(10):  # DEBUG level
        logger.debug(
          f"[MODEL→CALL] Available tools: {[t.get('function', {}).get('name', 'unknown') for t in self.tool_specs]}"
        )
        for idx, msg in enumerate(messages):
          role = getattr(msg, "role", "unknown")
          content_preview = ""
          if hasattr(msg, "content") and msg.content:
            content_str = str(msg.content)
            content_preview = content_str[:100] + ("..." if len(content_str) > 100 else "")
          logger.debug(f"[MODEL→CALL] Input message {idx}: role={role}, content_preview={repr(content_preview)}")

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

        # Dictionary to accumulate tool calls across chunks by index
        accumulated_tool_calls: Dict[int, ToolCall] = {}

        logger.debug(f"[MODEL→STREAM] Starting streaming model call with {len(messages)} messages")

        try:
          # Execute streaming with timeout protection
          async with asyncio.timeout(timeout_seconds):
            async for chunk in self.model.complete_chat(
              messages, stream=True, tools=self.tool_specs, agent_name=self.name, scope=scope, conversation=conversation
            ):
              chunk_count += 1
              logger.debug(f"[MODEL→STREAM] Processing chunk {chunk_count}")
              if logger.isEnabledFor(10):  # DEBUG level
                logger.debug(f"[MODEL→STREAM] Raw chunk: {chunk}")

              # Prevent runaway streams
              if chunk_count > max_chunks:
                logger.error(f"Stream exceeded maximum chunks ({max_chunks}), terminating")
                break

              # Process streaming response chunks
              if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                finished = choice.finish_reason is not None if hasattr(choice, "finish_reason") else False
                last_finished = finished

                # --- Handle tool call chunks (check FIRST before content) ---
                if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                  for tc in choice.delta.tool_calls:
                    # Get the index to accumulate tool calls properly
                    tc_index = tc.index if hasattr(tc, "index") else 0

                    if tc_index in accumulated_tool_calls:
                      # Accumulate arguments from this chunk
                      if hasattr(tc, "function") and hasattr(tc.function, "arguments") and tc.function.arguments:
                        accumulated_tool_calls[tc_index].function.arguments += tc.function.arguments
                      logger.debug(
                        f"[MODEL←STREAM] Accumulated tool call chunk: index={tc_index}, total_args_length={len(accumulated_tool_calls[tc_index].function.arguments)}"
                      )
                    else:
                      # First chunk for this tool call
                      # Generate unique ID with collision detection (some models don't provide IDs)
                      call_id = tc.id if hasattr(tc, "id") and tc.id else None
                      if not call_id:
                        call_id = await self._generate_unique_tool_call_id()

                      tool_name = (
                        tc.function.name if hasattr(tc, "function") and hasattr(tc.function, "name") else "unknown"
                      )
                      logger.debug(
                        f"[MODEL←STREAM] New tool call chunk: index={tc_index}, id={call_id}, name={tool_name}"
                      )
                      if logger.isEnabledFor(10):  # DEBUG level
                        logger.debug(f"[MODEL←STREAM] Tool call details: {tc}")

                      tool_call = ToolCall(
                        id=call_id,
                        function=tc.function if hasattr(tc, "function") else None,
                        type="function",
                      )
                      accumulated_tool_calls[tc_index] = tool_call

                  # Don't yield individual chunks - wait for finish
                  has_content = True

                # --- Handle content chunks (text responses) ---
                elif hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                  content = choice.delta.content
                  logger.debug(f"[MODEL←STREAM] Delta content: {repr(content)}, finished: {finished}")

                  if content:  # Non-empty content chunk
                    has_content = True
                    response = AssistantMessage(content=content, phase=Phase.EXECUTING)
                    logger.debug(
                      f"[MODEL←STREAM] Yielding content chunk: finished={finished}, content_length={len(content)}"
                    )
                    yield None, finished, response

                  elif finished and (content == "" or content is None):
                    # Empty content with finish_reason - termination chunk
                    logger.debug("[MODEL←STREAM] Received empty finish chunk - streaming complete")
                    has_content = True
                    # Don't yield empty response if we have tool calls - they'll be yielded below
                    if len(accumulated_tool_calls) == 0:
                      # Send empty response to signal completion only if no tool calls
                      response = AssistantMessage(content="", phase=Phase.EXECUTING)
                      yield None, True, response
                      break
                    # If we have tool calls, fall through to yield them below

                # --- Handle legacy format (message instead of delta) ---
                elif finished and hasattr(choice, "message") and choice.message.content:
                  # Some older model APIs use 'message' instead of 'delta'
                  logger.debug(f"[MODEL←STREAM] Legacy format message: content_length={len(choice.message.content)}")
                  response = AssistantMessage(content=choice.message.content, phase=Phase.EXECUTING)
                  yield None, True, response
                  has_content = True

                # Break on finish signal - yield accumulated tool calls if any
                if finished:
                  if len(accumulated_tool_calls) > 0:
                    logger.debug(f"[MODEL←STREAM] Yielding {len(accumulated_tool_calls)} accumulated tool calls")
                    response = AssistantMessage(tool_calls=list(accumulated_tool_calls.values()), phase=Phase.EXECUTING)
                    yield None, True, response
                  logger.debug("[MODEL←STREAM] Breaking due to finish_reason")
                  break
              else:
                logger.warning(f"Unexpected chunk format in streaming response: {chunk}")

        except asyncio.TimeoutError:
          logger.error(f"[MODEL←STREAM] Timeout after {timeout_seconds} seconds")
          response = AssistantMessage(
            content="I apologize, but the response took too long to generate.", phase=Phase.EXECUTING
          )
          yield None, True, response
          return

        except Exception as stream_error:
          logger.error(f"[MODEL←STREAM] Error: {stream_error}")
          logger.debug(f"[MODEL←STREAM] Error traceback: {traceback.format_exc()}")
          response = AssistantMessage(
            content="I apologize, but I encountered an error generating the response.", phase=Phase.EXECUTING
          )
          yield None, True, response
          return

        # --- Post-stream validation and cleanup ---
        logger.debug(
          f"[MODEL←STREAM] Complete: chunks={chunk_count}, has_content={has_content}, last_finished={last_finished}"
        )

        if chunk_count == 0:
          # No chunks received at all
          logger.warning("[MODEL←STREAM] No streaming chunks received")
          response = AssistantMessage(
            content="I apologize, but I encountered an issue generating a response.", phase=Phase.EXECUTING
          )
          yield None, True, response

        elif not has_content and not last_finished:
          # We got chunks but no actual content and no finish signal
          logger.warning(f"[MODEL←STREAM] Incomplete: {chunk_count} chunks but no content or finish signal")
          response = AssistantMessage(content="I apologize, but the response was incomplete.", phase=Phase.EXECUTING)
          yield None, True, response

        elif has_content and not last_finished:
          # We got content but no finish signal - force completion
          logger.warning("[MODEL←STREAM] Had content but no finish signal, forcing completion")
          response = AssistantMessage(content="", phase=Phase.EXECUTING)
          yield None, True, response

      # =========================================================================
      # NON-STREAMING PATH: Complete response delivery
      # =========================================================================
      else:
        logger.debug("[MODEL→NON-STREAM] Calling model (non-streaming)")
        response = await self.model.complete_chat(
          messages, stream=False, tools=self.tool_specs, agent_name=self.name, scope=scope, conversation=conversation
        )

        if logger.isEnabledFor(10):  # DEBUG level
          logger.debug(f"[MODEL←NON-STREAM] Raw response: {response}")

        if hasattr(response, "choices") and len(response.choices) > 0:
          message = response.choices[0].message

          if hasattr(message, "tool_calls") and message.tool_calls:
            logger.debug(f"[MODEL←NON-STREAM] Received {len(message.tool_calls)} tool calls")
            tool_calls = []
            for tc in message.tool_calls:
              # Generate unique ID with collision detection (some models don't provide IDs)
              call_id = tc.id if hasattr(tc, "id") and tc.id else None
              if not call_id:
                call_id = await self._generate_unique_tool_call_id()

              tool_name = tc.function.name if hasattr(tc, "function") and hasattr(tc.function, "name") else "unknown"
              tool_args = tc.function.arguments if hasattr(tc, "function") and hasattr(tc.function, "arguments") else ""
              logger.debug(f"[MODEL←NON-STREAM] Tool call: id={call_id}, name={tool_name}")
              if logger.isEnabledFor(10):  # DEBUG level
                logger.debug(f"[MODEL←NON-STREAM] Tool arguments: {tool_args}")

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
            content_length = len(message.content) if message.content else 0
            logger.debug(f"[MODEL←NON-STREAM] Received text response: content_length={content_length}")
            if logger.isEnabledFor(10):  # DEBUG level
              logger.debug(f"[MODEL←NON-STREAM] Content: {message.content}")
            assistant_message = AssistantMessage(content=message.content or "", phase=Phase.EXECUTING)

          yield None, True, assistant_message

    except Exception as e:
      logger.error(f"[MODEL←ERROR] Error in complete_chat: {e}")
      logger.debug(f"[MODEL←ERROR] Traceback: {traceback.format_exc()}")
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

    logger.debug(f"[TOOL→CALL] id={tool_call.id}, name={tool_name}")
    if logger.isEnabledFor(10):  # DEBUG level
      logger.debug(f"[TOOL→CALL] Arguments: {tool_args}")

    try:
      if tool_name in self.tools:
        tool = self.tools[tool_name]
        logger.debug(f"[TOOL→CALL] Invoking tool: {tool_name}")

        start_time = time.time()
        result = await tool.invoke(tool_args)
        elapsed_time = time.time() - start_time

        result_str = str(result)
        result_length = len(result_str)
        logger.debug(
          f"[TOOL←RESULT] id={tool_call.id}, name={tool_name}, elapsed={elapsed_time:.3f}s, result_length={result_length}"
        )
        if logger.isEnabledFor(10):  # DEBUG level
          # Truncate very long results for logging
          result_preview = result_str[:500] + ("..." if result_length > 500 else "")
          logger.debug(f"[TOOL←RESULT] Result preview: {result_preview}")

        response = ToolCallResponseMessage(
          tool_call_id=tool_call.id, name=tool_name, content=result_str, phase=Phase.EXECUTING
        )
        return None, response, result
      else:
        error_msg = f"Tool '{tool_name}' not found"
        logger.error(f"[TOOL←ERROR] {error_msg}, id={tool_call.id}")
        logger.debug(f"[TOOL←ERROR] Available tools: {list(self.tools.keys())}")
        response = ToolCallResponseMessage(
          tool_call_id=tool_call.id, name=tool_name, content=error_msg, phase=Phase.EXECUTING
        )
        return error_msg, response, None

    except Exception as e:
      error_msg = f"Tool execution failed: {str(e)}"
      logger.error(f"[TOOL←ERROR] id={tool_call.id}, name={tool_name}, error={error_msg}")
      logger.debug(f"[TOOL←ERROR] Traceback: {traceback.format_exc()}")
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
    tools: Optional[List] = None,
    context_template: Optional[ContextTemplate] = None,
    max_iterations: Optional[int] = None,
    max_execution_time: Optional[float] = None,
    max_messages_in_short_term_memory: int = MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT,
    max_tokens_in_short_term_memory: Optional[int] = MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT,
    enable_long_term_memory: bool = False,
    enable_ask_for_user_input: bool = False,
    enable_filesystem: bool = False,
    filesystem_visibility: str = FILESYSTEM_VISIBILITY_DEFAULT,
    subagents: Optional[Dict[str, SubagentConfig]] = None,
    subagent_runner_filter: Optional[str] = None,
    exposed_as: Optional[str] = None,
  ):
    """
    Start a single agent on a node.

    Factory method that handles initialization, memory setup, and registration.

    Args:
      node: Node instance where the agent will run
      instructions: System instructions defining agent behavior
      name: Unique agent name (auto-generated if None)
      model: LLM to use (defaults to claude-sonnet-4-v1)
      tools: List of tools and/or tool factories. Can contain:
        - InvokableTool instances (static, shared across all scopes)
        - ToolFactory instances (create scope-specific tools automatically)
      max_iterations: Maximum state machine iterations
      max_execution_time: Maximum execution time in seconds
      context_template: Optional custom context template for organizing model input
      max_messages_in_short_term_memory: Maximum messages in short-term memory
      max_tokens_in_short_term_memory: Maximum tokens in short-term memory
      enable_long_term_memory: Enable persistent long-term memory (database)
      enable_ask_for_user_input: Enable ask_user_for_input tool
      enable_filesystem: Enable filesystem tools
      filesystem_visibility: Filesystem visibility level ("all", "agent", "scope", or "conversation")
      subagents: Optional dictionary of subagent configurations
      subagent_runner_filter: Optional filter for subagent runner selection
      exposed_as: Optional external name for HTTP exposure

    Returns:
      AgentReference for interacting with the started agent

    Example:
      from autonomy import Agent, Tool, FilesystemTools

      agent = await Agent.start(
        node=node,
        name="assistant",
        tools=[
          Tool(my_function),  # Static tool
          FilesystemTools(),  # Factory - creates scope-specific tools
        ],
      )
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
      context_template,
      max_iterations,
      max_execution_time,
      max_messages_in_short_term_memory,
      max_tokens_in_short_term_memory,
      enable_long_term_memory,
      enable_ask_for_user_input,
      enable_filesystem,
      filesystem_visibility,
      subagents,
      subagent_runner_filter,
      exposed_as,
    )

  @staticmethod
  async def start_many(
    node: Node,
    instructions: str,
    number_of_agents: int,
    model: Optional[Model] = None,
    tools: Optional[List] = None,
    context_template: Optional[ContextTemplate] = None,
    max_iterations: Optional[int] = None,
    max_execution_time: Optional[float] = None,
    max_messages_in_short_term_memory: int = MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT,
    max_tokens_in_short_term_memory: Optional[int] = MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT,
    enable_long_term_memory: bool = False,
    enable_ask_for_user_input: bool = False,
    enable_filesystem: bool = False,
    filesystem_visibility: str = FILESYSTEM_VISIBILITY_DEFAULT,
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
          tools: List of tools and/or tool factories. Can contain:
    - InvokableTool instances (static, shared across all scopes)
    - ToolFactory instances (create scope-specific tools automatically)
          context_template: Optional custom context template for organizing model input
          max_iterations: Maximum state machine iterations
          max_execution_time: Maximum execution time in seconds
          max_messages_in_short_term_memory: Maximum messages in short-term memory
          max_tokens_in_short_term_memory: Maximum tokens in short-term memory
          enable_long_term_memory: Enable persistent long-term memory (database)
          enable_ask_for_user_input: Enable ask_user_for_input tool
          enable_filesystem: Enable filesystem tools
          filesystem_visibility: Filesystem visibility level ("all", "agent", "scope", or "conversation")

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
        context_template,
        max_iterations,
        max_execution_time,
        max_messages_in_short_term_memory,
        max_tokens_in_short_term_memory,
        enable_long_term_memory,
        enable_ask_for_user_input,
        enable_filesystem,
        filesystem_visibility,
        None,  # subagent_configs not supported for multiple agents
        None,  # subagent_runner_filter not supported for multiple agents
        None,  # exposed_as not supported for multiple agents
      )
      agents.append(agent_ref)

    return agents

  @staticmethod
  async def start_from_config(
    node: Node,
    config: AgentConfig,
  ):
    """
    Start a single agent using dictionary-based configuration.

    This method provides an alternative to Agent.start() for cases where
    configuration is stored in dictionaries, loaded from files (YAML/JSON),
    or dynamically generated.

    Args:
      node: Node instance where the agent will run
      config: Dictionary containing agent configuration. Only 'instructions'
              is required; all other fields are optional.

    Returns:
      AgentReference for interacting with the started agent

    Example:
      # From a dictionary
      config = {
        "instructions": "Help users with their questions",
        "name": "support-bot",
        "model": Model(name="claude-sonnet-4-v1"),
        "tools": [search_tool, database_tool],
        "max_iterations": 500,
      }
      agent = await Agent.start_from_config(node=node, config=config)

      # From a YAML file
      import yaml
      with open("agent_config.yaml") as f:
        config = yaml.safe_load(f)
      agent = await Agent.start_from_config(node=node, config=config)

      # Config composition
      base_config = {"model": Model(name="claude-sonnet-4-v1")}
      specific_config = {**base_config, "instructions": "...", "name": "agent1"}
      agent = await Agent.start_from_config(node=node, config=specific_config)
    """
    # Extract required fields
    instructions = config["instructions"]

    # Extract optional fields with defaults
    name = config.get("name")
    model = config.get("model")
    tools = config.get("tools")
    context_template = config.get("context_template")
    max_iterations = config.get("max_iterations")
    max_execution_time = config.get("max_execution_time")
    max_messages_in_short_term_memory = config.get(
      "max_messages_in_short_term_memory", MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT
    )
    max_tokens_in_short_term_memory = config.get(
      "max_tokens_in_short_term_memory", MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT
    )
    enable_long_term_memory = config.get("enable_long_term_memory", False)
    enable_ask_for_user_input = config.get("enable_ask_for_user_input", False)
    enable_filesystem = config.get("enable_filesystem", False)
    filesystem_visibility = config.get("filesystem_visibility", FILESYSTEM_VISIBILITY_DEFAULT)
    subagents = config.get("subagents")
    subagent_runner_filter = config.get("subagent_runner_filter")
    exposed_as = config.get("exposed_as")

    # Delegate to existing start() method
    return await Agent.start(
      node=node,
      instructions=instructions,
      name=name,
      model=model,
      tools=tools,
      context_template=context_template,
      max_iterations=max_iterations,
      max_execution_time=max_execution_time,
      max_messages_in_short_term_memory=max_messages_in_short_term_memory,
      max_tokens_in_short_term_memory=max_tokens_in_short_term_memory,
      enable_long_term_memory=enable_long_term_memory,
      enable_ask_for_user_input=enable_ask_for_user_input,
      enable_filesystem=enable_filesystem,
      filesystem_visibility=filesystem_visibility,
      subagents=subagents,
      subagent_runner_filter=subagent_runner_filter,
      exposed_as=exposed_as,
    )

  @staticmethod
  async def start_many_from_config(
    node: Node,
    config: AgentManyConfig,
  ):
    """
    Start multiple agents using dictionary-based configuration.

    This method provides an alternative to Agent.start_many() for cases where
    configuration is stored in dictionaries, loaded from files (YAML/JSON),
    or dynamically generated.

    Args:
      node: Node instance where the agents will run
      config: Dictionary containing agent configuration. Both 'instructions'
              and 'number_of_agents' are required.

    Returns:
      List of AgentReference objects for the started agents

    Example:
      # From a dictionary
      config = {
        "instructions": "Process customer requests",
        "number_of_agents": 10,
        "model": Model(name="claude-sonnet-4-v1"),
        "max_iterations": 500,
      }
      agents = await Agent.start_many_from_config(node=node, config=config)

      # From environment-based config
      import os
      config = {
        "instructions": os.getenv("AGENT_INSTRUCTIONS"),
        "number_of_agents": int(os.getenv("AGENT_COUNT", "5")),
      }
      agents = await Agent.start_many_from_config(node=node, config=config)
    """
    # Extract required fields
    instructions = config["instructions"]
    number_of_agents = config["number_of_agents"]

    # Extract optional fields with defaults
    model = config.get("model")
    tools = config.get("tools")
    context_template = config.get("context_template")
    max_iterations = config.get("max_iterations")
    max_execution_time = config.get("max_execution_time")
    max_messages_in_short_term_memory = config.get(
      "max_messages_in_short_term_memory", MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT
    )
    max_tokens_in_short_term_memory = config.get(
      "max_tokens_in_short_term_memory", MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT
    )
    enable_long_term_memory = config.get("enable_long_term_memory", False)
    enable_ask_for_user_input = config.get("enable_ask_for_user_input", False)
    enable_filesystem = config.get("enable_filesystem", False)
    filesystem_visibility = config.get("filesystem_visibility", FILESYSTEM_VISIBILITY_DEFAULT)

    # Delegate to existing start_many() method
    return await Agent.start_many(
      node=node,
      instructions=instructions,
      number_of_agents=number_of_agents,
      model=model,
      tools=tools,
      context_template=context_template,
      max_iterations=max_iterations,
      max_execution_time=max_execution_time,
      max_messages_in_short_term_memory=max_messages_in_short_term_memory,
      max_tokens_in_short_term_memory=max_tokens_in_short_term_memory,
      enable_long_term_memory=enable_long_term_memory,
      enable_ask_for_user_input=enable_ask_for_user_input,
      enable_filesystem=enable_filesystem,
      filesystem_visibility=filesystem_visibility,
    )

  @staticmethod
  async def start_agent_impl(
    node: Node,
    name: str,
    instructions: str,
    model: Model,
    tools: Optional[List],
    context_template: Optional[ContextTemplate],
    max_iterations: Optional[int],
    max_execution_time: Optional[float],
    max_messages_in_short_term_memory: int = MAX_MESSAGES_IN_SHORT_TERM_MEMORY_DEFAULT,
    max_tokens_in_short_term_memory: Optional[int] = MAX_TOKENS_IN_SHORT_TERM_MEMORY_DEFAULT,
    enable_long_term_memory: bool = False,
    enable_ask_for_user_input: bool = False,
    enable_filesystem: bool = False,
    filesystem_visibility: str = FILESYSTEM_VISIBILITY_DEFAULT,
    subagent_configs: Optional[Dict[str, SubagentConfig]] = None,
    subagent_runner_filter: Optional[str] = None,
    exposed_as: Optional[str] = None,
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
      tools: List of tools and/or tool factories
      context_template: Optional custom context template for organizing model input
      max_iterations: Maximum state machine iterations
      max_execution_time: Maximum execution time in seconds
      max_messages_in_short_term_memory: Maximum messages in short-term memory
      max_tokens_in_short_term_memory: Maximum tokens in short-term memory
      enable_long_term_memory: Enable persistent long-term memory (database)
      enable_ask_for_user_input: Enable ask_user_for_input tool
      enable_filesystem: Enable filesystem tools
      filesystem_visibility: Filesystem visibility level ("all", "agent", "scope", or "conversation")
      subagent_configs: Optional dictionary of subagent configurations
      subagent_runner_filter: Optional filter for subagent runner selection
      exposed_as: Optional external name for HTTP exposure

    Returns:
      AgentReference for interacting with the started agent
    """

    validate_address(name)

    if tools is None:
      tools = []

    # Add builtin filesystem tools if enabled (BEFORE factory detection)
    if enable_filesystem:
      from ..tools.filesystem import FilesystemTools

      tools = [FilesystemTools(visibility=filesystem_visibility)] + tools

    # Separate static tools from tool factories
    # Static tools: Shared across all workers (same instance for everyone)
    # Tool factories: Create isolated instances per worker (scope/conversation specific)
    #
    # Examples:
    # - Tool(get_time) → Static (shared)
    # - FilesystemTools() → Factory (creates isolated instances)
    static_tools = []
    tool_factories = []

    for tool in tools:
      # Detect factories by checking for create_tools method (ToolFactory protocol)
      # FilesystemTools implements this when visibility requires scope/conversation context
      if hasattr(tool, "create_tools") and callable(getattr(tool, "create_tools")):
        tool_factories.append(tool)
      else:
        # Regular tools are static and shared across all workers
        static_tools.append(tool)

    # Prepare static tools (shared across all workers/scopes/conversations)
    tool_specs, tools_dict = await prepare_tools(static_tools, node, enable_ask_for_user_input)

    from ..memory.memory import Memory

    memory = Memory(
      max_messages_in_short_term_memory=max_messages_in_short_term_memory,
      max_tokens_in_short_term_memory=max_tokens_in_short_term_memory,
      model=model,
      enable_long_term_memory=enable_long_term_memory,
    )
    await memory.initialize_database()

    def agent_creator(scope: Optional[str], conversation: Optional[str]):
      """
      Create an agent instance with scope/conversation-specific tools.

      Called by the spawner for each unique (scope, conversation) pair.
      This enables:
      - FilesystemTools to create isolated instances based on visibility level
      - Different users/conversations to have separate tool instances
      - Proper multi-tenant isolation

      Args:
        scope: User/tenant identifier (e.g., "user-alice")
        conversation: Session identifier (e.g., "chat-1")

      Returns:
        Agent instance with both static and scope-specific tools

      Execution Context (Rust ↔ Python):
        This function is called from Rust via PyO3, executing in a special context:

        1. Tokio async runtime detects need for new worker
        2. Spawner calls spawn_blocking() → creates new OS thread
        3. In blocking thread: Python::with_gil() → acquires Python GIL
        4. worker_constructor.call1(py, (scope, conversation)) → calls THIS function
          ┌─────────────────────────────────────────────────────────┐
          │ THIS FUNCTION EXECUTES HERE                             │
          │ - In a Rust-spawned OS thread (not main Python thread)  │
          │ - GIL is held (safe Python execution)                   │
          │ - NO asyncio event loop exists in this thread           │
          │ - Must create our own event loop to run async code      │
          └─────────────────────────────────────────────────────────┘
        5. Returns Agent object back through PyO3 → Rust gets PyObject wrapper
        6. Worker starts, ready to handle messages

      Why we need manual event loop management:
        - spawn_blocking threads don't have asyncio event loops
        - prepare_tools() is async and needs an event loop to run
        - We create a temporary event loop, run our async code, then clean up
      """
      # Create scope/conversation-specific tools from factories
      # Factories receive agent_name, scope, and conversation to determine isolation level
      # Example: FilesystemTools(visibility="conversation") creates isolated filesystem at:
      #   /tmp/agent-files/{agent_name}/{scope}/{conversation}/
      scope_specific_tools = []
      for factory in tool_factories:
        factory_tools = factory.create_tools(scope, conversation, agent_name=name)
        scope_specific_tools.extend(factory_tools)

      # Combine static tools (shared) with scope-specific tools (isolated)
      import asyncio

      all_tools = static_tools + scope_specific_tools

      # Prepare all tools for this specific agent instance
      # Create temporary event loop for this Rust-spawned thread (see detailed flow above)
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)
      try:
        scope_tool_specs, scope_tools_dict = loop.run_until_complete(
          prepare_tools(all_tools, node, enable_ask_for_user_input)
        )

        kwargs = {
          "node": node,
          "name": name,
          "instructions": instructions,
          "model": model,
          "tool_specs": scope_tool_specs,
          "tools": scope_tools_dict,
          "memory": memory,
        }
        if max_iterations is not None:
          kwargs["max_iterations"] = max_iterations
        if max_execution_time is not None:
          kwargs["max_execution_time"] = max_execution_time
        if subagent_configs is not None:
          kwargs["subagent_configs"] = subagent_configs
        if subagent_runner_filter is not None:
          kwargs["subagent_runner_filter"] = subagent_runner_filter
        if context_template is not None:
          kwargs["context_template"] = context_template
        if enable_ask_for_user_input:
          kwargs["enable_ask_for_user_input"] = enable_ask_for_user_input
        if enable_filesystem:
          kwargs["enable_filesystem"] = enable_filesystem
          kwargs["filesystem_visibility"] = filesystem_visibility

        agent_instance = Agent(**kwargs)

        # Initialize Subagents manager if subagents are configured
        if subagent_configs:
          from ..tools.subagents import Subagents
          from ..agents.builtin_tools import (
            StartSubagentTool,
            DelegateToSubagentTool,
            DelegateToSubagentsParallelTool,
            ListSubagentsTool,
            StopSubagentTool,
          )

          # Set runner filter BEFORE creating Subagents so it's available in __init__
          agent_instance.subagent_runner_filter = subagent_runner_filter
          agent_instance.subagents = Subagents(agent_instance)

          # Register subagent management tools
          subagent_tools = [
            StartSubagentTool(agent_instance),
            DelegateToSubagentTool(agent_instance),
            DelegateToSubagentsParallelTool(agent_instance),
            ListSubagentsTool(agent_instance),
            StopSubagentTool(agent_instance),
          ]

          # Add subagent tools to the agent's tool specs and tools dict
          for tool in subagent_tools:
            spec = loop.run_until_complete(tool.spec())
            scope_tool_specs.append(spec)
            scope_tools_dict[tool.name] = tool

          # Update agent's tools
          agent_instance.tool_specs = scope_tool_specs
          agent_instance.tools = scope_tools_dict

        return agent_instance
      finally:
        # Close the event loop after all async operations are complete
        loop.close()

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


async def prepare_tools(tools: List[InvokableTool], node: Node, enable_ask_for_user_input: bool = False):
  """
  Prepare tools for agent use by configuring node references and extracting specs.

  Transforms tool list into two data structures:
    1. tool_specs: Model-compatible JSON schemas (for model to see available tools)
    2. tools_dict: Name→tool mapping (for fast lookup during tool execution)

  Automatically includes built-in tools:
    - ask_user_for_input: Request input from user (human-in-the-loop) - only if enable_ask_for_user_input is True
    - get_current_time_utc: Get current time in UTC timezone
    - get_current_time: Get current time in a specific timezone

  Args:
    tools: List of InvokableTool instances to prepare
    node: Node instance to inject into tools that need it
    enable_ask_for_user_input: Whether to include ask_user_for_input tool

  Returns:
    Tuple of (tool_specs, tools_dict):
      - tool_specs: List of JSON schema dicts for model consumption
      - tools_dict: Dict mapping tool names to InvokableTool instances
  """
  from .builtin_tools import AskUserForInputTool, GetCurrentTimeUtcTool, GetCurrentTimeTool

  # Add built-in tools
  builtin_tools = [
    GetCurrentTimeUtcTool(),
    GetCurrentTimeTool(),
  ]

  # Conditionally add human-in-the-loop tools
  if enable_ask_for_user_input:
    builtin_tools.insert(0, AskUserForInputTool())

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
