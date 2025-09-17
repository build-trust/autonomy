import json
import traceback

from typing import List
from enum import Enum

from copy import deepcopy
from typing import Optional, AsyncGenerator, Tuple, Dict

import secrets
import copy

from .conversation_response import ConversationResponse
from ..tools.protocol import InvokableTool
from .agent_memory_knowledge import AgentMemoryKnowledge
from ..knowledge.noop import NoopKnowledge
from ..nodes.node import Node
from ..logs.logs import InfoContext, DebugContext
from ..planning import Planner
from ..knowledge import KnowledgeProvider
from ..memory.memory import Memory
from ..models.model import Model
from ..nodes.message import (
  ConversationSnippet,
  StreamedConversationSnippet,
  ConversationRole,
  Error,
  GetIdentifierRequest,
  GetIdentifierResponse,
  MessageConverter,
  AgentReference,
  ConversationMessage,
  AssistantMessage,
  ToolCall,
  GetConversationsRequest,
  GetConversationsResponse,
  Phase,
  ToolCallResponseMessage,
)
from .names import validate_name

from ..autonomy_in_rust_for_python import warn


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

  async def transition(self):
    """Execute the current state and determine the next state."""

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
        yield self.streaming_response.make_snippet(next_step)
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

    self.agent.logger.debug(f"Starting model completion, iteration {self.iteration}")

    async for err, finished, model_response in self.agent.complete_chat(
      self.scope, self.conversation, self.contextual_knowledge, stream=self.stream
    ):
      self.agent.logger.debug(f"Received chunk: err={err}, finished={finished}, response_type={type(model_response)}")

      if err:
        self.agent.logger.error(f"Model completion error: {err}")
        yield err
        self.state = AgentState.FINISHED
        return

      response_received = True
      # Remember the model response
      await self.agent.remember(self.scope, self.conversation, model_response)

      if len(model_response.tool_calls) > 0:
        # Record the event of the tool being called
        if self.stream:
          yield self.streaming_response.make_snippet(model_response)
        else:
          self.whole_response.messages.append(model_response)
        # Store tool calls for later processing
        self.tool_calls.extend(model_response.tool_calls)
        self.state = AgentState.TOOL_CALLING
      elif finished:
        self.agent.logger.debug(
          f"Processing finished response: stream={self.stream}, has_content={bool(model_response.content)}, has_tool_calls={bool(model_response.tool_calls)}"
        )

        # either we are streaming and we finished, or we are expecting a single response
        if self.stream:
          if model_response.content or model_response.tool_calls:
            yield self.streaming_response.make_snippet(model_response)
        else:
          self.whole_response.messages.append(model_response)

        if self.state == AgentState.TOOL_CALLING:
          self.agent.logger.debug("Returning from MODEL_CALLING due to TOOL_CALLING state")
          return
        else:
          if self.agent.planner is None:
            self.agent.logger.debug("Transitioning to FINISHED (no planner)")
            self.state = AgentState.FINISHED
          else:
            self.agent.logger.debug("Transitioning to PLANNING")
            self.state = AgentState.PLANNING
          return
      else:
        # This is an intermediate streaming chunk (not finished)
        if self.stream:
          if model_response.content or model_response.tool_calls:
            yield self.streaming_response.make_snippet(model_response)
        else:
          self.whole_response.messages.append(model_response)
        # Important: Do NOT return here for streaming chunks - continue processing

    # Safety check: if we didn't receive any response or never got finished=True
    if not response_received:
      self.agent.logger.error("No response received from model")
      yield Error("No response received from model")
      self.state = AgentState.FINISHED
      return

    # If we reach here and state hasn't changed, force transition to avoid infinite loop
    if self.state == AgentState.MODEL_CALLING:
      self.agent.logger.warning(
        f"Agent stuck in MODEL_CALLING state after {self.iteration} iterations, forcing FINISHED. Response received: {response_received}"
      )
      self.state = AgentState.FINISHED

  async def _handle_tool_calling_state(self):
    """Handle the TOOL_CALLING state: Process tool calls from the model response."""

    for tool_call in self.tool_calls:
      error, tool_call_response = await self.agent.call_tool(tool_call)
      if error:
        warn(f"MCP call failed: {error}")

      # the tool_call_response contains the error message if the tool call failed
      await self.agent.remember(self.scope, self.conversation, tool_call_response)
      if self.stream:
        yield self.streaming_response.make_snippet(tool_call_response)
      else:
        self.whole_response.messages.append(tool_call_response)

    # after processing tool calls, we need another model call
    self.state = AgentState.MODEL_CALLING

  async def _handle_finished_state(self):
    """Handle the FINISHED state: End the conversation and return the final response."""

    if self.stream:
      yield self.streaming_response.make_finished_snippet()
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

    while self.state != AgentState.FINISHED:
      async for result in self.transition():
        yield result

    async for result in self._handle_finished_state():
      yield result


class Agent(InfoContext, DebugContext):
  _logger = None
  tools: Dict[str, InvokableTool]
  memory_knowledge: Optional[AgentMemoryKnowledge]

  @classmethod
  def class_logger(cls):
    if cls._logger:
      return cls._logger
    else:
      from ..logs.logs import get_logger

      cls._logger = get_logger("agent")
      return cls._logger

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
  ):
    self.logger = Agent.class_logger()
    with self.info(f"Starting agent '{name}'", f"Started agent '{name}'"):
      self.node = node

      self.tools = tools
      self.tool_specs = tool_specs

    self.name = name
    self.model = model
    self.memory_model = memory_model
    self.memory_embeddings_model = memory_embeddings_model
    self.memory_knowledge = None

    self.memory = memory
    memory.set_instructions(system_message(instructions))

    self.planner = planner
    self.maximum_iterations = maximum_iterations

    self.knowledge = knowledge

    self.converter = MessageConverter.create(node)

  async def handle_message(self, context, message):
    try:
      self.logger.debug(f"Agent '{self.name}' received: {message}")

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
          self.logger.error(f"Unexpected message: {message}")
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
    messages: list[dict] = self.memory.get_messages_only(scope, conversation)
    return [self.converter.conversation_message_from_dict(m) for m in messages]

  async def remember(self, scope: str, conversation: str, message: ConversationMessage):
    # TODO: at some point memory becomes bigger than context window and we need to start pushing memories
    # into Knowledge
    # self.history_knowledge.add(messages=[model_response], user_id=scope)
    if not isinstance(message, dict):
      message = self.converter.message_to_dict(message)
    self.memory.add_message(scope, conversation, message)

  async def message_history(self, scope: str, conversation: str) -> list[dict]:
    return self.memory.get_messages(scope, conversation)

  async def determine_input_context(self, scope: str, conversation: str):
    """Determine the input context for the agent based on memory and knowledge."""

    # Get the message history as ConversationMessage objects (not raw dicts)
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
        messages.insert(-1, context_message)  # Insert before the last message (usually user input)

      # Call the model to generate a response
      if stream:
        # Model now returns async generator directly for streaming
        chunk_count = 0
        has_content = False
        accumulated_content = ""
        last_finished = False

        self.logger.debug(f"Starting streaming model call with {len(messages)} messages")

        async for chunk in self.model.complete_chat(messages, stream=True, tools=self.tool_specs):
          chunk_count += 1
          self.logger.debug(f"Processing streaming chunk {chunk_count}")

          # Process streaming response chunks
          if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            finished = choice.finish_reason is not None if hasattr(choice, "finish_reason") else False
            last_finished = finished

            # Handle content chunks
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
              content = choice.delta.content
              self.logger.debug(f"Delta content: {repr(content)}, finished: {finished}")

              if content:  # Non-empty content
                accumulated_content += content
                has_content = True
                response = AssistantMessage(content=content, phase=Phase.EXECUTING)
                self.logger.debug(f"Yielding content chunk: finished={finished}")
                yield None, finished, response
              elif finished and (content == "" or content is None):
                # Empty content with finish_reason - this is a termination chunk
                self.logger.debug("Received empty finish chunk - yielding final response")
                # Must yield a final response to signal completion to state machine
                if accumulated_content:
                  # Send accumulated content as final response
                  response = AssistantMessage(content=accumulated_content, phase=Phase.EXECUTING)
                else:
                  # Send empty response if no content was accumulated
                  response = AssistantMessage(content="", phase=Phase.EXECUTING)
                yield None, True, response
                has_content = True
            # Handle tool call chunks
            elif hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
              tool_calls = []
              for tc in choice.delta.tool_calls:
                tool_call = ToolCall(
                  id=tc.id if hasattr(tc, "id") else f"call_{secrets.token_hex(8)}",
                  function=tc.function if hasattr(tc, "function") else None,
                  type="function",
                )
                tool_calls.append(tool_call)
              response = AssistantMessage(tool_calls=tool_calls, phase=Phase.EXECUTING)
              yield None, finished, response
            # Handle legacy format - message instead of delta (for fallback clients)
            elif finished and hasattr(choice, "message") and choice.message.content:
              response = AssistantMessage(content=choice.message.content, phase=Phase.EXECUTING)
              yield None, True, response
              has_content = True

            # If we get a finish_reason, we're done
            if finished:
              self.logger.debug("Breaking due to finish_reason")
              break

        # Safety check - ensure we have some response
        self.logger.debug(
          f"Streaming complete: chunks={chunk_count}, has_content={has_content}, last_finished={last_finished}"
        )

        if chunk_count == 0:
          # No chunks received at all
          self.logger.warning("No streaming chunks received")
          response = AssistantMessage(
            content="I apologize, but I encountered an issue generating a response.", phase=Phase.EXECUTING
          )
          yield None, True, response
        elif not has_content and not last_finished:
          # We got chunks but no actual content and no finish signal
          self.logger.warning(f"Streaming incomplete: {chunk_count} chunks but no content or finish signal")
          response = AssistantMessage(
            content="I apologize, but I encountered an issue generating a response.", phase=Phase.EXECUTING
          )
          yield None, True, response
      else:
        # Non-streaming response
        response = await self.model.complete_chat(messages, stream=False, tools=self.tool_specs)
        if hasattr(response, "choices") and len(response.choices) > 0:
          message = response.choices[0].message
          if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
              tool_call = ToolCall(
                id=tc.id if hasattr(tc, "id") else f"call_{secrets.token_hex(8)}",
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
      self.logger.error(f"Error in complete_chat: {e}")
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

  async def start(self):
    """Start the agent worker."""
    # The agent is now running and ready to handle messages
    self.logger.info(f"Agent '{self.name}' is ready")

  async def stop(self):
    """Stop the agent worker."""
    self.logger.info(f"Agent '{self.name}' stopped")

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
    exposed_as: Optional[str] = None,
    knowledge: Optional[KnowledgeProvider] = None,
    max_iterations: int = 14,
  ):
    """Start an agent on a node."""

    if name is None:
      name = secrets.token_hex(12)

    validate_name(name)

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
    exposed_as: Optional[str],
  ):
    """Create and start an agent implementation."""

    validate_name(name)

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
      )

    await node.start_spawner(name, agent_creator, key_extractor, None, exposed_as)

    Agent.class_logger().debug(f"Successfully started agent {name}")

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
