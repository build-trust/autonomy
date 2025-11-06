"""
Tool factory protocol for scope-aware tool creation.

Enables tools to be created dynamically based on agent_name, scope, and conversation
context, allowing the spawner to create properly isolated tool instances for each worker.

This is essential for multi-tenant scenarios where tools need access to
scope-specific resources with configurable isolation levels (e.g., FilesystemTools
with visibility levels: all, agent, scope, or conversation).

Usage:
  # FilesystemTools implements ToolFactory protocol
  # Default visibility="conversation" provides maximum isolation
  fs_tools = FilesystemTools()  # Factory mode

  agent = await Agent.start(
    node=node,
    name="assistant",
    tools=[fs_tools],  # Just pass to tools parameter
  )

  # Custom factory example
  class MyToolFactory(ToolFactory):
    def create_tools(self, scope, conversation, agent_name=None):
      # Create tools with proper isolation
      fs = FilesystemTools(
        visibility="conversation",  # Default: per-conversation isolation
        agent_name=agent_name or "default",
        scope=scope or "default",
        conversation=conversation or "default"
      )
      return [
        Tool(fs.ls),
        Tool(fs.read_file),
        Tool(fs.write_file),
      ]

Architecture:
  Without factories (static tools):
    Agent.start -> creates tools -> all workers share same tool instances
    Example: Tool(get_time) - same instance for all users/conversations

  With factories (scope-aware tools):
    Agent.start -> registers factories -> spawner calls factory.create_tools() per worker
    Example: FilesystemTools() - creates isolated instance per worker

  Factory receives context:
    - agent_name: "assistant"
    - scope: "user-alice"
    - conversation: "chat-1"

  Factory creates tools with isolation:
    - visibility="all": /data/
    - visibility="agent": /data/assistant/
    - visibility="scope": /data/assistant/user-alice/
    - visibility="conversation": /data/assistant/user-alice/chat-1/ (default)

For practical guides:
- Tool usage: docs/_for-coding-agents/tools.mdx
- Filesystem tools: docs/_for-coding-agents/filesystem-tools.mdx
"""

from typing import List, Optional, Protocol
from .protocol import InvokableTool


class ToolFactory(Protocol):
  """
  Protocol for creating scope-aware tools with configurable isolation.

  Implement this protocol to create tools that need to be instantiated
  per worker rather than shared across all workers. This enables:
  - Multi-tenant isolation (different users get separate instances)
  - Session-specific resources (different conversations get separate instances)
  - Configurable visibility levels (all/agent/scope/conversation)

  The spawner will call create_tools() when creating a new worker,
  passing the agent_name, scope, and conversation extracted from the message.

  Note: FilesystemTools already implements this protocol when used in factory mode.
  You typically don't need to implement this yourself unless creating custom tools.

  Example:
    class DatabaseToolFactory:
      def __init__(self, connection_template):
        self.connection_template = connection_template

      def create_tools(self, scope, conversation, agent_name=None):
        from autonomy import Tool

        # Create tenant-specific database connection
        tenant_id = scope or "default"
        db = Database(tenant=tenant_id)

        return [
          Tool(db.query),
          Tool(db.insert),
          Tool(db.update),
        ]

    # FilesystemTools example (built-in factory)
    fs = FilesystemTools(visibility="conversation")  # Default
    # Creates isolated filesystem per conversation:
    # /tmp/agent-files/{agent_name}/{scope}/{conversation}/
  """

  def create_tools(
    self,
    scope: Optional[str],
    conversation: Optional[str],
    agent_name: Optional[str] = None,
  ) -> List[InvokableTool]:
    """
    Create tools for a specific agent, scope, and conversation context.

    Called by the spawner when creating a new worker. Should return
    a list of InvokableTool instances configured for the given context.

    The context parameters enable isolation at different levels:
    - agent_name only: Isolate per agent
    - agent_name + scope: Isolate per user/tenant
    - agent_name + scope + conversation: Isolate per session (default pattern)

    Args:
      scope: Scope identifier (e.g., "user-alice", "tenant-123")
             None if no scope was provided in the message
             Defaults to "default" if None
      conversation: Conversation identifier (e.g., "chat-1", "session-abc")
                   None if no conversation was provided in the message
                   Defaults to "default" if None
      agent_name: Agent name (e.g., "assistant", "support-bot")
                  Provided by the agent framework
                  Defaults to "default" if None

    Returns:
      List of InvokableTool instances ready for use by the agent

    Examples:
      # FilesystemTools with conversation-level isolation (default)
      def create_tools(self, scope, conversation, agent_name=None):
        fs = FilesystemTools(
          visibility="conversation",  # Default
          agent_name=agent_name or "default",
          scope=scope or "default",
          conversation=conversation or "default"
        )
        # Creates: /tmp/agent-files/{agent_name}/{scope}/{conversation}/
        return [Tool(fs.ls), Tool(fs.read_file), Tool(fs.write_file)]

      # FilesystemTools with scope-level isolation (shared across conversations)
      def create_tools(self, scope, conversation, agent_name=None):
        fs = FilesystemTools(
          visibility="scope",
          agent_name=agent_name or "default",
          scope=scope or "default"
        )
        # Creates: /tmp/agent-files/{agent_name}/{scope}/
        return [Tool(fs.ls), Tool(fs.read_file), Tool(fs.write_file)]

      # Custom database tools with tenant isolation
      def create_tools(self, scope, conversation, agent_name=None):
        db = Database(tenant=scope or "default")
        return [Tool(db.query), Tool(db.insert), Tool(db.update)]
    """
    ...


__all__ = ["ToolFactory"]
