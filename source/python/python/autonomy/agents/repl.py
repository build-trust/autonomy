import asyncio
import secrets
import re
import os
from typing import Set
from contextlib import asynccontextmanager

from ..helpers.parse_socket_address import parse_socket_address
from ..logs import get_logger

DEFAULT_HOST = os.environ.get("DEFAULT_HOST_REPL", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("DEFAULT_PORT_REPL", "7000"))

logger = get_logger("repl")


class Repl:
  def __init__(
    self, agent_reference, listen_address=f"{DEFAULT_HOST}:{DEFAULT_PORT}", functions=None, timeout=120, stream=True
  ):
    logger.info("Starting the repl")
    self.functions = functions or {}
    self.host = DEFAULT_HOST
    self.port = DEFAULT_PORT
    self.set_host_and_port(listen_address)
    self.agent_reference = agent_reference
    self.server = None
    self.scope = None
    self.conversation = None
    self.timeout = timeout
    self.stream = stream

    # Production improvements: Connection tracking for graceful shutdown
    self.active_connections: Set[asyncio.StreamWriter] = set()
    self._shutdown_in_progress = False

    self.short_help_text = "Type ':help' or ':h' to see this message again.\nType ':quit' or ':q' to disconnect."

    self.help_text = (
      "Type ':help' or ':h' to see this message again.\n"
      "Type ':scope' or ':s' to set the scope for future interactions.\n"
      "Type ':conversation' or ':c' to set the conversation for future interactions.\n"
      "Type ':functions' to see the list of available functions.\n"
      "Type ':run' to run a predefined function.\n"
      "Type ':quit' or ':q' to disconnect."
    )

  # TODO: should be on Reference class
  @staticmethod
  async def start(
    agent_reference, listen_address=f"{DEFAULT_HOST}:{DEFAULT_PORT}", functions=None, timeout=None, stream=True
  ):
    repl = Repl(agent_reference, listen_address, functions, timeout, stream)
    # start the repl in a separate thread since we might also have a HTTP server running
    asyncio.create_task(repl.start_impl())

  def set_host_and_port(self, listen_address):
    try:
      host, port = parse_socket_address(listen_address)
      self.host = host
      self.port = port
    except ValueError:
      raise ValueError(f"Invalid listen_address: {listen_address}")

  async def list_functions(self, writer):
    logger.debug("List functions")
    functions_list = "\n".join([f"{name}" for name in self.functions.keys()])
    await self.write_repl_message(writer, functions_list)

  async def update_scope(self, message, writer):
    try:
      _, scope = message.split(" ", 1)
      self.scope = scope
      await self.write_repl_message(writer, f"Scope set to {self.scope}")
    except ValueError:
      await self.write_repl_message(writer, "Invalid scope format. Use ':scope <scope>'.")

  async def update_conversation(self, message, writer):
    try:
      _, conversation = message.split(" ", 1)
      self.conversation = conversation
      await self.write_repl_message(writer, f"Conversation set to {self.conversation}")
    except ValueError:
      await self.write_repl_message(writer, "Invalid conversation format. Use ':conversation <conversation>'.")

  async def write_repl_message(self, writer, message):
    """
    Write a REPL instruction with newlines at the end to leave some space for the prompt
    """
    message = (message + "\n\n").encode()
    writer.write(f"{len(message)}\n".encode())
    writer.write(message)
    await writer.drain()

  async def write(self, writer, message):
    """
    Write a message to the output with a length encoding
    """

    for word in re.findall(r"\S+|\s+", message):
      word = word.encode()
      writer.write(f"{len(word)}\n".encode())
      writer.write(word)
      await writer.drain()
      await asyncio.sleep(0.03)

  async def _safe_wait_closed(self, writer: asyncio.StreamWriter) -> None:
    """Safely wait for a writer to close with error handling."""
    try:
      await writer.wait_closed()
    except Exception as e:
      # Log but don't fail - connection cleanup should be best-effort
      logger.debug(f"Connection cleanup error: {e}")

  async def _handle_repl_command(self, message: str, writer: asyncio.StreamWriter) -> None:
    """Handle REPL commands with improved error handling."""
    try:
      if message.lower() in (":help", ":h"):
        await self.write_repl_message(writer, self.help_text)

      elif message.lower().split()[0] in (":scope", ":s"):
        await self.update_scope(message, writer)

      elif message.lower().split()[0] in (":conversation", ":c"):
        await self.update_conversation(message, writer)

      elif message.lower() == ":functions":
        await self.list_functions(writer)

      elif message.lower().split()[0] == ":run":
        await self._handle_function_run(message, writer)

      else:
        await self.write_repl_message(writer, f"Unknown command: {message}")

    except Exception as e:
      logger.error(f"Error handling REPL command '{message}': {e}")
      await self.write_repl_message(writer, f"Command error: {str(e)}")

  async def _handle_function_run(self, message: str, writer: asyncio.StreamWriter) -> None:
    """Handle :run command with better error handling."""
    parts = message.split(" ")
    if len(parts) != 2:
      await self.write_repl_message(writer, "Usage: :run <function_name>")
      return

    function_name = parts[1]
    if function_name not in self.functions:
      await self.write_repl_message(writer, "Function not found.")
      return

    try:
      result = await self.functions[function_name]()
      await self.write_repl_message(writer, str(result))
    except Exception as e:
      logger.error(f"Error executing function '{function_name}': {e}")
      await self.write_repl_message(writer, f"Function execution error: {str(e)}")

  async def _handle_agent_message(self, message: str, writer: asyncio.StreamWriter) -> None:
    """Handle messages to the agent with improved error handling."""
    try:
      if self.stream:
        await self._handle_streaming_response(message, writer)
      else:
        await self._handle_non_streaming_response(message, writer)
    except Exception as e:
      logger.error(f"Error processing agent message: {e}")
      await self.write_repl_message(writer, f"Agent communication error: {str(e)}")

  async def _handle_streaming_response(self, message: str, writer: asyncio.StreamWriter) -> None:
    """Handle streaming agent responses with timeout protection."""
    buffer = ""
    response_timeout = self.timeout or 120.0

    try:
      stream = self.agent_reference.send_stream(
        message, scope=self.scope, conversation=self.conversation, timeout=self.timeout
      )

      async for response in stream:
        # Handle StreamedConversationSnippet
        if hasattr(response, "snippet") and hasattr(response, "finished"):
          if len(response.snippet.messages) > 0:
            content = response.snippet.messages[0].content
            if hasattr(content, "text"):
              received = content.text
            else:
              received = str(content)

            buffer += received
            if len(buffer) > 20:
              await self.write(writer, buffer)
              buffer = ""

          if response.finished:
            if buffer:
              await self.write(writer, buffer)
            writer.write(b"0\n")
            await writer.drain()
            break
        else:
          # Handle regular conversation response
          content = response.messages[0].content
          if hasattr(content, "text"):
            received = content.text
          else:
            received = str(content)

          await self.write(writer, received)
          writer.write(b"0\n")
          await writer.drain()
          break

    except asyncio.TimeoutError:
      await self.write_repl_message(writer, f"Response timed out after {response_timeout}s")
    except AttributeError as e:
      # Agent doesn't support streaming
      logger.warning(f"Agent doesn't support streaming: {e}")
      await self._handle_non_streaming_response(message, writer)

  async def _handle_non_streaming_response(self, message: str, writer: asyncio.StreamWriter) -> None:
    """Handle non-streaming agent responses with timeout protection."""
    response_timeout = self.timeout or 120.0

    try:
      reply = await asyncio.wait_for(
        self.agent_reference.send(message, scope=self.scope, conversation=self.conversation, timeout=self.timeout),
        timeout=response_timeout,
      )

      first_message = reply[0]
      if hasattr(first_message.content, "text"):
        content = first_message.content.text
      else:
        content = str(first_message.content)

      await self.write(writer, content)
      writer.write(b"0\n")
      await writer.drain()

    except asyncio.TimeoutError:
      await self.write_repl_message(writer, f"Response timed out after {response_timeout}s")

  async def handle_client(self, reader, writer):
    """
    Improved client handler with better error handling and connection tracking.
    """
    client_address = writer.get_extra_info("peername")
    logger.debug(f"New client connected from {client_address}")

    # Track this connection
    self.active_connections.add(writer)

    # Generate scope and conversation for this client
    self.scope = secrets.token_hex(16)
    self.conversation = secrets.token_hex(12)

    try:
      # Send welcome message
      welcome_message = (
        "\nWelcome to Autonomy 👋\n\n"
        f"You are connected to your agent - {getattr(self.agent_reference.node, 'name', 'node')}/{self.agent_reference.name}\n"
        "Ask it a question or give it a task.\n\n"
      )
      await self.write_repl_message(writer, welcome_message + self.short_help_text)

      # Main message handling loop
      while not self._shutdown_in_progress:
        try:
          # Read message with timeout to prevent hanging
          data = await asyncio.wait_for(reader.readline(), timeout=300.0)  # 5 minute timeout
          if not data:
            logger.debug(f"Client {client_address} disconnected (no data)")
            break

          # Handle multiline support
          separator = b'"""\n'
          if data == separator:
            data = await reader.readuntil(separator)
            data = data[: -len(separator)]

          message = data.decode().strip()
          if not message:
            continue

          # Handle quit commands
          if message.lower() in (":quit", ":q"):
            logger.debug(f"Client {client_address} requested disconnect")
            break

          # Handle REPL commands
          if message.startswith(":"):
            await self._handle_repl_command(message, writer)
            continue

          # Handle regular messages to agent
          await self._handle_agent_message(message, writer)

        except asyncio.TimeoutError:
          logger.warning(f"Client {client_address} timed out")
          break
        except UnicodeDecodeError as e:
          logger.warning(f"Invalid message encoding from {client_address}: {e}")
          await self.write_repl_message(writer, "Error: Invalid message encoding. Please use UTF-8.")
        except ConnectionResetError:
          logger.debug(f"Client {client_address} reset connection")
          break
        except Exception as e:
          logger.error(f"Error handling message from {client_address}: {e}")
          # Send error message to client but continue serving
          try:
            await self.write_repl_message(writer, f"Error: {str(e)}")
          except:
            break  # If we can't send error message, connection is broken

    except Exception as e:
      logger.error(f"Fatal error in client handler for {client_address}: {e}")
    finally:
      # Clean up connection
      self.active_connections.discard(writer)
      if not writer.is_closing():
        try:
          writer.close()
          await self._safe_wait_closed(writer)
        except Exception as e:
          logger.debug(f"Error closing connection to {client_address}: {e}")

      logger.debug(f"Client {client_address} handler finished")

  async def start_impl(self):
    self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
    logger.info("Started the repl")
    async with self.server:
      await self.server.serve_forever()

  async def stop(self, timeout: float = 5.0):
    """
    Improved stop method with timeout protection and connection cleanup.
    """
    if self._shutdown_in_progress:
      logger.warning("Shutdown already in progress")
      return

    self._shutdown_in_progress = True
    logger.info(f"Stopping REPL server (timeout: {timeout}s)")

    try:
      if self.server:
        # Step 1: Stop accepting new connections
        self.server.close()
        logger.debug("Server closed to new connections")

        # Step 2: Close active connections gracefully
        if self.active_connections:
          logger.info(f"Closing {len(self.active_connections)} active connections")
          close_tasks = []

          for writer in list(self.active_connections):
            if not writer.is_closing():
              try:
                # Send a polite disconnect message if possible
                writer.write(b"0\n")  # End-of-stream marker
                await writer.drain()
              except:
                pass  # Ignore errors when sending disconnect

              writer.close()
              close_tasks.append(self._safe_wait_closed(writer))

          # Wait for connections to close with timeout
          if close_tasks:
            try:
              await asyncio.wait_for(
                asyncio.gather(*close_tasks, return_exceptions=True),
                timeout=timeout / 2,  # Use half timeout for connections
              )
            except asyncio.TimeoutError:
              logger.warning("Some connections did not close gracefully")

        # Step 3: Wait for server shutdown with remaining timeout
        try:
          remaining_timeout = max(timeout / 2, 1.0)
          await asyncio.wait_for(self.server.wait_closed(), timeout=remaining_timeout)
          logger.info("REPL server stopped successfully")
        except asyncio.TimeoutError:
          logger.warning(f"Server shutdown timed out after {timeout}s, forcing close")
          # Server is effectively closed anyway, continue

        self.server = None

    except Exception as e:
      logger.error(f"Error during server shutdown: {e}")
      # Don't re-raise - we want shutdown to always complete
    finally:
      self.active_connections.clear()
      self._shutdown_in_progress = False

  @asynccontextmanager
  async def managed_lifecycle(self, startup_timeout: float = 10.0, shutdown_timeout: float = 5.0):
    """
    Context manager for REPL lifecycle management.

    This can be used in production applications that need guaranteed cleanup.

    Example:
        async with repl.managed_lifecycle() as repl_instance:
            # REPL is running
            await some_application_logic()
            # REPL will be cleanly shut down
    """
    server_task = None
    try:
      # Start server
      server_task = asyncio.create_task(self.start_impl())

      # Wait for server to be ready
      start_time = asyncio.get_event_loop().time()
      while not self.server and (asyncio.get_event_loop().time() - start_time) < startup_timeout:
        await asyncio.sleep(0.1)

      if not self.server:
        raise RuntimeError(f"REPL server failed to start within {startup_timeout}s")

      logger.info("REPL server started successfully")
      yield self

    except Exception as e:
      logger.error(f"Error in REPL lifecycle: {e}")
      raise
    finally:
      # Clean shutdown
      if server_task and not server_task.done():
        server_task.cancel()
        try:
          await asyncio.wait_for(server_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
          pass

      await self.stop(timeout=shutdown_timeout)
