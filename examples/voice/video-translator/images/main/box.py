"""Integrate with Box API to handle video file operations and webhooks.

This module provides the Box client class to download and upload files
and to manage webhooks for automatic video processing.
"""

import asyncio
import tempfile

from os import environ, getenv
from pathlib import Path
from typing import Callable, Optional, Set

try:
    from box_sdk_gen import (
        BoxClient,
        BoxCCGAuth,
        CCGConfig,
        CreateWebhookTarget,
        CreateWebhookTargetTypeField,
        CreateWebhookTriggers,
    )

    BOX_SDK_AVAILABLE = True
except ImportError:
    BOX_SDK_AVAILABLE = False
    BoxClient = None
    BoxCCGAuth = None
    CCGConfig = None
    CreateWebhookTarget = None
    CreateWebhookTargetTypeField = None
    CreateWebhookTriggers = None


def _box_call(func, *args, **kwargs):
    """Run synchronous Box SDK calls in a thread pool."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


# =============================================================================
# Module-Level Client Management
# =============================================================================

_client: Optional["Box"] = None


def is_configured() -> bool:
    """Check if Box credentials are configured.

    Returns:
      True if all required Box environment variables are set
    """
    return all(
        [
            environ.get("BOX_CLIENT_ID"),
            environ.get("BOX_CLIENT_SECRET"),
            environ.get("BOX_ENTERPRISE_ID"),
        ]
    )


def get_client() -> Optional["Box"]:
    """Get or create the Box client singleton.

    Returns:
      Box client instance if configured, None otherwise
    """
    global _client
    if _client is None and is_configured():
        try:
            _client = Box()
        except Exception as e:
            print(f"Warning: Could not initialize Box client: {e}")
            return None
    return _client


async def initialize() -> bool:
    """Initialize Box client and set up webhook if configured.

    Reads BOX_FOLDER_ID and WEBHOOK_BASE_URL from environment to optionally
    configure automatic webhook setup.

    Returns:
      True if Box client was initialized successfully
    """
    global _client

    if not is_configured():
        print("Box credentials not configured - Box integration disabled")
        print("Demo UI upload is still available at /")
        return False

    client = get_client()
    if not client:
        return False

    print("Box client initialized")

    folder_id = getenv("BOX_FOLDER_ID")
    webhook_url = getenv("WEBHOOK_BASE_URL")

    if folder_id and webhook_url:
        try:
            await client.ensure_webhook(folder_id)
            print(f"Box webhook configured for folder: {folder_id}")
        except Exception as e:
            print(f"Warning: Could not set up Box webhook: {e}")

    return True


# =============================================================================
# Webhook Handling
# =============================================================================


def validate_webhook_payload(payload: dict, video_extensions: Set[str]) -> dict:
    """Validate a Box webhook payload and determine the action to take.

    Args:
      payload: The webhook payload from Box
      video_extensions: Set of valid video file extensions (e.g., {".mp4", ".mov"})

    Returns:
      Dictionary with action and relevant data:
      - {"action": "challenge", "challenge": str} for webhook validation
      - {"action": "ignore", "response": dict} for non-video or non-upload events
      - {"action": "process", "file_id": str, "filename": str} for videos to process
    """
    if "challenge" in payload:
        return {"action": "challenge", "challenge": payload["challenge"]}

    trigger = payload.get("trigger")
    source = payload.get("source", {})
    file_id = source.get("id")
    file_name = source.get("name", "")
    file_type = source.get("type")

    if trigger != "FILE.UPLOADED" or file_type != "file":
        return {
            "action": "ignore",
            "response": {"status": "ignored", "reason": "Not a file upload event"},
        }

    ext = Path(file_name).suffix.lower()
    if ext not in video_extensions:
        return {
            "action": "ignore",
            "response": {"status": "ignored", "reason": "Not a video file"},
        }

    return {"action": "process", "file_id": file_id, "filename": file_name}


async def process_video(
    file_id: str, filename: str, process_func: Callable, result_formatter: Callable
) -> None:
    """Process a video from Box in the background.

    Downloads the video, processes it using the provided function,
    and uploads results back to Box.

    Args:
      file_id: Box file ID
      filename: Original filename
      process_func: Async function(video_path, filename) -> result dict
      result_formatter: Function(result) -> markdown string
    """
    client = get_client()

    if not client:
        print(f"Box client not available, cannot process {filename}")
        return

    try:
        print(f"Processing video from Box: {filename} (ID: {file_id})")

        video_bytes = await client.download_file(file_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / filename
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            result = await process_func(str(video_path), filename)
            result_md = result_formatter(result)
            parent_folder_id = await client.get_file_parent_folder(file_id)
            result_filename = Path(filename).stem + "_transcript.md"

            await client.upload_file(
                folder_id=parent_folder_id,
                filename=result_filename,
                content=result_md.encode("utf-8"),
            )

            print(f"Uploaded translation result: {result_filename}")

    except Exception as e:
        print(f"Error processing Box video {filename}: {e}")


# =============================================================================
# Box Client Class
# =============================================================================


class Box:
    """Interact with Box API to handle video file operations and webhooks."""

    def __init__(self):
        if not BOX_SDK_AVAILABLE:
            raise ImportError(
                "box-sdk-gen is not installed. Install it with: pip install box-sdk-gen"
            )

        self.client = BoxClient(
            auth=BoxCCGAuth(
                config=CCGConfig(
                    client_id=environ["BOX_CLIENT_ID"],
                    client_secret=environ["BOX_CLIENT_SECRET"],
                    enterprise_id=environ["BOX_ENTERPRISE_ID"],
                )
            )
        )

    async def download_file(self, file_id: str) -> bytes:
        """Download a file from Box and return its content as bytes."""
        content = await _box_call(self.client.downloads.download_file, file_id)
        return content.read()

    async def upload_file(self, folder_id: str, filename: str, content: bytes) -> str:
        """Upload a file to Box.

        Args:
          folder_id: The Box folder ID to upload to
          filename: Name for the uploaded file
          content: File content as bytes

        Returns:
          The file ID of the uploaded file
        """
        import io

        file_stream = io.BytesIO(content)

        try:
            uploaded = await _box_call(
                self.client.uploads.upload_file,
                attributes={"name": filename, "parent": {"id": folder_id}},
                file=file_stream,
            )
            return uploaded.entries[0].id
        except Exception as e:
            if "item_name_in_use" in str(e).lower():
                existing_file_id = await self._find_file_in_folder(folder_id, filename)
                if existing_file_id:
                    file_stream.seek(0)
                    uploaded = await _box_call(
                        self.client.uploads.upload_file_version,
                        file_id=existing_file_id,
                        attributes={},
                        file=file_stream,
                    )
                    return uploaded.entries[0].id
            raise

    async def _find_file_in_folder(
        self, folder_id: str, filename: str
    ) -> Optional[str]:
        """Find a file by name in a folder.

        Args:
          folder_id: The Box folder ID to search
          filename: Name of the file to find

        Returns:
          The file ID if found, None otherwise
        """
        items = await _box_call(self.client.folders.get_folder_items, folder_id)

        for item in items.entries:
            if item.type == "file" and item.name == filename:
                return item.id

        return None

    async def get_file_info(self, file_id: str):
        """Retrieve file metadata from Box.

        Args:
          file_id: The Box file ID

        Returns:
          File metadata object
        """
        return await _box_call(self.client.files.get_file_by_id, file_id)

    async def get_file_parent_folder(self, file_id: str) -> str:
        """Get the parent folder ID of a file.

        Args:
          file_id: The Box file ID

        Returns:
          The parent folder ID
        """
        file_info = await self.get_file_info(file_id)
        return file_info.parent.id

    async def list_folder_items(self, folder_id: str):
        """List items in a folder.

        Args:
          folder_id: The Box folder ID

        Returns:
          List of folder item entries
        """
        items = await _box_call(self.client.folders.get_folder_items, folder_id)
        return items.entries

    async def list_folder_by_path(self, path: str):
        """Navigate to a folder by path and list its items.

        Args:
          path: Folder path (e.g., "/videos/incoming")

        Returns:
          List of folder item entries

        Raises:
          ValueError: If folder not found in path
        """
        folder_id = "0"  # Root folder

        if path and path != "/":
            for folder_name in path.strip("/").split("/"):
                items = await self.list_folder_items(folder_id)
                found = False
                for item in items:
                    if item.type == "folder" and item.name == folder_name:
                        folder_id = item.id
                        found = True
                        break
                if not found:
                    raise ValueError(f"Folder not found: {folder_name}")

        return await self.list_folder_items(folder_id)

    # ===========================================================================
    # Manage Webhooks
    # ===========================================================================

    async def list_webhooks(self):
        """List all webhooks for this application.

        Returns:
          List of webhook entries
        """
        webhooks = await _box_call(self.client.webhooks.get_webhooks)
        return webhooks.entries

    async def create_webhook(
        self, folder_id: str, address: str, triggers: list = None
    ) -> str:
        """Create a webhook for a folder.

        Args:
          folder_id: The Box folder ID to watch
          address: The URL to receive webhook notifications
          triggers: List of triggers (default: FILE.UPLOADED)

        Returns:
          The webhook ID
        """
        if triggers is None:
            triggers = [CreateWebhookTriggers.FILE_UPLOADED]

        webhook = await _box_call(
            self.client.webhooks.create_webhook,
            target=CreateWebhookTarget(
                id=folder_id, type=CreateWebhookTargetTypeField.FOLDER
            ),
            address=address,
            triggers=triggers,
        )

        return webhook.id

    async def delete_webhook(self, webhook_id: str):
        """Delete a webhook.

        Args:
          webhook_id: The webhook ID to delete
        """
        await _box_call(self.client.webhooks.delete_webhook_by_id, webhook_id)

    async def ensure_webhook(self, folder_id: str, webhook_base_url: str = None) -> str:
        """Ensure a webhook exists for the given folder.

        If a webhook already exists for this folder, returns its ID.
        Otherwise creates a new webhook.

        Args:
          folder_id: The Box folder ID to watch
          webhook_base_url: Base URL for the webhook (uses environment if not provided)

        Returns:
          The webhook ID

        Raises:
          ValueError: If webhook URL not configured
        """
        if webhook_base_url is None:
            webhook_base_url = environ.get("WEBHOOK_BASE_URL", "")

        if not webhook_base_url:
            raise ValueError(
                "Webhook URL not configured. Set WEBHOOK_BASE_URL environment variable "
                "or pass webhook_base_url parameter."
            )

        webhook_address = f"{webhook_base_url.rstrip('/')}/webhook/box"
        existing = await self.list_webhooks()
        for webhook in existing:
            if (
                webhook.target
                and webhook.target.id == folder_id
                and webhook.target.type == "folder"
            ):
                print(f"Webhook already exists for folder {folder_id}: {webhook.id}")
                return webhook.id

        webhook_id = await self.create_webhook(
            folder_id=folder_id,
            address=webhook_address,
            triggers=[CreateWebhookTriggers.FILE_UPLOADED],
        )

        print(f"Created webhook for folder {folder_id}: {webhook_id}")
        return webhook_id
