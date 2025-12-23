import asyncio
from os import environ
from typing import Optional

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


class Box:
  """Box API client for video file operations and webhooks."""

  def __init__(self):
    if not BOX_SDK_AVAILABLE:
      raise ImportError("box-sdk-gen is not installed. Install it with: pip install box-sdk-gen")

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
    content = await _box_call(
      self.client.downloads.download_file, file_id
    )
    return content.read()

  async def upload_file(
    self,
    folder_id: str,
    filename: str,
    content: bytes
  ) -> str:
    """
    Upload a file to Box.

    Returns the file ID of the uploaded file.
    """
    import io

    file_stream = io.BytesIO(content)

    # Try to upload, if file exists, upload new version
    try:
      uploaded = await _box_call(
        self.client.uploads.upload_file,
        attributes={
          "name": filename,
          "parent": {"id": folder_id}
        },
        file=file_stream
      )
      return uploaded.entries[0].id
    except Exception as e:
      # If file already exists, find it and upload new version
      if "item_name_in_use" in str(e).lower():
        existing_file_id = await self._find_file_in_folder(folder_id, filename)
        if existing_file_id:
          file_stream.seek(0)
          uploaded = await _box_call(
            self.client.uploads.upload_file_version,
            file_id=existing_file_id,
            attributes={},
            file=file_stream
          )
          return uploaded.entries[0].id
      raise

  async def _find_file_in_folder(self, folder_id: str, filename: str) -> Optional[str]:
    """Find a file by name in a folder."""
    items = await _box_call(
      self.client.folders.get_folder_items, folder_id
    )

    for item in items.entries:
      if item.type == "file" and item.name == filename:
        return item.id

    return None

  async def get_file_info(self, file_id: str):
    """Get file metadata."""
    return await _box_call(
      self.client.files.get_file_by_id, file_id
    )

  async def get_file_parent_folder(self, file_id: str) -> str:
    """Get the parent folder ID of a file."""
    file_info = await self.get_file_info(file_id)
    return file_info.parent.id

  async def list_folder_items(self, folder_id: str):
    """List items in a folder."""
    items = await _box_call(
      self.client.folders.get_folder_items, folder_id
    )
    return items.entries

  async def list_folder_by_path(self, path: str):
    """Navigate to a folder by path and list its items."""
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
  # Webhook Management
  # ===========================================================================

  async def list_webhooks(self):
    """List all webhooks for this application."""
    webhooks = await _box_call(self.client.webhooks.get_webhooks)
    return webhooks.entries

  async def create_webhook(
    self,
    folder_id: str,
    address: str,
    triggers: list = None
  ) -> str:
    """
    Create a webhook for a folder.

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
        id=folder_id,
        type=CreateWebhookTargetTypeField.FOLDER
      ),
      address=address,
      triggers=triggers
    )

    return webhook.id

  async def delete_webhook(self, webhook_id: str):
    """Delete a webhook."""
    await _box_call(
      self.client.webhooks.delete_webhook_by_id, webhook_id
    )

  async def ensure_webhook(
    self,
    folder_id: str,
    webhook_base_url: str = None
  ) -> str:
    """
    Ensure a webhook exists for the given folder.

    If a webhook already exists for this folder, returns its ID.
    Otherwise creates a new webhook.

    Args:
      folder_id: The Box folder ID to watch
      webhook_base_url: Base URL for the webhook (uses environment if not provided)

    Returns:
      The webhook ID
    """
    # Determine webhook URL
    if webhook_base_url is None:
      # Try to get from environment or use a placeholder
      webhook_base_url = environ.get("WEBHOOK_BASE_URL", "")

    if not webhook_base_url:
      raise ValueError(
        "Webhook URL not configured. Set WEBHOOK_BASE_URL environment variable "
        "or pass webhook_base_url parameter."
      )

    webhook_address = f"{webhook_base_url.rstrip('/')}/webhook/box"

    # Check existing webhooks
    existing = await self.list_webhooks()
    for webhook in existing:
      if (
        webhook.target
        and webhook.target.id == folder_id
        and webhook.target.type == "folder"
      ):
        print(f"Webhook already exists for folder {folder_id}: {webhook.id}")
        return webhook.id

    # Create new webhook
    webhook_id = await self.create_webhook(
      folder_id=folder_id,
      address=webhook_address,
      triggers=[CreateWebhookTriggers.FILE_UPLOADED]
    )

    print(f"Created webhook for folder {folder_id}: {webhook_id}")
    return webhook_id
