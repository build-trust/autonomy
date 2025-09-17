from ..nodes.node import _pick_cluster_name, _pick_zone_name
from ..nodes.remote import list_nodes


class Zone:
  @staticmethod
  async def nodes(node, filter=None):
    cluster = _pick_cluster_name()
    zone = _pick_zone_name()
    return await list_nodes(node, f"{cluster}-{zone}", filter=filter)

  @staticmethod
  async def node(node, name):
    """Find a specific node by short name within the zone.

    Automatically constructs the full node name using the convention:
    {cluster}-{zone}-{name}

    Args:
        node: The local node to search from
        name: The short name of the node (e.g., "server", "worker1")

    Returns:
        RemoteNode if found, None otherwise
    """
    cluster = _pick_cluster_name()
    zone = _pick_zone_name()
    full_name = f"{cluster}-{zone}-{name}"

    zone_nodes = await Zone.nodes(node)
    for remote_node in zone_nodes:
      if remote_node.name == full_name:
        return remote_node
    return None
