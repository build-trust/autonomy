from ..nodes.node import _pick_cluster_name
from ..nodes.remote import list_nodes


class Cluster:
  @staticmethod
  async def nodes(node, filter=None):
    cluster = _pick_cluster_name()
    return await list_nodes(node, cluster, filter=filter)
