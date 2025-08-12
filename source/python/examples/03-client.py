from autonomy import Node, RemoteNode, info
from sys import argv


async def main(node):
  remote_node = RemoteNode(node, argv[1])
  for i in range(5):
    reply = await remote_node.send_and_receive("echoer", "hello")
    info(f"{i}> {reply}")


Node.start(main, wait_until_interrupted=False)

# Create a zone called `hello`, then:
#
# export NODE=client
# export CLUSTER="$(ockam cluster show)"
# export ENROLLMENT_TICKET="$(ockam zone ticket hello --relay $NODE)"
# OCKAM_SQLITE_IN_MEMORY=1 uv run --active examples/03-client.py "$CLUSTER-hello-server"
