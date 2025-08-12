from autonomy import Node, info


class Echoer:
  async def handle_message(self, context, message):
    info(f"Echoer received: {message}")
    await context.reply(message)


async def main(node):
  await node.start_worker("echoer", Echoer())


Node.start(main)

# Create a zone called `hello`, then:
#
# export NODE=server
# export CLUSTER="$(ockam cluster show)"
# export ENROLLMENT_TICKET="$(ockam zone ticket hello --relay $NODE)"
# OCKAM_SQLITE_IN_MEMORY=1 uv run --active examples/03-server.py
