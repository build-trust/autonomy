from autonomy import Node, info


class Echoer:
  async def handle_message(self, context, message):
    info(f"Echoer received: {message}")
    await context.reply(message)


async def main(node):
  await node.start_worker("echoer", Echoer())
  reply = await node.send_and_receive("echoer", "hello")
  info(f"Reply received: {reply}")

  sender = await node.create_mailbox("sender")
  receiver = await node.create_mailbox("receiver")

  await sender.send("receiver", "test")

  msg = await receiver.receive(timeout=1)
  info(f"Received: {msg}")


Node.start(main, wait_until_interrupted=False)

# OCKAM_SQLITE_IN_MEMORY=1 uv run --active examples/02.py
