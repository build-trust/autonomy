from autonomy import Node


class Echoer:
  async def handle_message(self, context, message):
    print(f"`echoer` received: {message}\n")
    await context.reply(message)


async def main(node):
  await node.start_worker("echoer", Echoer())
  reply = await node.send_and_receive("echoer", "hello")
  print(f"`main` received: {reply}\n")


Node.start(main)
