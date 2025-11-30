from autonomy import Agent, HttpServer, Model, Node


async def main(node):
  await Agent.start(
    node=node,
    name="henry",
    instructions="You are Henry, a helpful and friendly assistant.",
    model=Model("claude-sonnet-4-v1")
  )


Node.start(main, http_server=HttpServer(listen_address="0.0.0.0:9000"))
