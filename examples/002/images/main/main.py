from autonomy import Agent, Model, Node


async def main(node):
  await Agent.start(
    node=node,
    name="henry",
    instructions="You are Henry, an expert legal assistant",
    model=Model("claude-sonnet-4-v1")
  )


Node.start(main)
