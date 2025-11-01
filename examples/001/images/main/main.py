from autonomy import Agent, Model, Node, Repl


async def main(node):
  agent = await Agent.start(
    node=node,
    name="jack",
    instructions="You are Jack Sparrow",
    model=Model("llama3.2")
  )
  await Repl.start(agent)


Node.start(main)
