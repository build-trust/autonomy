from autonomy import Node, Agent, Repl


async def main(node):
    agent = await Agent.start(node, "You are Jack Sparrow", "jack")
    await Repl.start(agent)


Node.start(main)
