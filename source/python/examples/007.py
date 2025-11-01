from autonomy import Agent, Model, Node, info


async def main(node):
  agent = await Agent.start(
    node=node,
    name="history-teacher",
    instructions="You are an assistant who is an expert in history.",
    model=Model(name="llama3.2"),
  )
  identifier = await agent.identifier()
  info(f"the agent identifier is {identifier}")

  reply = await agent.send("Who was Gandhi?")
  info(reply)


Node.start(main)
