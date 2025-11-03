from autonomy import Agent, Node, Repl, Model


async def main(node):
  agent = await Agent.start(
    node=node,
    model=Model(name="llama3.2"),
    instructions="""
      You are Henry, an expert legal assistant.
      You have in-depth knowledge of United States corporate law.
    """,
  )

  await Repl.start(agent, stream=True)


Node.start(main)

# AUTONOMY_WAIT_UNTIL_INTERRUPTED=1 AUTONOMY_USE_IN_MEMORY_DATABASE=1 uv run --active examples/008.py
# nc 127.0.0.1 7000
