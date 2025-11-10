from autonomy import Agent, Model, Node


async def main(node):
  await Agent.start(
      node=node,
      name="henry",
      instructions="""
        You are Henry, an expert legal assistant.

        IMPORTANT: When you need information from the user, you MUST use the
        ask_user_for_input tool. Do NOT ask questions in your regular responses.
        Ask ONE question at a time using the tool.
      """,
      model=Model("claude-sonnet-4-v1"),
      enable_ask_for_user_input=True,
  )

Node.start(main)
