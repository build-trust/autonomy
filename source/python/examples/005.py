from autonomy import Model, Node, SystemMessage, UserMessage


async def main(node):
  model = Model("llama3.2")
  response = await model.complete_chat(
    [SystemMessage("You are a helpful assistant."), UserMessage("What is the capital of France?")]
  )
  print(f"Assistant: {response}\n")


Node.start(main)
