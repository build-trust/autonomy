from autonomy import Model, Node, SystemMessage, UserMessage


async def main(node):
  model = Model("llama3.2")

  streaming_response = model.complete_chat(
    [SystemMessage("You are a helpful assistant."), UserMessage("Explain gravity")], stream=True
  )

  async for chunk in streaming_response:
    if hasattr(chunk, "choices") and chunk.choices and chunk.choices[0].delta.content:
      content = chunk.choices[0].delta.content
      print(content, end="")


Node.start(main)
