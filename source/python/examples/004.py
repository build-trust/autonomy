from autonomy import Model, Node


async def main(node):
  print(Model.list())


Node.start(main)
