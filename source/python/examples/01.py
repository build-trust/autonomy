from autonomy import Node


async def main(node):
  pass


Node.start(main, wait_until_interrupted=False)

# OCKAM_SQLITE_IN_MEMORY=1 uv run --active examples/01.py
