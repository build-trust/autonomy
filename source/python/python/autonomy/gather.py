import asyncio


async def gather(*coros_or_futures, batch_size=None):
  """
  Like asyncio.gather but with optional batch_size to limit concurrency.

  Args:
    *coros_or_futures: Coroutines or futures to execute
    batch_size: Optional maximum number of concurrent tasks. If None, all tasks run concurrently.

  Returns:
    List of results in the same order as the input coroutines/futures.
  """
  if not batch_size:
    return await asyncio.gather(*coros_or_futures)

  sem = asyncio.Semaphore(batch_size)

  async def batch_task(f):
    async with sem:
      return await f

  futures = [batch_task(f) for f in coros_or_futures]

  return await asyncio.gather(*futures)
