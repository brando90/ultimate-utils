"""
File demonstrating looping through asynchronous iterators.

Definitions:
async for = allows to loop through an asynchronous iterator.

plan:
- my guess is a good plan is to go through the generator & yield example in real python: https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines

Ref:
    - https://stackoverflow.com/questions/56161595/how-to-use-async-for-in-python
    - https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines
    - (idk, don't like this one too much... https://stackoverflow.com/questions/67092070/why-do-we-need-async-for-and-async-with

todo: create a concrete async forn example, likely contrasting it with a blocking for and how to "implement"
    an equivalent for with no async for but instead with an for with awaits.
"""

import time

import asyncio


async def process_all():
    """
    Example where the async for loop allows to loop through concurrently many things without blocking on each individual
    iteration but blocks (waits) for all tasks to run.
    ref:
    - https://stackoverflow.com/questions/56161595/how-to-use-async-for-in-python/72758067#72758067
    """
    tasks = []

    async for obj in my_async_generator:
        # Python 3.7+. Use ensure_future for older versions.
        task = asyncio.create_task(process_obj(obj))  # concurrently dispatches a coroutine to be executed.
        tasks.append(task)

    await asyncio.gather(*tasks)


async def process_obj(obj):
    await asyncio.sleep(5)  # expensive IO


if __name__ == '__main__':
    # - test asyncio
    s = time.perf_counter()
    asyncio.run(process_all())
    # - print stats
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
    print('Success, done!\a')
