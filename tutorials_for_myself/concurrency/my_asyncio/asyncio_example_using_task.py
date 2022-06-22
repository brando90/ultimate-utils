"""
Goal: example showing how to run multiple things concurrently with
the task object.

Main take away:
- it seems that creating task objects run code concurrently (and truly in parallel! even if it's my own code)
- simply doing await my_coroutine(args) to some coroutine doesn't run it in true concurrency. yes it will switch the
to the awaited coroutine but it seems it needs something else to (e.g. a task) to run in paralle.

recall:
- coroutines are functions that give up control back to the caller.
- await runs coroutines (can only be ran inside a coroutine, so in things with async)
- async defines coroutines.

todo: asyncio sleep vs normal sleep

ref:
    - inspired from: https://www.youtube.com/watch?v=t5Bo1Je9EmE&t=808s
"""
import time

import asyncio


async def my_print(msg: str, delay: float = 1.5):
    print(f'{msg=} (inside: {my_print=})')
    await asyncio.sleep(delay=delay)


# -  sync version

async def main_sync():
    """Looks concurrent but it's actually blocking/sync."""
    print('main')
    await my_print('text')
    print('finished')


# - async version

async def main_async():
    """Runs one of our own tasks concurrently!"""
    print('main')
    task = asyncio.create_task(my_print('text'))  # Schedule the execution of a coroutine object in a spawn task.
    print('finished')


# - __main__

if __name__ == '__main__':
    print('in __main__ ')
    # - run sync
    start_time: float = time.time()
    asyncio.run(main_sync())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")
    # - run async
    start_time: float = time.time()
    asyncio.run(main_async())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")
