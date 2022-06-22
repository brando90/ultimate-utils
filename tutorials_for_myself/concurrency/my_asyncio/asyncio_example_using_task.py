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
- note: await asyncio.sleep() simulates a expensive IO operation but also, the await means the next line won't be
executed until the sleep ("expensive io") is done.

todo: asyncio sleep vs normal sleep

ref:
    - inspired from: https://www.youtube.com/watch?v=t5Bo1Je9EmE&t=808s
"""
import time

import asyncio


async def my_print_gives_back_control(msg: str, delay: float = 1.5):
    """ Prints sleeps for an amount of time, but gives control back to the caller during the sleep and after the sleep
    is completed it then continues execution.
    """
    print(f'start: {msg=} (inside: {my_print_gives_back_control=})')
    # awaiting the expensive io (asyncio.sleep) means the caller gets back control
    await asyncio.sleep(delay=delay)
    print(f'done: {msg=} (inside: {my_print_gives_back_control=})')


async def my_print_blocking(msg: str, delay: float = 1.5):
    """ Although it's a coroutine due to async, it actually blocks due to no await being callded inside the sleep.
    If you wrap the sleep in a asyncio.sleep + await it, then that signals to only continue the code inside here after
    the await but it also allows this function to give back control to the caller.
    """
    print(f'start: {msg=} (inside: {my_print_gives_back_control=})')
    # code will block since there is no await on the expensive io (sleep) here. await gives back control to caller.
    time.sleep(delay)
    print(f'done: {msg=} (inside: {my_print_gives_back_control=})')


# -  sync version

async def main_sync():
    """Looks concurrent but it's actually blocking/sync."""
    print('main')
    ret = await my_print_gives_back_control('text')
    assert ret is None
    print('finished')


async def main_sync2_await_task():
    """Looks concurrent but it's actually blocking/sync but by awaint the task."""
    print('main')
    task = asyncio.create_task(my_print_gives_back_control('text'))
    ret = await task
    assert ret is None
    print('finished')


# - async version

async def main_async():
    """Runs one of our own tasks concurrently!"""
    print('main')
    task = asyncio.create_task(my_print_gives_back_control('text'))
    print('finished')


async def main_async2_run_task_asap_once_an_await_is_seen():
    """
    Demos how a task runs in parallel once an await is seen that actually gives back control to the caller via another
    await inside it.
    The result is that once the await is reached the task is ran and since 1 < 5 it finishes and then we finish the
    initial await we had.

    Note: try 1st delay longer than 2nd delay. Result is that we didn't wait for the "parallel" task to finish if the
    await finished sooner!
    """
    print('main')
    task = asyncio.create_task(my_print_gives_back_control('text', delay=1))
    ret = await my_print_gives_back_control('text_wait_for_me_to_be_done', delay=5)
    assert ret is None
    print('finished')


# - sync/async mix

async def main_async3():
    """
    Runs the task asap but the await is on a blocking code that never gives back control to the caller. Thus, the task
    is not runat the first await so we always see the complete termination of the blocking code.
    However surprisingly it does try to run the task at the end of the code but always fails.
    If you await task however it does complete it.
    """
    print('main')
    task = asyncio.create_task(my_print_gives_back_control('text', delay=0.01))
    ret = await my_print_blocking('text_wait_for_me_to_be_done', delay=5)
    assert ret is None
    print('finished')


# - __main__

if __name__ == '__main__':
    print('---- in __main__ ')

    # - run sync
    start_time: float = time.time()
    asyncio.run(main_sync())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")
    # - run sync
    start_time: float = time.time()
    asyncio.run(main_sync2_await_task())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")
    # - run async
    start_time: float = time.time()
    asyncio.run(main_async())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")
    # - run async
    start_time: float = time.time()
    asyncio.run(main_async2_run_task_asap_once_an_await_is_seen())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")
    # - run async
    start_time: float = time.time()
    asyncio.run(main_async3())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")

    print('---- Done')
