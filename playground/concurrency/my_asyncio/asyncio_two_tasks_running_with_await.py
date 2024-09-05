"""
Example showing how two scheduled tasks exchange control with await statements and are allowed to interleave in a controlled
way. Note, nothing is running in true parallelism. Do the uncomments to see how without the await calls -- the code
blocks and doesn't interleave anymore. This is still useful because the asyncio.sleep(delay) simulate real IOs that can
be thought of in parallel.

recall:
- await = does not continue until the await is completed. But also returns control to the caller to do something else.
Usually this is assumed that some slow io is being doing during the await or a task has already been scheduled.
- async = defines a coroutine i.e. a function that is able to give control back to the caller or call other coroutines.
- await asyncio.sleep(5) = simulates an expensive io that lasts 5 seconds before returns or allowing the next code inside
its own coroutine to run. However, the await does give it power to give control back to the caller to run other stuff if
it can.

ref:
    - insipred from: https://www.youtube.com/watch?v=t5Bo1Je9EmE&t=808s
"""
from asyncio import Task

import time

import asyncio


async def fetch_data():
    print('fetching data')
    await asyncio.sleep(10)  # simulates fetching data from expensive io e.g. coq
    # time.sleep(10)  # uncomment to have this code to block -- even if scheduled as a task the other tasks won't be able to run!
    print('done fetching data')
    return {'data': 1}  # simulates returning e.g. json data


async def print_numbers(num_steps: int = 10):
    """ """
    for i in range(num_steps):
        print(i)
        # gives back control to caller e.g. event loop but it doesn't execute the rest i.e. the next loop it until the delay is done
        await asyncio.sleep(0.5)
        # time.sleep(1)  # uncomment to have this code block and not return control to the caller. Note it also means that this code never gives control back to data fetcheer until the whole loop is done.


async def main_doesnt_complete_async_calls():
    # schedules both tasks to run
    task1 = asyncio.create_task(fetch_data())
    task2 = asyncio.create_task(print_numbers())

    value = task1
    assert isinstance(value, Task)
    assert isinstance(task2, Task)
    assert value != {'data': 1}
    print(value)


async def main_does_complete_async_calls():
    # schedules both tasks to run
    task1 = asyncio.create_task(fetch_data())
    task2 = asyncio.create_task(print_numbers())

    # does not execute the code bellow it until task1 is done.
    value = await task1
    assert value == {'data': 1}
    print(value)
    ret = await task2
    assert ret is None


# - __main__

if __name__ == '__main__':
    print('---- in __main__ ')

    # # - schedules tasks but doesn't complete them nor do we get the return values
    start_time: float = time.time()
    asyncio.run(main_doesnt_complete_async_calls())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")

    # - schedules tasks and complete them and get the return values
    start_time: float = time.time()
    asyncio.run(main_does_complete_async_calls())
    duration = time.time() - start_time
    print(f"Success: {duration} seconds.\n\a")

    print('---- Done')
