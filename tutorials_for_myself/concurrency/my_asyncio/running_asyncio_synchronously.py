"""

I am using python's async (asyncio) library. Currently I have a deadlock. I do not want to debug the deadlock. I want my code to act synchronously, as if python's asyncio wasn't present but crucially **without editing my code**. How do I run my python asyncio code so that I don't have to edit it (or very minimally edit it) and it stop using asyncio's asynchronous features that might be deadlocking it?

If you want to run your asyncio-based code synchronously without making significant changes to your existing code, you can create a wrapper function that runs your asynchronous code inside an event loop using asyncio.run(). This will effectively make your code synchronous since the event loop will execute and wait for each async task in the order they are called.

Here's an example of how you can achieve this:

Suppose you have a file async_code.py with the following content:

python
Copy code
import asyncio

async def async_task1():
    print("Task 1 started")
    await asyncio.sleep(1)
    print("Task 1 finished")

async def async_task2():
    print("Task 2 started")
    await asyncio.sleep(1)
    print("Task 2 finished")

async def main():
    await async_task1()
    await async_task2()
Now, create a wrapper script sync_wrapper.py that runs the main function of your existing async_code.py:

python
Copy code
import asyncio
from async_code import main

def run_sync(async_function):
    asyncio.run(async_function())

if __name__ == "__main__":
    run_sync(main)
By doing this, you won't have to significantly edit your existing code, and it will run synchronously.

However, if you still experience a deadlock, it might be due to improper use of asyncio primitives, such as asyncio.Lock, asyncio.Queue, or asyncio.Condition. In that case, you might need to refactor your code to fix those issues.




Brando Miranda
I am not using any of those. I am only using the standard await calls and async key words. Therefore, would the code you suggested be sufficient that if there is a deadlock caused by asyncio that it will be by passed? I want essentially that my code is "unchanged, not refactored" (e.g. by running asycio with a synchronous flag) and therefore run fully synchronously.
Apologies for the confusion. If you're only using await and async keywords, and not using any asyncio primitives that can cause deadlocks, then the proposed solution should work in most cases. The asyncio.run() function will take care of running your asynchronous code synchronously, as it completes one task at a time.

However, it is important to note that asyncio.run() is not a perfect "synchronous flag" for asyncio. It will still use an event loop to run your asynchronous code, and there might be edge cases where running the code synchronously with this method might not work as expected.

Nonetheless, using asyncio.run() as shown in the previous example should help you bypass most issues related to deadlocks, as long as you're not using any additional asyncio features that could cause deadlocks.
"""

import asyncio


async def async_task1():
    print("Task 1 started")
    await asyncio.sleep(1)
    print("Task 1 finished")


async def async_task2():
    print("Task 2 started")
    await asyncio.sleep(1)
    print("Task 2 finished")


async def main():
    await async_task1()
    await async_task2()


import asyncio


def run_sync(async_function):
    asyncio.run(async_function())


if __name__ == "__main__":
    run_sync(main)
