"""
-- Notes from [1]

See asyncio_example.py file.

1. https://realpython.com/python-concurrency/
2. https://realpython.com/async-io-python/
3. https://stackoverflow.com/a/51116910/6843734
4. https://realpython.com/python-async-features/
5. https://trio.readthedocs.io/en/stable/tutorial.html trio has a gitter channel for asking and they try to make their
tutorial very easy to understand!V

todo - read [2] later (or [3] but thats not a tutorial and its more details so perhaps not a priority).


Appendix:

For I/O-bound problems, there’s a general rule of thumb in the Python community:
    “Use asyncio when you can, threading when you must.”
asyncio can provide the best speed up for this type of program, but sometimes you will require critical libraries that
have not been ported to take advantage of asyncio.
Remember that any task that doesn’t give up control to the event loop will block all of the other tasks

-- Notes from [2]

definitions:

- asynchronous = 1) dictionary def: not happening at the same time e.g. happening indepedently
    2) computing def: happening independently of the main program flow.
    3) asynchronous routies are able to "pause" whle waiting on their ultimate results and allow other programs to run
    in the meantime.
- coroutines = a coroutine is a function that can suspend its execution before reaching return and it can indirectly pass
    control to another coroutine for some time.
    In python coroutines are specialized versions of (python) generators.
- asynchronous IO (async IO) = language agnostic paradigm that implements the computational idea of async IO.
- async/await = python keywords ued to manage coroutines (check if it's true python keywords that implement async IO).
- asyncio = python pkg that provides the api for running and managing coroutines
    (check if this is true: the pkg that implements the async IO paradigm)

async IO is a single-threaded, single-process design: it uses cooperative multitasking.

"""

import asyncio
from asyncio import Task
from asyncio.events import AbstractEventLoop

import aiohttp
from aiohttp.client import ClientSession

import time


async def download_site(session: ClientSession, url: str) -> str:
    async with session.get(url) as response:
        print(f"Read {response.content_length} from {url}")
        return response.text()


async def download_all_sites(sites: list[str]):
    async with aiohttp.ClientSession() as session:
        tasks: list[Task] = []
        for url in sites:
            task: Task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        content_from_url: list[str] = await asyncio.gather(*tasks, return_exceptions=True)
        print(content_from_url)
        return content_from_url


if __name__ == "__main__":
    # - args
    num_sites: int = 80
    sites: list[str] = ["https://www.jython.org", "http://olympus.realpython.org/dice"] * num_sites
    start_time: float = time.time()

    # - run event loop manager and run all tasks with cooperative concurrency
    # asyncio.get_event_loop().run_until_complete(download_all_sites(sites))
    event_loop: AbstractEventLoop = asyncio.get_event_loop()
    asyncio.run(download_all_sites(sites))

    # - print stats about the content download and duration
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")
    print('Success.\a')

