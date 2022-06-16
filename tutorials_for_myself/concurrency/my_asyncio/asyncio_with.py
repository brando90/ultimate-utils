"""
1. https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines
2. https://realpython.com/python-concurrency/
3. https://stackoverflow.com/questions/67092070/why-do-we-need-async-for-and-async-with

todo - async with, async for.

todo: meaning of:
    - The async for and async with statements are only needed to the extent that using plain for or with would “break”
        the nature of await in the coroutine. This distinction between asynchronicity and concurrency is a key one to grasp
    - One exception to this that you’ll see in the next code is the async with statement, which creates a context
        manager from an object you would normally await. While the semantics are a little different, the idea is the
        same: to flag this context manager as something that can get swapped out.
    - download_site() at the top is almost identical to the threading version with the exception of the async keyword on
        the function definition line and the async with keywords when you actually call session.get().
        You’ll see later why Session can be passed in here rather than using thread-local storage.
    - An asynchronous context manager is a context manager that is able to suspend execution in its enter and exit
        methods.
"""

import asyncio
from asyncio import Task

import time

import aiohttp
from aiohttp.client_reqrep import ClientResponse

from typing import Coroutine


async def download_site(coroutine_name: str, session: aiohttp.ClientSession, url: str) -> ClientResponse:
    """
    Calls an expensive io (get data from a url) using the special session (awaitable) object. Note that not all objects
    are awaitable.
    """
    # - the with statement is bad here in my opion since async with is already mysterious and it's being used twice
    # async with session.get(url) as response:
    #     print("Read {0} from {1}".format(response.content_length, url))
    # - this won't work since it only creates the coroutine. It **has** to be awaited. The trick to have it be (buggy)
    # synchronous is to have the main coroutine call each task we want in order instead of giving all the tasks we want
    # at once to the vent loop e.g. with the asyncio.gather which gives all coroutines, gets the result in a list and
    # thus doesn't block!
    # response = session.get(url)
    # - right way to do async code is to have this await so someone else can run. Note, if the download_site/ parent
    # program is awaited in a for loop this won't work regardless.
    response = await session.get(url)
    print(f"Read {response.content_length} from {url} using {coroutine_name=}")
    return response

async def download_all_sites_not_actually_async_buggy(sites: list[str]) -> list[ClientResponse]:
    """
    Code to demo the none async code. The code isn't truly asynchronous/concurrent because we are awaiting all the io
    calls (to the network) in the for loop. To avoid this issue, give the list of coroutines to a function that actually
    dispatches the io like asyncio.gather.

    My understanding is that async with allows the object given to be a awaitable object. This means that the object
    created is an object that does io calls so it might block so it's often the case we await it. Recall that when we
    run await f() f is either 1) coroutine that gains control (but might block code!) or 2) io call that takes a long
    time. But because of how python works after the await finishes the program expects the response to "actually be
    there". Thus, doing await blindly doesn't speed up the code. Do awaits on real io calls and call them with things
    that give it to the event loop (e.g. asyncio.gather).

    """
    # - create a awaitable object without having the context manager explode if it gives up execution.
    # - crucially, the session is an aiosession - so it is actually awaitable so we can actually give it to
    # - asyncio.gather and thus in the async code we truly take advantage of the concurrency of asynchronous programming
    async with aiohttp.ClientSession() as session:
    # with aiohttp.ClientSession() as session:  # won't work because there is an await inside this with
        tasks: list[Task] = []
        responses: list[ClientResponse] = []
        for i, url in enumerate(sites):
            task: Task = asyncio.ensure_future(download_site(f'coroutine{i}', session, url))
            tasks.append(task)
            response: ClientResponse = await session.get(url)
            responses.append(response)
        return responses


async def download_all_sites_truly_async(sites: list[str]) -> list[ClientResponse]:
    """
    Truly async program that calls creates a bunch of coroutines that download data from urls and the uses gather to
    have the event loop run it asynchronously (and thus efficiently). Note there is only one process though.
    """
    # - indicates that session is an async obj that will likely be awaited since it likely does an expensive io that
    # - waits so it wants to give control back to the event loop or other coroutines so they can do stuff while the
    # - io happens
    async with aiohttp.ClientSession() as session:
        tasks: list[Task] = []
        for i, url in enumerate(sites):
            task: Task = asyncio.ensure_future(download_site(f'coroutine{i}', session, url))
            tasks.append(task)
        responses: list[ClientResponse] = await asyncio.gather(*tasks, return_exceptions=True)
        return responses


if __name__ == "__main__":
    # - args
    sites = ["https://www.jython.org", "http://olympus.realpython.org/dice"] * 80
    start_time = time.time()

    # - run main async code
    # main_coroutine: Coroutine = download_all_sites_truly_async(sites)
    main_coroutine: Coroutine = download_all_sites_not_actually_async_buggy(sites)
    responses: list[ClientResponse] = asyncio.run(main_coroutine)

    # - print stats
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")
    print('Success, done!\a')