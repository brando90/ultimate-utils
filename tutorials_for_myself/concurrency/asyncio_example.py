"""
-- Notes from [1]

Threading and asyncio both run on a single processor and therefore only run one at a time [1].

Note: threads.py has a very good block with good defintions for io-bound, cpu-bound if you need to recall it.

General idea for asyncio is that there is a general event loop that controls how and when each tasks gets run.
The event loop is aware of each task and knows what states they are in.
For simplicitly of exponsition assume there are only two states:
a) Ready state
b) Waiting state

a indicates that a task has work to do and can be run - while b indicates that a task is waiting for a response from an
external thing (e.g. io, printer, disk, network, coq, etc). This simplified event loop has two lists of tasks
(ready_to_run_lst, waiting_lst) and runs things from the ready to run list. Once a task runs it is in complete control
until it cooperatively hands back control to the event loop.

The way it works is that the task that was ran does what it needs to do (usually an io operation, or an interleaved op
or something like that) but crucially it gives control back to the event loop when the task wants.
Note that this means the task might not have fully completed getting what is "fully needs".
This is probably useful when the user whats to implement the interleaving himself.
Once the task cooperatively gives back control to the event loop it is placed by the event loop in either the
ready to run list or wainting list (depending how fast the io ran, etc). Then the event loop goes through the wainting
loop to see if anything waiting has "returned".
Once the all the tasks have been sorted into the right list the event loop is able to choose what to run next (e.g. by
choosing the one that has been waiting to be ran the longest). This repeats until the event loop code you wrote is done.

The crucial point (and distinction with threads) that we want to emphasizes is that in asyncio, an operation is never
interrupted in the middle and every switching/interleaving is done deliberately by the programmer.
In a way you don't have to worry about making your code thread safe.

For more details see [2], [3].

Asyncio syntax:

i) await = this is where the code you wrote calls an expensive function (e.g. an io) and thus hands back control to the
    event loop. Then the event loop will likely put it in the waiting loop and runs some other task. Likely eventually
    the event loop comes back to this function and runs the remaining code given that we have the value from the io now.
iii) async = approximately a flag that tells python the defined function might use await. This is not strictly true but
    it gives you a simple model while your getting started. todo - clarify async.
    todo - context manager with async

ii) awaiting = when you call something (e.g. a function) that requires waiting for the io response/return/value.
iv) async with = this creates a context manager from an object you would normally await - i.e. an object you would
    wait to get the return value from an io. So usually we swap out (switch) from this object.
    todo - e.g.


Note: - any function that calls await needs to be marked with async or you’ll get a syntax error otherwise.
      - a task never gives up control without intentionally doing so e.g. never in the middle of an op.


Cons: - note how this also requires more thinking carefully (but feels less dangerous than threading due to no pre-emptive
    switching) due to the concurrency. Another disadvantage is again the idisocyncracies of using this in python + learning
    new syntax and details for it to actually work.
      - understanding the semanics of new syntax + learning where to really put the syntax to avoid semantic errors.
      - we needed a special asycio compatible lib for requests, since the normal requests is not designed to inform
    the event loop that it's block (or done blocking)
      - if one of the tasks doesn't cooperate properly then the whole code can be a mess and slow it down.
      - not all libraries support the async IO paradigm in python (e.g. asyncio, trio, etc).

Pro: + despite learning where to put await and async might be annoying it forces your to think carefully about your code
    which on itself can be an advantage (e.g. better, faster, less bugs due to thinking carefully)
     + often faster...? (skeptical)

1. https://realpython.com/python-concurrency/
2. https://realpython.com/async-io-python/
3. https://stackoverflow.com/a/51116910/6843734

todo - read [2] later (or [3] but thats not a tutorial and its more details so perhaps not a priority).

asynchronous = 1) dictionary def: not happening at the same time e.g. happening indepedently 2) computing def: happening independently of the main program flow

Appendix:

For I/O-bound problems, there’s a general rule of thumb in the Python community:
    “Use asyncio when you can, threading when you must.”
asyncio can provide the best speed up for this type of program, but sometimes you will require critical libraries that
have not been ported to take advantage of asyncio.
Remember that any task that doesn’t give up control to the event loop will block all of the other tasks

-- Notes from [2]

see asyncio_example2.py file.

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

