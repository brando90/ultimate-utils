"""
-- Notes from [1]

Threading and asyncio both run on a single processor and therefore only run one at a time [1]. It's cooperative concurrency.

Note: threads.py has a very good block with good defintions for io-bound, cpu-bound if you need to recall it.
Note: coroutine is an important definition to understand before proceeding. Definition provided at the end of this tutorial.

General idea for asyncio is that there is a general event loop that controls how and when each tasks gets run.
The event loop is aware of each task and knows what states they are in.
For simplicitly of exponsition assume there are only two states:
a) Ready state
b) Waiting state

a) indicates that a task has work to do and can be run - while b) indicates that a task is waiting for a response from an
external thing (e.g. io, printer, disk, network, coq, etc). This simplified event loop has two lists of tasks
(ready_to_run_lst, waiting_lst) and runs things from the ready to run list. Once a task runs it is in complete control
until it cooperatively hands back control to the event loop.

The way it works is that the task that was ran does what it needs to do (usually an io operation, or an interleaved op
or something like that) but crucially it gives control back to the event loop when the running task (with control) thinks is best.
(Note that this means the task might not have fully completed getting what is "fully needs".
This is probably useful when the user whats to implement the interleaving himself.)
Once the task cooperatively gives back control to the event loop it is placed by the event loop in either the
ready to run list or waiting list (depending how fast the io ran, etc). Then the event loop goes through the waiting
loop to see if anything waiting has "returned".
Once all the tasks have been sorted into the right list the event loop is able to choose what to run next (e.g. by
choosing the one that has been waiting to be ran the longest). This repeats until the event loop code you wrote is done.

The crucial point (and distinction with threads) that we want to emphasizes is that in asyncio, an operation is never
interrupted in the middle and every switching/interleaving is done deliberately by the programmer.
In a way you don't have to worry about making your code thread safe.

For more details see [2], [3].

Asyncio syntax:

i) await = this is where the code you wrote calls an expensive function (e.g. an io) and thus hands back control to the
    event loop. Then the event loop will likely put it in the waiting loop and runs some other task. Likely eventually
    the event loop comes back to this function and runs the remaining code given that we have the value from the io now.
    await = the key word that does (mainly) two things 1) gives control back to the event loop to see if there is something
    else to run if we called it on a real expensive io operation (e.g. calling network, printer, etc) 2) gives control to
    the new coroutine (code that might give up control copperatively) that it is awaiting. If this is your own code with async
    then it means it will go into this new async function (coroutine) you defined.
    No real async benefits are being experienced until you call (await) a real io e.g. asyncio.sleep is the typical debug example.
    todo: clarify, I think await doesn't actually give control back to the event loop but instead runs the "coroutine" this
        await is pointing too. This means that if it's a real IO then it will actually give it back to the event loop
        to do something else. In this case it is actually doing something "in parallel" in the async way.
        Otherwise, it is your own python coroutine and thus gives it the control but "no true async parallelism" happens.
iii) async = approximately a flag that tells python the defined function might use await. This is not strictly true but
    it gives you a simple model while your getting started. todo - clarify async.
    async = defines a coroutine. This doesn't define a real io, it only defines a function that can give up and give the
    execution power to other coroutines or the (asyncio) event loop.
    todo - context manager with async

ii) awaiting = when you call something (e.g. a function) that usually requires waiting for the io response/return/value.
    todo: though it seems it's also the python keyword to give control to a coroutine you wrote in python or give
    control to the event loop assuming your awaiting an actual io call.
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

asynchronous = 1) dictionary def: not happening at the same time
    e.g. happening indepedently 2) computing def: happening independently of the main program flow

couroutine = are computer program components that generalize subroutines for non-preemptive multitasking, by allowing execution to be suspended and resumed.
So basically it's a routine/"function" that can give up control in "a controlled way" (i.e. not randomly like with threads).
Usually they are associated with a single process -- so it's concurrent but not parallel.
Interesting note: Coroutines are well-suited for implementing familiar program components such as cooperative tasks, exceptions, event loops, iterators, infinite lists and pipes.
Likely we have an event loop in this document as an example. I guess yield and operators too are good examples!
Interesting contrast with subroutines: Subroutines are special cases of coroutines.[3] When subroutines are invoked, execution begins at the start,
and once a subroutine exits, it is finished; an instance of a subroutine only returns once, and does not hold state between invocations.
By contrast, coroutines can exit by calling other coroutines, which may later return to the point where they were invoked in the original coroutine;
from the coroutine's point of view, it is not exiting but calling another coroutine.
Coroutines are very similar to threads. However, coroutines are cooperatively multitasked, whereas threads are typically preemptively multitasked.

event loop = event loop is a programming construct or design pattern that waits for and dispatches events or messages in a program.

Appendix:

For I/O-bound problems, there’s a general rule of thumb in the Python community:
    “Use asyncio when you can, threading when you must.”
asyncio can provide the best speed up for this type of program, but sometimes you will require critical libraries that
have not been ported to take advantage of asyncio.
Remember that any task that doesn’t give up control to the event loop will block all of the other tasks

-- Notes from [2]

see asyncio_example2.py file.

The sync code (synchronous.py) should have taken longer e.g. in one run the async file (asyncio_example.py) took:
Downloaded 160 sites in 0.4063692092895508 seconds
While the sync option took:
Downloaded 160 in 3.351937770843506 seconds
"""

import asyncio
from asyncio import Task
from asyncio.events import AbstractEventLoop

import aiohttp
from aiohttp import ClientResponse
from aiohttp.client import ClientSession

from typing import Coroutine

import time


async def download_site(session: ClientSession, url: str) -> str:
    async with session.get(url) as response:
        print(f"Read {response.content_length} from {url}")
        return response.text()


async def download_all_sites(sites: list[str]) -> list[str]:
    # async with = this creates a context manager from an object you would normally await - i.e. an object you would wait to get the return value from an io. So usually we swap out (switch) from this object.
    async with aiohttp.ClientSession() as session:  # we will usually away session.FUNCS
        # create all the download code a coroutines/task to be later managed/run by the event loop
        tasks: list[Task] = []
        for url in sites:
            # creates a task from a coroutine todo: basically it seems it creates a callable coroutine? (i.e. function that is able to give up control cooperatively or runs an external io and also thus gives back control cooperatively to the event loop). read more? https://stackoverflow.com/questions/36342899/asyncio-ensure-future-vs-baseeventloop-create-task-vs-simple-coroutine
            task: Task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        # runs tasks/coroutines in the event loop and aggrates the results. todo: does this halt until all coroutines have returned? I think so due to the paridgm of how async code works.
        content_from_url: list[ClientResponse.text] = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(content_from_url[0], Coroutine)  # note allresponses are coroutines
        print(f'result after aggregating/doing all coroutine tasks/jobs = {content_from_url=}')
        # this is needed since the response is in a coroutine object for some reason, if this part confuses you see: https://stackoverflow.com/a/72606161/1601580
        content_from_url_as_str: list[str] = await asyncio.gather(*content_from_url, return_exceptions=True)
        print(f'result after getting response from coroutines that hold the text = {content_from_url_as_str=}')
        return content_from_url_as_str


if __name__ == "__main__":
    # - args
    num_sites: int = 80
    sites: list[str] = ["https://www.jython.org", "http://olympus.realpython.org/dice"] * num_sites
    start_time: float = time.time()

    # - run the same 160 tasks but without async paradigm, should be slower!
    # note: you can't actually do this here because you have the async definitions to your functions.
    # to test the synchronous version see the synchronous.py file. Then compare the two run times.
    # await download_all_sites(sites)
    # download_all_sites(sites)

    # - Execute the coroutine coro and return the result.
    asyncio.run(download_all_sites(sites))

    # - run event loop manager and run all tasks with cooperative concurrency
    # asyncio.get_event_loop().run_until_complete(download_all_sites(sites))

    # makes explicit the creation of the event loop that manages the coroutines & external ios
    # event_loop: AbstractEventLoop = asyncio.get_event_loop()
    # asyncio.run(download_all_sites(sites))

    # making creating the coroutine that hasn't been ran yet with it's args explicit
    # event_loop: AbstractEventLoop = asyncio.get_event_loop()
    # download_all_sites_coroutine: Coroutine = download_all_sites(sites)
    # asyncio.run(download_all_sites_coroutine)

    # - print stats about the content download and duration
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")
    print('Success.\a')
