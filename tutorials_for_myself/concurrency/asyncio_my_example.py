"""
Best example is from count [1].

1. https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines

todo - async with, async for.



async = defines a coroutine. This doesn't define a real io, it only defines a function that can give up and give the
    execution power to other coroutines or the (asyncio) event loop.

await = the key word that does (mainly) two things 1) gives control back to the event loop to see if there is something
    else to run if we called it on a real expensive io operation (e.g. calling network, printer, etc) 2) gives control to
    the new coroutine that it is awaiting. If this is your own code with async then it means it will go into this new async
    function (coroutine) you defined.
    No real async benefits are being experienced until you call a real io.

"""

import asyncio
from asyncio.events import AbstractEventLoop

import time

from typing import Coroutine

async def expensive_io(msg_to_io_device: str = '', io_time: int = 2.5) -> str:
    """
    Simulates an expensive IO and returns the fake result.

    We do need asyncio.sleep and not just time.sleep because although this expensive_io is named a coroutine it is not
    a real io operation. It seems that coroutines are things that swap control between your code. The io operation never
    gets control. So actually creating a coroutine called expensive_io is confusing because it's not really an io. But
    a function like handle_expensive_io as a coroutine makes sense perhaps depending on how you want to craft your code.
    Overall the main thing to remember is that await gives control to a different coroutine (if it's your code) and if
    it's not a coroutine but a true io then doesn't actually give control to the io but instead it gives control back to
    the event loop to see if there is something else we can run while we wait for the io.
    """
    # time.sleep(io_time)  # simulating expensive io
    await asyncio.sleep(io_time)  # note this might usually return something e.g. in Coq.
    return f'expensive io done {msg_to_io_device}'

async def my_coroutine(coroutine_name: str = 'my_coroutine'):
    print(f'running coroutine named: {coroutine_name}')
    result1: str = await expensive_io(coroutine_name)
    print(f'{result1=}')
    result2: str = await expensive_io(coroutine_name)
    print(f'{result2=}')

async def main1(coroutines: list[Coroutine]):
    print('--- starting main1 coroutine')
    results: list = await asyncio.gather(*coroutines, return_exceptions=True)
    print('--- Done with main1 coroutine')
    return f'Result from main1: {results=}'

def test1():
    """
    Single coroutine that waits a couple of times but since the event loop has nothing else to do it is forced to wait.

    Learned from this example: you *have* to await a routine that is a asyncio coroutine e.g. with the async keyword.
    This is interesting because it means that calling a async function actually creates a coroutine (in python it
    creates a generator).
    """
    # - run event loop
    event_loop: AbstractEventLoop = asyncio.get_event_loop()
    coroutine: Coroutine = my_coroutine()
    event_loop.run_until_complete(coroutine)

def test2():
    """
    Multiple coroutines.

    What seems to happen is that asyncio.run has the event loop and it runs the coroutine given. Thus since this in the
    async paradigm, it means the given coroutine has all control. For example if it's gather then gather has all control
    of the computation its sort of the main routine. In gather what likely happens is that it just awaits all the given
    coroutines immediately and once all are done aggregates results. If you pass main then what happens is that main is
    given all the control. Then if you have awaits inside your main then those are given full control. In the case of
    simulating an expensive operation you need to call asyncio.sleep - otherwise the code will block and slow down.
    In this case it's because you need a fake a real io.
    """
    # - run event loop
    event_loop: AbstractEventLoop = asyncio.get_event_loop()
    coroutines: list[Coroutine] = [my_coroutine('coroutine1'), my_coroutine('coroutine2')]
    # result = event_loop.run_until_complete(asyncio.gather(*coroutines, return_exceptions=True))
    # result = event_loop.run_until_complete(main1(coroutines))
    result = asyncio.run(main1(coroutines))
    print(f'{result=}')

if __name__ == '__main__':
    # - test asyncio
    s = time.perf_counter()
    test1()
    # test2()
    # - print stats
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
    print('Success, done!\a')