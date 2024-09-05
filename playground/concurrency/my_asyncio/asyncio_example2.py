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
    3) asynchronous routies are able to "pause" while waiting on their ultimate results and allow
    other indepedent programs to run in the meantime.
- coroutines = a coroutine is a function that can suspend its execution before reaching return & it can indirectly pass
    control to another coroutine for some time.
    In python coroutines are specialized versions of (python) generators.
    Coroutines are computer program components that generalize subroutines for non-preemptive multitasking, by allowing
    execution to be suspended and resumed.
- asynchronous IO (async IO) = language agnostic paradigm that implements the computational idea of async IO.
- async/await = python keywords used to manage coroutines (check if it's true python keywords that implement async IO).
- asyncio = python pkg that provides the api for running and managing coroutines
    (check if this is true: the pkg that implements the async IO paradigm)
async = defines a coroutine. This doesn't define a real io, it only defines a function that can give up and give the
    execution power to other coroutines or the (asyncio) event loop.

await = the key word that does (mainly) two things 1) gives control back to the event loop to see if there is something
    else to run if we called it on a real expensive io operation (e.g. calling network, printer, etc) 2) gives control to
    the new coroutine that it is awaiting. If this is your own code with async then it means it will go into this new async
    function (coroutine) you defined.
    No real async benefits are being experienced until you call a real io.

async IO is a single-threaded, single-process design: it uses cooperative multitasking.

cons: - not all libraries support the async IO paradigm in python (e.g. asyncio, trio, etc).

"""

import asyncio

async def count(coroutine_name: str = ''):
    print(f'coroutine name: {coroutine_name=}')
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count('cor1'), count('cor2'), count('cor3'))

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

