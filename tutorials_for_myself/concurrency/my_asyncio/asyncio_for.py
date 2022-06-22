"""
File demonstrating looping through asynchronous iterators.

Definitions:
async for = allows to loop through an asynchronous iterator.

plan:
- my guess is a good plan is to go through the generator & yield example in real python: https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines

Ref:
    - https://stackoverflow.com/questions/56161595/how-to-use-async-for-in-python
    - https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines
    - (idk, don't like this one too much... https://stackoverflow.com/questions/67092070/why-do-we-need-async-for-and-async-with

todo: create a concrete async forn example, likely contrasting it with a blocking for and how to "implement"
    an equivalent for with no async for but instead with an for with awaits.
"""

import asyncio

import time


def test1():
    """"""
    pass

if __name__ == '__main__':
    # - test asyncio
    s = time.perf_counter()
    test1()
    # - print stats
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
    print('Success, done!\a')