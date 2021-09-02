"""
1. https://realpython.com/async-io-python/#the-asyncawait-syntax-and-native-coroutines

todo - async with, async for.
"""

import asyncio
from asyncio.events import AbstractEventLoop

import time

from typing import Coroutine

def test1():
    """
    """
    pass

if __name__ == '__main__':
    # - test asyncio
    s = time.perf_counter()
    test1()
    # - print stats
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
    print('Success, done!\a')