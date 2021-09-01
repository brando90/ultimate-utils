"""
Threading and asyncio both run on a single processor and therefore only run one at a time [1].

In threading the OS knows about the each thread and can pre-emptively (obstruct ahead of time ~ at any time) it's
computation to later resume it.
This type of concurrency is nice because the code doesn't have to do anything to do the switch explicitly - it is also
hard for the same reason (e.g. hard to forsee when order of computation happens - since it happens "at any time").

On the other had asyncio uses cooperative multitasking - i.e. the tasks have to explicitly (= cooperatively) give up
control and switch who's turn it is to do computation.
The benefit is that you always know up front where your tasks will be swithed.

Concurrency helps speed up in two cases 1) IO-bound 2) CPU-bound

a) IO-bound = what bounds the speed of your program is the speed of input-output (external) resources.
Usually this means that your cpu (program) is working with external resources that are slower
(e.g. filesystems, networks, coq, printer, internet, disk etc.)
    - speed up: interleaving io-operations (so mp, threading, asyncio speed it up)
    - todo: how to speed up? Right now I think the way to spped it up is by overlapping io calls so all the io calls
        happen simultaneously. I'd guess that both mp and asyncio/threading would speed it up.

b) CPU-bound = what bounds the speed of your program is the speed of your program/cpu e.g. calling an external resources is
faster than your code/cpu so your program/cpu is the bottleneck.
    - speed up: fitting more computation somehow on the same time.
    - todo: how to speed up? Right now my guess is that the only way to speed up is with mp. This is the only way I can
        think of to speed it up. It seems that both rely really on simulataneity, but one is within your cpu or not
        (e.g. remote procedure calls can speed things up but someone has to do it - way to "cheat" this of course).
        CPU-bound I can't imagine it speeds things up since it takes the same time to finish wether they are interleaved
        or not. But I can it making a different for user UI experience. It wouldn't be nice that the computer freezes on
        you.


Threading cons: not only is it difficult to think about usually but sometimes there will be small idiosyncratic
implementation details (e.g. the threading.local()) specific to each langauge that requires even more time to
think about or learn about. This is another reason to avoid threading - it is subtle beyond the reasoning of concurrecy.

Note: the syntax for this looks nearly identical to mp. Probably means both are very "interchangable" to some extent.

1. https://realpython.com/python-concurrency/



"""

import concurrent.futures
import requests
from requests.sessions import Session
import threading
import time

from typing import Iterator

# - Get the thread local object that helps manage locking and sharing of data.
# it looks a bit odd and like a global [1] but it really is used for each thread.
# It helps the user of the threading module not have to worry
# about data sharing, locks etc. I wasn't able to pass it as a function cleanly otherwise the .map function would have
# needed to get both .local & url and I decided to simply respect the original tutorial.
thread_local: threading.local = threading.local()

def get_session_for_each_thread() -> Session:
    """
    Each thread needs it own session object (probably because each session is not thread safe) so
    this code gets a session for each thread.

    Ref: https://github.com/psf/requests/issues/2766
    """
    if not hasattr(thread_local, "session"):
        thread_local.session: Session = requests.Session()
    return thread_local.session

def download_content_from_single_site(url: str) -> str:
    """
    Gets content as string form url and create a session for each thread.

    Example use is in a map function eg executor.map(download_content_from_single_site, sites)
    :param url:
    :return:
    """
    session: Session = get_session_for_each_thread()
    # - get content from url
    with session.get(url) as response:
        content_from_url: str = response.content
        print(f"Read {len(content_from_url)} from {url}")
        return content_from_url


def download_content_from_all_sites_using_threads(sites: list[str]) -> list[str]:
    """
    Get all downloads from sites using threads.

    The nice thing of using this is that even if we have a single cpu, we can overlap the data call for the io-device
    and thus overlap the io calls - speeding code up. Cost is that thinking about threading is usually hard the
    arbitrary details of sepcific code can often make it even harder to use and remember.
    """
    all_content_from_sites: list[str] = []

    # - create a pool of threads that donwloads all content concurrently/overlapingly (note not necesserily in parallel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        all_site_contents_from_threads: Iterator[str] = executor.map(download_content_from_single_site, sites)
        all_content_from_sites.extend(all_site_contents_from_threads)
    return all_content_from_sites


if __name__ == "__main__":
    # - args
    num_sites: int = 80
    sites: list[str] = ["https://www.jython.org", "http://olympus.realpython.org/dice"] * num_sites
    start_time: float = time.time()

    # - download content from sites
    all_content_from_sites: list[str] = download_content_from_all_sites_using_threads(sites)
    duration: float = time.time() - start_time

    # - print stats about the content download and duration
    print(f"Downloaded {len(all_content_from_sites)} in {duration} seconds")
    print('Success.\a')