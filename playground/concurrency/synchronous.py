"""
The great thing about this version of code is that, well, it’s easy. It was comparatively easy to write and debug.
It’s also more straight-forward to think about. There’s only one train of thought running through it, so you can predict
what the next step is and how it will behave [1].

If your program takes very little time and there is essentially no benefits of concurrency this is enough! Stop here. ;)

The sync code (synchronous.py) should have taken longer e.g. in one run the async file (asyncio_example.py) took:
Downloaded 160 sites in 0.4063692092895508 seconds
While the sync option took:
Downloaded 160 in 3.351937770843506 seconds

1. https://realpython.com/python-concurrency/
"""

import requests
from requests.sessions import Session
import time


def download_content_from_single_site(url: str, session: Session) -> str:
    """
    Gets content from url using requests package and returns it's content as a string.
    """
    # - get content from url
    with session.get(url) as response:
        content_from_url: str = response.content
        print(f"Read {len(content_from_url)} from {url}")
        return content_from_url


def download_content_from_all_sites(sites: list[str]) -> list[str]:
    """
    Return all content from the list of sites as a list of strings.
    """
    # - open a session with requests and then get all data from url
    all_content_from_sites: list[str] = []
    with requests.Session() as session:  # speeds things up vs only using get()
        for url in sites:
            content_from_url: str = download_content_from_single_site(url, session)
            all_content_from_sites.append(content_from_url)
    return all_content_from_sites


if __name__ == "__main__":
    # - args
    num_sites: int = 80
    sites: list[str] = ["https://www.jython.org", "http://olympus.realpython.org/dice"] * num_sites
    start_time: float = time.time()

    # - download content from sites
    all_content_from_sites: list[str] = download_content_from_all_sites(sites)
    duration: float = time.time() - start_time

    # - print stats about the content download and duration
    print(f"Downloaded {len(all_content_from_sites)} in {duration} seconds")
    print('Success.\a')
