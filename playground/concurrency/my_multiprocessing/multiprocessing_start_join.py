"""
multiprocessing is great for cpu-bound code.
"""
import time
from multiprocessing import Process


def cpu_bound(number: int = 5_000_000, delay: float = 2.0):
    time.sleep(delay)
    return sum(i * i for i in range(number))


def main(size: int = 10) -> int:
    processes = []
    for rank in range(size):
        # target is the function the (parallel) process will run with args
        p = Process(target=cpu_bound, args=(rank, ))
        p.start()  # start process
        # p.join()  # this would make the code synchronous!
        processes.append(p)

    # wait for all processes to finish by blocking one by one (this code could be problematic see spawn: https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses )
    for p in processes:
        p.join()  # blocks until p is done


if __name__ == "__main__":
    # numbers: list[int] = [5_000_000 + x for x in range(20)
    start_time = time.time()
    main()
    # print(f'{sum_n=}')
    duration = time.time() - start_time
    print(f"Duration {duration} seconds")
