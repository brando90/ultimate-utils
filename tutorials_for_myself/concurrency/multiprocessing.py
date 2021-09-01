"""
Only multiprocessing runs things in parallel [1].


1. https://realpython.com/python-concurrency/
"""


# TODO - sort out the code bellow with functions + doc string saying what each example shows



# %%

"""
Goal: train in a mp way by computing each example in a seperate process.


tutorial: https://pytorch.org/docs/stable/notes/multiprocessing.html
full example: https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py

Things to figure out:
- fork or spwan for us? see pytorch but see this too https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.run
- shared memory
- do we need num_workers=0, 1 or 2? (one for main thread other for pre-fetching batches)
- run test and check that the 112 process do improve the time for a loop (add progress part for dataloder

docs: https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing

(original python mp, they are compatible: https://docs.python.org/3/library/multiprocessing.html)
"""

# def train(cpu_parallel=True):
#     num_cpus = get_num_cpus()  # 112 is the plan for intel's clsuter as an arparse or function
#     model.shared_memory()  # TODO do we need this?
#     # add progressbar for data loader to check if multiprocessing is helping
#     for batch_idx, batch in dataloader:
#         # do this mellow with pool when cpu_parallel=True
#         with Pool(num_cpus) as pool:
#             losses = pool.map(target=model.forward, args=batch)
#             loss = torch.sum(losses)
#             # now do .step as normal

# https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from torch.multiprocessing import Pool


class SimpleDataSet(Dataset):

    def __init__(self, Din, num_examples=23):
        self.x_dataset = [torch.randn(Din) for _ in range(num_examples)]
        # target function is x*x
        self.y_dataset = [x ** 2 for x in self.x_dataset]

    def __len__(self):
        return len(self.x_dataset)

    def __getitem__(self, idx):
        return self.x_dataset[idx], self.y_dataset[idx]


def get_loss(args):
    x, y, model = args
    y_pred = model(x)
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y)
    return loss


def get_dataloader(D, num_workers, batch_size):
    ds = SimpleDataSet(D)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return dl


def train_fake_data():
    num_workers = 2
    Din, Dout = 3, 1
    model = nn.Linear(Din, Dout).share_memory()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    batch_size = 2
    num_epochs = 10
    # num_batches = 5
    num_procs = 5
    dataloader = get_dataloader(Din, num_workers, batch_size)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(num_epochs):
        for _, batch in enumerate(dataloader):
            batch = [(torch.randn(Din), torch.randn(Dout), model) for _ in batch]
            with Pool(num_procs) as pool:
                optimizer.zero_grad()

                losses = pool.map(get_loss, batch)
                loss = torch.mean(losses)
                loss.backward()

                optimizer.step()
            # scheduler
            scheduler.step()


if __name__ == '__main__':
    # start = time.time()
    # train()
    train_fake_data()
    # print(f'execution time: {time.time() - start}')

# %%

"""
The distributed package included in PyTorch (i.e., torch.distributed) enables researchers and practitioners to
easily parallelize their computations across processes and clusters of machines.

As opposed to the multiprocessing (torch.multiprocessing) package, processes can use different communication backends
and are not restricted to being executed on the same machine.


https://pytorch.org/tutorials/intermediate/dist_tuto.html

"""
"""run.py:"""
# !/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size):
    """
    Distributed function to be implemented later.

    This is the function that is actually ran in each distributed process.
    """
    pass


def init_process_and_run_parallel_fun(rank, size, fn, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library for process to communicate/coordinate with each other/master. It's a backend library.
    """
    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # TODO: I think this is what makes sure that each process can talk to master,
    dist.init_process_group(backend, rank=rank, world_size=size)
    # run parallel function
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # target is the function the (parallel) process will run with args
        p = Process(target=init_process_and_run_parallel_fun, args=(rank, size, run))
        p.start()  # start process
        processes.append(p)

    # wait for all processes to finish by blocking one by one (this code could be problematic see spawn: https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses )
    for p in processes:
        p.join()  # blocks until p is done