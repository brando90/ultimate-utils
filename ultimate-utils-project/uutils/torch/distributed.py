"""
For code used in distributed training.
"""
import time
from argparse import Namespace
from typing import Tuple

import torch
import torch.distributed as dist

import os

from torch import Tensor, nn, optim

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import Dataset, DataLoader, DistributedSampler

def process_batch_ddp(opts, batch):
    """
    Make sure opts has the gpu for each worker.

    :param opts:
    :param batch:
    :return:
    """
    x, y = batch
    if type(x) == torch.Tensor:
        x = x.to(opts.gpu)
    if type(y) == torch.Tensor:
        y = y.to(opts.gpu)
    return x, y

def set_sharing_strategy(new_strategy=None):
    """
    https://pytorch.org/docs/stable/multiprocessing.html
    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    https://stackoverflow.com/questions/66426199/how-does-one-setup-the-set-sharing-strategy-strategy-for-multiprocessing-in-pyto
    """
    from sys import platform

    if new_strategy is not None:
        mp.set_sharing_strategy(new_strategy=new_strategy)
    else:
        if platform == 'darwin':  # OS X
            # only sharing strategy available at OS X
            mp.set_sharing_strategy('file_system')
        else:
            # ulimit -n 32767 or ulimit -n unlimited (perhaps later do try catch to execute this increase fd limit)
            mp.set_sharing_strategy('file_descriptor')

def use_file_system_sharing_strategy():
    """
    when to many file descriptor error happens

    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    """
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_process(rank, world_size, master_port, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.

    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_DISABLE=1

    https://stackoverflow.com/questions/61075390/about-pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8

    https://pytorch.org/docs/stable/distributed.html#common-environment-variables
    """
    import torch.distributed as dist
    import os
    import torch

    if rank != -1:  # -1 rank indicates serial code
        print(f'setting up rank={rank} (with world_size={world_size})')
        # MASTER_ADDR = 'localhost'
        MASTER_ADDR = '127.0.0.1'
        MASTER_PORT = master_port
        # set up the master's ip address so this child process can coordinate
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        print(f"{MASTER_ADDR=}")
        os.environ['MASTER_PORT'] = MASTER_PORT
        print(f"{MASTER_PORT}")

        # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        if torch.cuda.is_available():
            # unsure if this is really needed
            # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
            # os.environ['NCCL_IB_DISABLE'] = '1'
            backend = 'nccl'
        print(f'{backend=}')
        # Initializes the default distributed process group, and this will also initialize the distributed package.
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        print(f'--> done setting up rank={rank}')

def cleanup(rank):
    """ Destroy a given process group, and deinitialize the distributed package """
    # only destroy the process distributed group if the code is not running serially
    if rank != -1:  # -1 rank indicates serial code
        dist.destroy_process_group()

def get_batch(batch: Tuple[Tensor, Tensor], rank) -> Tuple[Tensor, Tensor]:
    x, y = batch
    if torch.cuda.is_available():
        x, y = x.to(rank), y.to(rank)
    else:
        # I don't think this is needed...
        # x, y = x.share_memory_(), y.share_memory_()
        pass
    return x, y

def is_running_serially(rank):
    """ is it running with a single serial process. """
    return rank == -1

def is_running_parallel(rank):
    """if it's not serial then it's parallel. """
    return not is_running_serially(rank)

def is_lead_worker(rank):
    """
    -1 = means serial code so main proc = lead worker = master
    0 = first rank is the lead worker (in charge of printing, logging, checkpoiniting etc.)
    :return:
    """
    am_I_lead_worker = rank == 0 or is_running_serially(rank)
    return am_I_lead_worker

def print_process_info(rank):
    """
    Prints the rank given, the current process obj name/info and the pid (according to os python lib).

    :param rank:
    :return:
    """
    print(f'{rank=}')
    print(f'{mp.current_process()=}')
    print(f'{os.getpid()=}')

def print_gpu_info():
    # get device name if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        print(f'\nopts.gpu_name = {torch.cuda.get_device_name(0)}\n')
    except:
        pass
    print(f'{device=}')

    # opts.PID = str(os.getpid())
    if torch.cuda.is_available():
        nccl = torch.cuda.nccl.version()
        print(f'{nccl=}')

def move_to_ddp(rank, opts, model):
    if is_running_parallel(rank):
        # model.criterion = self.opts.criterion.to(rank)  # I think its not needed since I already put it in the TP so when TP is moved to DDP the rank is moved automatically I hope
        # if gpu avail do the standard of creating a model and moving the model to the GPU with id rank
        if torch.cuda.is_available():
            # create model and move it to GPU with id rank
            tactic_predictor = model.to(opts.gpu)
            model = DistributedDataParallel(model, find_unused_parameters=True, device_ids=[opts.gpu])
        else:
            # if we want multiple cpu just make sure the model is shared properly accross the cpus with shared_memory()
            # note that op is a no op if it's already in shared_memory
            model = model.share_memory()
            model = DistributedDataParallel(model, find_unused_parameters=True)  # I think removing the devices ids should be fine...
    else:  # running serially
        if torch.cuda.is_available():
            # create model and move it to GPU with id rank
            model = model.to(opts.gpu)

    return model

def clean_end_with_sigsegv_hack(rank):
    """
    this is is just a hack to pause all processes that are not the lead worker i.e. rank=0.

    :return:
    """
    from datetime import time

    if is_running_parallel(rank):
        torch.distributed.barrier()
        if rank != 0:
            time.sleep(1)

# -- tests

def runfn_test(rank, world_size, master_port):
    setup_process(rank, world_size, master_port)
    cleanup(rank)

def test_setup():
    print('test_setup')
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 4
    master_port = find_free_port()
    mp.spawn(runfn_test, args=(world_size, master_port), nprocs=world_size)
    print('successful test_setup!')

class QuadraticDataset(Dataset):

    def __init__(self, Din, nb_examples=200):
        self.Din = Din
        self.x = torch.randn(nb_examples, self.Din)
        self.y = self.x**2 + self.x + 3

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dist_dataloader_test(rank, opts):
    train_dataset = QuadraticDataset(opts.Din)
    sampler = DistributedSampler(train_dataset, num_replicas=opts.world_size, rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=0,
        sampler=sampler,
        pin_memory=True)
    return train_loader

def run_parallel_training_loop(rank, opts):
    print_process_info(rank)
    opts.gpu = rank
    # You need to call torch.cuda.set_device(rank) before init_process_group is called.
    torch.cuda.set_device(opts.gpu)  # https://github.com/pytorch/pytorch/issues/54550
    setup_process(rank, opts.world_size, opts.master_port)

    # get ddp model
    opts.Din, opts.Dout = 10, 10
    model = nn.Linear(opts.Din, opts.Dout)
    model = move_to_ddp(rank, opts, model)
    criterion = nn.MSELoss().to(opts.gpu)

    # can distributed dataloader
    train_loader = get_dist_dataloader_test(rank, opts)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # do training
    for epoch in range(opts.epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
            optimizer.step()

    # Destroy a given process group, and deinitialize the distributed package
    cleanup(rank)

def test_basic_ddp_example():
    """
    Useful links:
    - https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
    - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

    """
    print('test_setup')
    opts = Namespace(epochs=3, batch_size=8)
    if torch.cuda.is_available():
        opts.world_size = torch.cuda.device_count()
    else:
        opts.world_size = 4
    opts.master_port = find_free_port()
    mp.spawn(run_parallel_training_loop, args=(opts,), nprocs=opts.world_size)

def test_basic_mnist_example():
    pass

if __name__ == '__main__':
    print('starting __main__')
    start = time.time()
    # test_setup()
    test_basic_ddp_example()
    print(f'execution length = {time.time() - start}')
    print('Done!\a\n')
