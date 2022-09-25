"""
For code used in distributed training.
"""

import time
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Union, Callable, Any, Optional

import torch
import torch.distributed as dist

from torch import Tensor, nn, optim

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import Dataset, DataLoader, DistributedSampler

import os

from pdb import set_trace as st


def set_gpu_id_if_available_simple(args):
    """
    Main idea is args.gpu = rank for simple case except in debug/serially running.

    :param args:
    :return:
    """
    if torch.cuda.is_available():
        # if running serially then there is only 1 gpu the 0th one otherwise the rank is the gpu in simple cases
        args.gpu = 0 if is_running_serially(args.rank) else args.rank  # makes sure code works with 1 gpu and serially
    else:
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_devices(args):
    """
    Set device to the gpu id if its distributed pytorch parallel otherwise to the device available.

    :param args:
    :return:
    """
    if is_running_serially(args.rank):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = args.rank

    # todo - I'm not sure if the code bellow should be here...
    if is_running_parallel(args.rank):
        if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")) != 'cpu':
            torch.cuda.set_device(args.device)  # is this right if we do parallel cpu?


def set_devices_and_seed_ala_l2l(args: Namespace, seed: Optional[None] = None, cuda: bool = True) -> torch.device:
    """
    original code:

    print(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device_id = rank % torch.cuda.device_count()
        device = torch.device('cuda:' + str(device_id))
    print(rank, ':', device)
    """
    # - set device
    if is_running_serially(args.rank):
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device('cpu')
        if cuda and torch.cuda.device_count():
            device_id = args.rank % torch.cuda.device_count()
            args.device = torch.device('cuda:' + str(device_id))
        print(args.rank, ':', args.device)

    # - set seed
    import random
    import numpy as np

    seed: int = args.seed if seed is None else seed
    # rank: int = torch.distributed.get_rank()
    seed += args.rank
    random.seed(seed)
    np.random.seed(seed)
    if cuda and torch.cuda.device_count():
        torch.manual_seed(seed)
        # sebas code doesn't have this so I am leaving it out.
        # if is_running_parallel(args.rank):
        #     if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")) != 'cpu':
        #         torch.cuda.set_device(args.device)  # is this right if we do parallel cpu?


# torch.cuda.manual_seed(seed)


def process_batch_ddp(args: Namespace, batch: Any) -> tuple[Tensor, Tensor]:
    """
    Make sure args has the gpu for each worker.

    :param args:
    :param batch:
    :return:
    """
    x, y = batch
    if type(x) == torch.Tensor:
        # x = x.to(args.gpu)
        x = x.to(args.device)
    if type(y) == torch.Tensor:
        # y = y.to(args.gpu)
        y = y.to(args.device)
    return x, y


def process_batch_ddp_union_rfs(args: Namespace, batch: Any) -> tuple[Tensor, Tensor]:
    """
    Make sure args has the gpu for each worker.

    :param args:
    :param batch:
    :return:
    """
    if len(batch) == 3:
        x, y, _ = batch
    elif len(batch) == 2:
        x, y = batch
    else:
        # img, target, item, sample_idx = batch
        x, y, item, sample_idx = batch
        raise NotImplementedError
    x = x.to(args.device)
    y = y.to(args.device)
    return x, y


def move_to_ddp_gpu_via_dict_mutation(args: Namespace, batch: dict) -> dict:
    # temporary fix for backwards compatibility
    return move_model_to_ddp_gpu_via_dict_mutation(args, batch)


def move_model_to_ddp_gpu_via_dict_mutation(args: Namespace, batch: dict) -> dict:
    """
    Mutates the data batch and returns the mutated version.
    Note that the batch is assumed to have the specific different types of data in
    different batches according to the name of that type.
    e.g.
        batch = {'x': torch_uu.randn([B, T, D]), 'y': torch_uu.randn([B, T, V])}
    holds two batches with names 'x' and 'y' that are tensors.
    In general the dict format for batches is useful because any type of data can be added to help
    with faster prototyping. The key is that they are not tuples (x,y) or anything like that
    since you might want to return anything and batch it. e.g. for each example return the
    adjacency matrix, or the depth embedding etc.

    :param args:
    :param batch:
    :return:
    """
    for data_name, batch_data in batch.items():
        if isinstance(batch_data, torch.Tensor):
            batch[data_name] = batch_data.to(args.gpu)
    return batch


def process_batch_ddp_tactic_prediction(args, batch):
    """
    Make sure args has the gpu for each worker.

    :param args:
    :param batch:
    :return:
    """
    processed_batch = {'goal': [], 'local_context': [], 'env': [], 'tac_label': []}
    if type(batch) is dict:
        y = torch.tensor(batch['tac_label'], dtype=torch.long).to(args.gpu)
        batch['tac_label'] = y
        processed_batch = batch
    else:
        # when treating entire goal, lc, env as 1 AST/ABT
        x, y = batch
        if type(x) == torch.Tensor:
            x = x.to(args.device)
        if type(y) == torch.Tensor:
            y = y.to(args.device)
        processed_batch['goal'] = x
        processed_batch['tac_label'] = y
    return processed_batch


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


def setup_process(args, rank, world_size, master_port, init_method=None, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.

    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_DISABLE=1

    init_method:
    In your training program, you are supposed to call the following function at the beginning to start
    the distributed backend. It is strongly recommended that
        init_method=env://
    Other init methods (e.g. tcp://) may
    work, but env:// is the one that is officially supported by this module. https://pytorch.org/docs/stable/distributed.html#launch-utility


    https://stackoverflow.com/questions/61075390/about-pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8

    https://pytorch.org/docs/stable/distributed.html#common-environment-variables
    """
    import torch.distributed as dist
    import os
    import torch

    if is_running_parallel(rank):
        print(f'----> setting up rank={rank} (with world_size={world_size})')
        # MASTER_ADDR = 'localhost'
        MASTER_ADDR = '127.0.0.1'
        MASTER_PORT = master_port
        # set up the master's ip address so this child process can coordinate
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        print(f"---> {MASTER_ADDR=}")
        os.environ['MASTER_PORT'] = MASTER_PORT
        print(f"---> {MASTER_PORT=}")

        # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        if torch.cuda.is_available():
            backend = 'nccl'
            # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called. https://github.com/pytorch/pytorch/issues/54550
            torch.cuda.set_device(
                args.device)  # is this right if we do parallel cpu? # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called. https://github.com/pytorch/pytorch/issues/54550
        print(f'---> {backend=}')

        # Initializes the default distributed process group, and this will also initialize the distributed package.
        print(f'About to call: init_process_group('
              f'{backend}, {init_method=}, {rank=}, {world_size=})'
              f'')
        dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
        print(f'----> done setting up rank={rank}')
        torch.distributed.barrier()


def init_process_group_l2l(args, local_rank, world_size, init_method=None, backend='gloo'):
    """
    based on

    WORLD_SIZE = 2

    import os
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f'{local_rank=}\n')

    torch.distributed.init_process_group(
        'gloo',
        init_method=None,
        rank=local_rank,
        world_size=WORLD_SIZE,
    )

    rank = torch.distributed.get_rank()
    print(f'{rank=}\n')
    """
    if is_running_parallel(local_rank):
        if torch.cuda.is_available():
            # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called. https://github.com/pytorch/pytorch/issues/54550
            backend = 'nccl'
        torch.distributed.init_process_group(
            backend,
            init_method=init_method,
            rank=local_rank,
            world_size=world_size,
        )
        # torch.distributed.init_process_group(
        #     backend,
        #     init_method=None,
        #     rank=local_rank,
        #     world_size=world_size,
        # )
        # dist.init_process_group(backend=backend, init_method="env://")
        # dist.init_process_group(backend=backend, init_method=None)
        # torch.distributed.barrier()  # causes this warning: https://github.com/pytorch/pytorch/issues/60752


def cleanup(rank):
    """ Destroy a given process group, and deinitialize the distributed package """
    # only destroy the process distributed group if the code is not running serially
    if is_running_parallel(rank):
        torch.distributed.barrier()
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


def is_lead_worker(rank: int) -> bool:
    """
    Returns true if the current process is the lead worker.

    -1 = means serial code so main proc = lead worker = master
    0 = first rank is the lead worker (in charge of printing, logging, checkpoiniting etc.)
    :return:
    """
    am_I_lead_worker: bool = rank == 0 or is_running_serially(rank)
    return am_I_lead_worker


def print_process_info(rank, flush=False):
    """
    Prints the rank given, the current process obj name/info and the pid (according to os python lib).

    :param flush:
    :param rank:
    :return:
    """
    # import sys
    # sys.stdout.flush()  # no delay in print statements
    print(f'-> {rank=}', flush=flush)
    print(f'-> {mp.current_process()=}', flush=flush)
    print(f'-> {os.getpid()=}', flush=flush)


def print_gpu_info():
    # get device name if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        print(f'\ngpu_name = {torch.cuda.get_device_name(0)}\n')
    except:
        pass
    print(f'{device=}')

    # args.PID = str(os.getpid())
    if torch.cuda.is_available():
        nccl = torch.cuda.nccl.version()
        print(f'{nccl=}')


def move_model_to_ddp(rank: int, args: Namespace, model: nn.Module, force: bool = False):
    """
    Moves the model to a ddp object.

    Note:
        - the dataloader should be a distributed data loader when it's created,
        so this is only to move the model.

    :param rank:
    :param args:
    :param model:
    :param force: force is meant to force it into DDP. Meant for debugging.
    :return:
    """
    if is_running_parallel(rank) or force:
        # model.criterion = self.args.criterion.to(rank)  # I think its not needed since I already put it in the TP so when TP is moved to DDP the rank is moved automatically I hope
        # if gpu avail do the standard of creating a model and moving the model to the GPU with id rank
        if torch.cuda.is_available():
            # create model and move it to GPU with id rank
            model = model.to(args.device)
            model = DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.device])
        else:
            # if we want multiple cpu just make sure the model is shared properly accross the cpus with shared_memory()
            # note that op is a no op if it's already in shared_memory
            model = model.share_memory()
            model = DistributedDataParallel(model,
                                            find_unused_parameters=True)  # I think removing the devices ids should be fine...
    else:  # running serially
        model = model.to(args.device)
    return model


def move_model_to_dist_device_or_serial_device(rank: int, args: Namespace, model: nn.Module, force: bool = False):
    if not hasattr(args, 'dist_option'):
        # for backwards compatibility, default try to do ddp or just serial to device
        model = move_model_to_ddp(rank, args, model, force)
    else:
        if args.dist_option == 'ddp':
            model = move_model_to_ddp(rank, args, model, force)
        elif args.dist_option == 'l2l_dist':
            # based on https://github.com/learnables/learn2learn/issues/263
            model = model.to(args.device)  # this works for serial and parallel (my guess, just moves to proc's device)
        elif is_running_serially(args.rank):
            model = move_model_to_ddp(rank, args, model, force)
        else:
            raise ValueError(f'Not a valid way to move a model to (dist) device: {args.dist_option=}')
    return model


# for backwards compatibility
move_to_ddp = move_model_to_ddp


def move_opt_to_cherry_opt_and_sync_params(args: Namespace, syn: int = 1):
    if is_running_parallel(args.rank):
        import cherry
        args.opt = cherry.optim.Distributed(args.model.parameters(), opt=args.opt, sync=syn)
        args.opt.sync_parameters()
    return args.opt


def create_distributed_data_loader_from_datasets(args: Namespace,
                                                 rank: int,
                                                 world_size: int, merge: Callable,
                                                 datasets: dict[str, Dataset],
                                                 data_loader_constructor: Callable) -> DataLoader:
    """
    Given a list of data sets, creates the distributed dataloader by having a distributed sampler.
    Since creating a new data set has it's own interface we let the user get those themselves (unfortuantely).

    note:
        - data_loader_constructor: usual values:
            - DataLoader
    """
    from torch.utils.data import DistributedSampler

    # - get dist samplers
    assert (args.batch_size >= world_size)
    train_sampler = DistributedSampler(datasets['train'], num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(datasets['val'], num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(datasets['test'], num_replicas=world_size, rank=rank)
    args.num_workers = 0  # todo - zero for distributed data loaders for now. Figure out if it's ok.

    # - get dist dataloaders
    train_dataloader = data_loader_constructor(datasets['train'],
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               collate_fn=merge,
                                               num_workers=args.num_workers)
    val_dataloader = data_loader_constructor(datasets['val'],
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             collate_fn=merge,
                                             num_workers=args.num_workers)
    test_dataloader = data_loader_constructor(datasets['test'],
                                              batch_size=args.batch_size,
                                              sampler=test_sampler,
                                              collate_fn=merge,
                                              num_workers=args.num_workers)

    # - return distributed dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataloaders


def create_distributed_dataloaders_from_torchmeta_datasets(args: Namespace,
                                                           rank: int,
                                                           world_size: int,
                                                           datasets: dict[str, Dataset]) -> dict[str, DataLoader]:
    """
    Given a list of data sets, creates the distributed dataloader by having a distributed sampler.
    Since creating a new data set has it's own interface we let the user get those themselves (unfortuantely).

    note:
        - data_loader_constructor: usual values:
            - DataLoader
    """
    from torchmeta.utils.data import BatchMetaDataLoader
    from torch.utils.data import DistributedSampler

    # - get dist samplers
    assert (args.batch_size >= world_size)
    train_sampler = DistributedSampler(datasets['train'], num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(datasets['val'], num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(datasets['test'], num_replicas=world_size, rank=rank)
    args.num_workers = 0  # todo - zero for distributed data loaders for now. Figure out if it's ok.

    # - get dist dataloaders
    train_dataloader = BatchMetaDataLoader(datasets['train'],
                                           batch_size=args.meta_batch_size_train,
                                           sampler=train_sampler,
                                           num_workers=args.num_workers)
    val_dataloader = BatchMetaDataLoader(datasets['val'],
                                         batch_size=args.meta_batch_size_eval,
                                         sampler=val_sampler,
                                         num_workers=args.num_workers)
    test_dataloader = BatchMetaDataLoader(datasets['test'],
                                          batch_size=args.meta_batch_size_eval,
                                          sampler=test_sampler,
                                          num_workers=args.num_workers)

    # - return distributed dataloaders
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    return dataloaders


def clean_end_with_sigsegv_hack(rank):
    """
    this is is just a hack to pause all processes that are not the lead worker i.e. rank=0.

    :return:
    """
    import time

    if is_running_parallel(rank):
        torch.distributed.barrier()
        if rank != 0:
            time.sleep(1)


def dist_log(msg: str, rank, flush: bool = False, use_pprint: bool = False):
    """Prints only if the current process is the leader (e.g. rank = -1)."""
    #         torch.distributed.barrier() # could be used to sync prints?
    if is_lead_worker(rank):
        if not use_pprint:
            print(msg, flush=flush)
        else:
            from pprint import pprint
            pprint(msg)


def print_dist(msg: str, rank: int, flush: bool = False):
    dist_log(msg, rank, flush)


def pprint_dist(msg: str, rank, flush: bool = False):
    dist_log(msg, rank, flush, use_pprint=True)


def get_model_from_ddp(mdl: Union[nn.Module, DistributedDataParallel]) -> nn.Module:
    """Gets model from a ddp pytorch class without errors."""
    if isinstance(mdl, DistributedDataParallel):
        return mdl.module
    else:
        return mdl


def get_local_rank() -> int:
    try:
        local_rank: int = int(os.environ["LOCAL_RANK"])
    except:
        local_rank: int = -1
    return local_rank


def is_main_process_using_local_rank() -> bool:
    """
    Determines if it's the main process using the local rank.

    based on print statements:
        local_rank=0
        local_rank=1

    other ref:
        # - set up processes a la l2l
        local_rank: int = get_local_rank()
        print(f'{local_rank=}')
        init_process_group_l2l(args, local_rank=local_rank, world_size=args.world_size, init_method=args.init_method)
        rank: int = torch.distributed.get_rank() if is_running_parallel(local_rank) else -1
        args.rank = rank  # have each process save the rank
        set_devices_and_seed_ala_l2l(args)  # args.device = rank or .device
        print(f'setup process done for rank={args.rank}')
    """
    local_rank: int = get_local_rank()
    return local_rank == -1 or local_rank == 0  # -1 means serial, 0 likely means parallel


# -- tests

# - hello world parallel test (no ddp)

def runfn_test(rank, args, world_size, master_port):
    args.gpu = rank
    setup_process(rank, world_size, master_port)
    if rank == 0:
        print(f'hello world from: {rank=}')
    cleanup(rank)


def hello_world_test():
    print('hello_world_test')
    args = Namespace()
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 4
    args.master_port = find_free_port()
    mp.spawn(runfn_test, args=(args, world_size, args.master_port), nprocs=world_size)
    print('successful test_setup!')


# - real test

class QuadraticDataset(Dataset):

    def __init__(self, Din, nb_examples=200):
        self.Din = Din
        self.x = torch.randn(nb_examples, self.Din)
        self.y = self.x ** 2 + self.x + 3

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dist_dataloader_test(rank, args):
    train_dataset = QuadraticDataset(args.Din)
    sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        sampler=sampler,
        pin_memory=True)
    return train_loader


def run_parallel_training_loop(rank, args):
    print_process_info(rank)
    print_gpu_info()
    args.gpu = rank
    # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called.
    torch.cuda.set_device(args.gpu)  # https://github.com/pytorch/pytorch/issues/54550
    setup_process(args, rank, args.world_size, args.master_port)

    # get ddp model
    args.Din, args.Dout = 10, 10
    model = nn.Linear(args.Din, args.Dout)
    model = move_model_to_ddp(rank, args, model)
    criterion = nn.MSELoss().to(args.gpu)

    # can distributed dataloader
    train_loader = get_dist_dataloader_test(rank, args)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # do training
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            if rank == 0:
                print(f'{loss=}')

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
            optimizer.step()

    # Destroy a given process group, and deinitialize the distributed package
    cleanup(rank)


def ddp_example_sythetic_data_test():
    """
    Useful links:
    - https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
    - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

    """
    print('test_basic_ddp_example')
    args = Namespace(epochs=3, batch_size=8)
    if torch.cuda.is_available():
        args.world_size = torch.cuda.device_count()
    else:
        args.world_size = 4
    args.master_port = find_free_port()
    print('about to run mp.spawn---')
    mp.spawn(run_parallel_training_loop, args=(args,), nprocs=args.world_size)


# - tb example (probably not needed due to wandb)

class TestDistAgent:
    def __init__(self, args, model, criterion, dataloader, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.dataloader = dataloader
        if is_lead_worker(self.args.rank):
            from torch.utils.tensorboard import SummaryWriter  # I don't think this works
            args.tb_dir = Path('~/ultimate-utils/').expanduser()
            self.args.tb = SummaryWriter(log_dir=args.tb_dir)

    def train(self, n_epoch):
        for i, (images, labels) in enumerate(self.dataloader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.log(f'{i=}: {loss=}')

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
            self.optimizer.step()
        return loss.item()

    def log(self, string):
        """ logs only if you are rank 0"""
        if is_lead_worker(self.args.rank):
            # print(string + f' rank{self.args.rank}')
            print(string)

    def log_tb(self, it, tag1, loss):
        if is_lead_worker(self.args.rank):
            self.args.tb.add_scalar(it, tag1, loss)


def run_parallel_training_loop_with_tb(rank, args):
    print_process_info(rank)
    print_gpu_info()
    args.rank = rank
    args.gpu = rank
    # You need to call torch_uu.cuda.set_device(rank) before init_process_group is called.
    setup_process(args, rank, args.world_size, args.master_port)

    # get ddp model
    args.Din, args.Dout = 10, 10
    model = nn.Linear(args.Din, args.Dout)
    model = move_model_to_ddp(rank, args, model)
    criterion = nn.MSELoss().to(args.gpu)

    # can distributed dataloader
    dataloader = get_dist_dataloader_test(rank, args)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # do training
    agent = TestDistAgent(args, model, criterion, dataloader, optimizer)
    for n_epoch in range(args.epochs):
        agent.log(f'\n{n_epoch=}')

        # training
        train_loss, train_acc = agent.train(n_epoch)
        agent.log(f'{n_epoch=}: {train_loss=}')
        agent.log_tb(it=n_epoch, tag1='train_loss', loss=train_loss)

    # Destroy a given process group, and deinitialize the distributed package
    cleanup(rank)


def basic_ddp_example_with_tensorboard_test():
    """
    Useful links:
    - https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
    - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

    """
    print('test_basic_ddp_example_with_tensorboard')
    args = Namespace(epochs=3, batch_size=8)
    if torch.cuda.is_available():
        args.world_size = torch.cuda.device_count()
    else:
        args.world_size = 4
    args.master_port = find_free_port()
    print('about to run mp.spawn---')

    # self.args.tb = SummaryWriter(log_dir=args.tb_dir)
    mp.spawn(run_parallel_training_loop_with_tb, args=(args,), nprocs=args.world_size)


def basic_mnist_example_test():
    pass


if __name__ == '__main__':
    print('starting distributed.__main__')
    start = time.time()
    # hello_world_test()
    ddp_example_sythetic_data_test()
    # test_basic_ddp_example_with_tensorboard()
    print(f'execution length = {time.time() - start} seconds')
    print('Done Distributed!\a\n')
