#%%
"""
test a basic DDP example
"""
from argparse import Namespace

import torch
from torch import nn

import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from meta_learning.base_models.learner_from_opt_as_few_shot_paper import get_learner_from_args
from uutils.torch_uu import process_meta_batch
from uutils.torch_uu.dataloaders import get_distributed_dataloader_miniimagenet_torchmeta, get_args_for_mini_imagenet
from uutils.torch_uu.distributed import print_process_info, print_gpu_info, setup_process, move_model_to_ddp, \
    cleanup, find_free_port


def get_dist_dataloader_torch_meta_mini_imagenet(args) -> dict[str, DataLoader]:
    dataloaders: dict[str, DataLoader] = get_distributed_dataloader_miniimagenet_torchmeta(args)
    return dataloaders

def run_parallel_training_loop(rank: int, args: Namespace):
    """
    Run torchmeta examples with a distributed dataloader.

    This should distribute the following loop:
    for batch_idx, batch in enumerate(dataloader['train']):
        print(f'{batch_idx=}')
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
        print(f'Train inputs shape: {spt_x.size()}')  # (2, 25, 3, 28, 28)
        print(f'Train targets shape: {spt_y.size()}'.format(spt_y.shape))  # (2, 25)

        print(f'Test inputs shape: {qry_x.size()}')  # (2, 75, 3, 28, 28)
        print(f'Test targets shape: {qry_y.size()}')  # (2, 75)
        break

    Note:
        usual loop for ddp looks as follows:

    for i, batch in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        if rank == 0:
            print(f'{loss=}')

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
        optimizer.step()
    """
    print(f'-> started ps with {rank=}')
    args.rank = rank
    print_process_info(args.rank)
    print_gpu_info()
    args.gpu = rank
    setup_process(args, rank, master_port=args.master_port, world_size=args.world_size)

    # get ddp model
    print('about to create model')
    # args.Din, args.Dout = 10, 10
    # model = nn.Linear(args.Din, args.Dout)
    model = get_learner_from_args(args)
    model = move_model_to_ddp(rank, args, model)
    criterion = nn.CrossEntropyLoss().to(args.gpu)
    print('done creating ddp model')

    # can distributed dataloader
    print('about to create torch meta data loader')
    dataloaders: dict[str, DataLoader] = get_distributed_dataloader_miniimagenet_torchmeta(args)
    print('done created distributed data loaders')
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # do training
    print('about to train')
    for batch_idx, batch in enumerate(dataloaders['train']):
        print(f'{batch_idx=}')
        spt_x, spt_y, qry_x, qry_y = process_meta_batch(args, batch)
        outputs = model(spt_x)
        loss = criterion(outputs, spt_y)
        if rank == 0:
            print(f'{loss=}')

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # When the backward() returns, param.grad already contains the synchronized gradient tensor.
        optimizer.step()

    # Destroy a given process group, and deinitialize the distributed package
    cleanup(rank)

def hello(rank: int, args):
    print(f'hello {rank=}')

def ddp_example_torchmeta_dataloader_test():
    """
    Useful links:
    - https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
    - https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

    """
    print('test_basic_ddp_example')
    # args = Namespace(epochs=3, batch_size=8)
    args = get_args_for_mini_imagenet()
    if torch.cuda.is_available():
        args.world_size = torch.cuda.device_count()
    else:
        args.world_size = 4
    args.master_port = find_free_port()
    print('\nABOUT TO SPAWN WORKERS (via mp.spawn)')
    # mp.spawn(hello, args=(args,), nprocs=args.world_size)
    mp.spawn(run_parallel_training_loop, args=(args,), nprocs=args.world_size)
    print('mp.spawn finished\a')

if __name__ == '__main__':
    print('')
    ddp_example_torchmeta_dataloader_test()
    print('Done\a')