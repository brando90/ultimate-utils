import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.tensorboard import SummaryWriter

import uutils
from uutils.torch_uu import AverageMeter
from uutils.torch_uu.distributed import is_lead_worker, move_to_ddp_gpu_via_dict_mutation, print_process_info

from argparse import Namespace

import progressbar

from dataclasses import dataclass
from dataclasses import field

@dataclass
class SLAgent:
    args: Namespace
    mdl: nn.Module
    optimizer: torch.optim.Optimizer

    dataloaders: dict
    scheduler: torch.optim.lr_scheduler
    tb: SummaryWriter

    best_val_loss: float = float('inf')

    @property
    def mdl(self) -> torch.nn.Module:
        # checks if it's a ddp class and returns the model correctly
        return uutils.torch.get_model(self.mdl_)
#
# def forward_one_batch(agent, batch: dict, training: bool) -> tuple[Tensor, float]:
#     """
#     Forward pass for one batch.
#
#     :param batch:
#     :param training:
#     :return:
#     """
#     agent.mdl.train() if training else agent.mdl.eval()
#     batch = move_to_ddp_gpu_via_dict_mutation(agent.args, batch)
#     train_loss, logits = agent.mdl(batch)
#     train_acc = agent.accuracy(logits, ty_batch=batch['batch_target_tys'], y_batch=batch['batch_targets'])
#     return train_loss, train_acc
#
# def train_single_batch(agent):
#     """
#     Trains a single batch.
#
#     :return:
#     """
#     train_batch = next(iter(agent.dataloaders['train']))
#     val_batch = next(iter(agent.dataloaders['val']))
#     uutils.torch_uu.train_single_batch_agent(agent, train_batch, val_batch)
#

# - tests

def get_args():
    args = Namespace(rank=-1, world_size=1, merge=merge_x_termpath_y_ruleseq_one_term_per_type, ctx=ctx, batch_size=4)
    args.device = 'cpu'
    return args

def test1():
    args = get_args()
    # -- pass data through the model
    dataloaders = get_dataloaders(args, args.rank, args.world_size, args.merge)
    # -- model
    # -- agent
    agent = Agent(args, mdl)
    for batch_idx, batch in enumerate(dataloaders['train']):
        print(f'{batch_idx=}')
        x_batch, y_batch = batch
        y_pred = mdl(x_batch, y_batch)
        print(f'{x_batch=}')
        print(f'{y_batch=}')
        if batch_idx > 5:
            break

if __name__ == "__main__":
    test1()