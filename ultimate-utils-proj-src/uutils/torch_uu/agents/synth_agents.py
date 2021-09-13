import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch.utils.tensorboard import SummaryWriter

from torchtext.vocab import vocab

import math

from lark import Lark

from progressbar import ProgressBar

import parsers

from uutils import make_args_pickable, set_system_wide_force_flush
from uutils.torch_uu import AverageMeter, accuracy, print_dataloaders_info
from uutils.torch_uu.distributed import is_running_serially, is_lead_worker, print_process_info, process_batch_ddp
from uutils.torch_uu.distributed import move_to_ddp_gpu_via_dict_mutation

import gc

import uutils

from argparse import Namespace

from typing import Tuple

from pprint import pprint

import progressbar

from pdb import set_trace as st

Batch = list
Seq = list

def process_batch(args, x_batch, y_batch):
    return x_batch, y_batch

class SynthAgent:

    def __init__(self, args: Namespace, mdl: nn.Module, optimizer, dataloaders, scheduler):
        self.args = args
        self.mdl_ = mdl
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.best_val_loss = float('inf')
        if is_lead_worker(self.args.rank):
            self.tb = SummaryWriter(log_dir=args.tb_dir)

    @property
    def target_parser(self):
        return self.dataloaders['train'].dataset.extended_ty_parser

    @property
    def mdl(self) -> torch.nn.Module:
        return uutils.torch.get_model(self.mdl_)

    def forward_one_batch(self, batch: dict, training: bool) -> Tuple[Tensor, float]:
        """
        Forward pass for one batch.

        Note that the batch is assumed to have the specific different types of data in
        different batches according to the name of that type.
        e.g.
            batch = {'x': torch_uu.randn([B, T, D]), 'y': torch_uu.randn([B, T, V])}
        holds two batches with names 'x' and 'y' that are tensors.
        In general the dict format for batches is useful because any type of data can be added to help
        with faster prototyping. The key is that they are not tuples (x,y) or anything like that
        since you might want to return anything and batch it. e.g. for each example return the
        adjacency matrix, or the depth embedding etc.
        :param batch:
        :param training:
        :return:
        """
        self.mdl.train() if training else self.mdl.eval()
        batch = move_to_ddp_gpu_via_dict_mutation(self.args, batch)
        train_loss, logits = self.mdl(batch)
        train_acc = self.accuracy(logits, ty_batch=batch['batch_target_tys'], y_batch=batch['batch_targets'])
        return train_loss, train_acc

    def train_single_batch(self):
        """
        Trains a single batch.

        :return:
        """
        train_batch = next(iter(self.dataloaders['train']))
        val_batch = next(iter(self.dataloaders['val']))
        uutils.torch.train_single_batch_agent(self, train_batch, val_batch)

    def train_one_epoch(self, epoch_num):
        """
        Train model for one epoch - i.e. through the entire data set once.

        :param epoch_num:
        :return:
        """
        # avg_loss = AverageMeter('train loss')
        # avg_acc = AverageMeter('train accuracy')
        #
        # bar = ProgressBar(max_value=len(self.dataloaders['train']))
        # for i, batch in enumerate(self.dataloaders['train']):
        #     self.mdl.train()
        #     x_batch, y_batch, ty_y_batch = batch['batch_inputs'], batch['batch_targets'], batch['batch_target_tys']
        #     x_batch, y_batch = process_batch_ddp(self.args, [x_batch, y_batch])
        #     train_loss, logits = self.mdl(x_batch, y_batch)
        #     train_acc = self.accuracy(logits, ty_batch, y_batch)
        #     avg_loss.update(train_loss.item(), self.args.batch_size)
        #     avg_acc.update(train_acc, self.args.batch_size)
        #
        #     self.optimizer.zero_grad()
        #     train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        #     self.optimizer.step()  # the right update is done since all procs have the right synced grads
        #
        #     if self.is_lead_worker() and i % 100 == 0:
        #         self.log(f'{i=}, {avg_loss.item()=}, {avg_acc.item()=}')
        #         bar.update(i) if self.is_lead_worker() else None
        #     gc.collect()
        #
        # return avg_loss.item(), avg_acc.item()
        pass

    def main_train_loop_until_convergence(self, args: Namespace, start_it: int, acc_tolerance: float = 1.0,
                                          train_loss_tolerance: float = 0.001):
        """
        Trains model based on iterations not epochs.

        :param args:
        :param start_epoch:
        :return:
        """
        self.log('Starting training...')

        bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
        self.args.it = 0
        train_loss, train_acc = float('inf'), 0.0
        len_train: int = len(self.dataloaders['train'])
        while True:
            # -- train for one epoch
            avg_loss = AverageMeter('train loss')
            avg_acc = AverageMeter('train accuracy')
            for i, batch in enumerate(self.dataloaders['train']):
                train_loss, train_acc = self.forward_one_batch(batch, training=True)
                avg_loss.update(train_loss.item(), self.args.batch_size)
                avg_acc.update(train_acc, self.args.batch_size)

                self.optimizer.zero_grad()
                train_loss.backward()  # each process synchronizes it's gradients in the backward pass
                self.optimizer.step()  # the right update is done since all procs have the right synced grads

                # -- validation (mainly for the scheduler)
                # if args.validation:
                #     val_loss, val_acc = self.valid(epoch_num)

                # -- annealing/decaying the learning rate
                # if (self.args.it % 2000 == 0 and self.args.it != 0 and self.scheduler is not None) or args.debug:  # call scheduler every
                # if (self.args.it % math.ceil(2.0 * len_train) == 0 and self.args.it != 0 and self.scheduler is not None) or args.debug:  # call scheduler every
                #     self.scheduler.step()
                self.scheduler.step()

                # -- log it stats
                if self.args.it % 10 == 0 and self.is_lead_worker():
                    self.log_train_stats(self.args.it, avg_loss.item(), avg_acc.item(), save_val_ckpt=True)
                    bar_it.update(self.args.it) if self.is_lead_worker() else None
                    if args.it % len_train == 0:
                        self.save(self.args.it)

                # -- compute halting condition (if not training for total its then check if converged, else continue training until number its has been met)
                halt: bool = avg_acc.item() >= acc_tolerance or avg_loss.item() <= train_loss_tolerance if args.num_its == -1 else self.args.it >= args.num_its
                # -- halt if halting condition met
                if halt or args.debug:
                    if self.is_lead_worker():
                        self.log_train_stats(self.args.it, avg_loss.item(), avg_acc.item(), save_val_ckpt=True, val_iterations=5)
                        self.save(self.args.it)  # very slow, save infrequently e.g. every epoch, x its, end, etc.
                        bar_it.update(self.args.it) if self.is_lead_worker() else None
                    # return train_loss, train_acc
                    return avg_loss.item(), avg_acc.item()
                self.args.it += 1
                # gc.collect()
        # return train_loss, train_acc
        return avg_loss.item(), avg_acc.item()

    def main_train_loop_based_on_fixed_number_of_epochs(self, args, start_epoch):
        """
        Trains model one epoch at a time - i.e. it's epochs based rather than iteration based.

        :param args:
        :param start_epoch:
        :return:
        """
        self.log('Starting training...')

        # bar_epoch = uutils.get_good_progressbar(max_value=args.num_epochs)
        # bar_it = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
        bar_it = uutils.get_good_progressbar(max_value=args.num_epochs * len(self.dataloaders['train']))
        self.args.it = 0
        train_loss, train_acc = float('inf'), 0.0
        len_train: int = len(self.dataloaders['train'])
        # for epoch_num in bar_epoch(range(start_epoch, start_epoch + args.num_epochs)):
        for epoch_num in range(start_epoch, start_epoch + args.num_epochs):
            args.epoch_num = epoch_num
            # -- train for one epoch
            avg_loss = AverageMeter('train loss')
            avg_acc = AverageMeter('train accuracy')
            for i, batch in enumerate(self.dataloaders['train']):
                train_loss, train_acc = self.forward_one_batch(batch, training=True)
                avg_loss.update(train_loss.item(), self.args.batch_size)
                avg_acc.update(train_acc, self.args.batch_size)

                self.optimizer.zero_grad()
                train_loss.backward()  # each process synchronizes it's gradients in the backward pass
                self.optimizer.step()  # the right update is done since all procs have the right synced grads

                # -- validation (mainly for the scheduler)
                # if args.validation:
                #     val_loss, val_acc = self.valid(epoch_num)

                # -- annealing/decaying the learning rate
                # if (self.args.it % 2000 == 0 and self.args.it != 0 and self.scheduler is not None) or args.debug:  # call scheduler every
                # if (self.args.it % math.ceil(2.0 * len_train) == 0 and self.args.it != 0 and self.scheduler is not None) or args.debug:  # call scheduler every
                #     self.scheduler.step()
                self.scheduler.step()

                # -- log it stats
                if self.args.it % 10 == 0 and self.is_lead_worker():
                    self.log_train_stats(self.args.it, avg_loss.item(), avg_acc.item(), save_val_ckpt=True)
                    bar_it.update(self.args.it) if self.is_lead_worker() else None
                    if args.it % len_train == 0:
                        self.save(self.args.it)
                self.args.it += 1
                # gc.collect()
                if self.args.debug:  # if debug break from entire training, the logging is done at the end, differs from it
                    break

            # -- log full epoch stats
            # if self.is_lead_worker():
            #     self.log_train_stats(self.args.epoch_n, avg_loss.item(), avg_acc.item(), save_val_ckpt=True)
            #     # bar_epoch.update(self.args.epoch_num) if self.is_lead_worker() else None
            #     if epoch_num % 1 == 0:  # save every epoch
            #         self.save(self.args.epoch_num)
            if self.args.debug:  # if debug break from entire training, the logging is done at the end, differs from it
                break

        # - log/save stop when we halt
        if self.is_lead_worker():
            self.log_train_stats(self.args.it, avg_loss.item(), avg_acc.item(), save_val_ckpt=True, val_iterations=5)
            self.save(self.args.it)  # very slow, save infrequently e.g. every epoch, x its, end, etc.
            # self.save(self.args.epoch_n)  # very slow, save infrequently e.g. every epoch, x its, end, etc.
        return avg_loss.item(), avg_acc.item()

    def main_train_loop_based_on_fixed_iterations(self, args: Namespace, start_it: int, acc_tolerance: float = 1.0, train_loss_tolerance: float = 0.001):
        """
        Trains model based on a fixed number of iterations (not epochs).

        :param args:
        :param start_epoch:
        :return:
        """
        avg_loss, avg_acc = self.main_train_loop_until_convergence(args, start_it, acc_tolerance, train_loss_tolerance)
        return avg_loss, avg_acc

    def valid(self, epoch_num: int, val_iterations: int = 0, val_ckpt: bool = True):
        avg_loss = AverageMeter('val loss')
        avg_acc = AverageMeter('val accuracy')

        for i, batch in enumerate(self.dataloaders['val']):
            self.mdl.eval()
            train_loss, train_acc = self.forward_one_batch(batch, training=False)
            avg_loss.update(train_loss.item(), self.args.batch_size)
            avg_acc.update(train_acc, self.args.batch_size)

            gc.collect()
            if i >= val_iterations:
                break
            if self.args.debug:
                break

        if avg_loss.item() < self.best_val_loss and val_ckpt:
            self.best_val_loss = avg_loss.item()
            self._save(epoch_num, ckpt_filename='mdl_best_val.pt')
        return avg_loss.item(), avg_acc.item()

    def evaluate(self):
        pass

    def accuracy(self, y_logits: Tensor, ty_batch: Batch[Type], y_batch: Batch[Seq[Token]]) -> int:
        """
        Calculate the accuracy of a batch of predicted types vs the real types.
        [B, R] -> [B, types]

        Algorithm:
            1. given a y_pred [B, T, R] -> rule_seq_pred [B, T]
                - do greddy, choose most likely rule
            2. rule_seq_pred [B, T] -> ty/term_pred [B]
                - cst -> term, if any term fails to reconstruct an "empty term"
            3. compare for each term in batch acc = y_pred[b].typeof(ctx) == y_pred.type(ctx) / B
        :return:
        """
        y = self.mdl.batch_target2index_tensor(y_batch)
        assert y_batch[0][0] == '<sos>' and y_batch[0][-1] == '<eos>'
        tys_pred, ys_pred = self.synthesize_terms_greedy(y_logits, y)
        assert len(tys_pred) == len(ty_batch)
        # print(f"{tys_pred=}")
        # print(f"{ty_batch=}")
        # print(f'{y=}\n{ys_pred}')
        correct = 0.0
        for i, (ty_pred, ty_truth) in enumerate(zip(tys_pred, ty_batch)):
            # todo - probably need to fix how terms are generated
            # if str(ty_pred) == str(ty_truth) or ty_pred == ty_truth:
            # if str(ty_pred) == str(ty_truth) or ty_pred == ty_truth or all(ys_pred[i] == y[i]):
            if str(ty_pred) == str(ty_truth) or ty_pred == ty_truth or self.equal(ys_pred[i], y[i]):
                correct += 1
        acc = correct / len(ty_batch)
        return acc

    def synthesize_terms_greedy(self, y_logits: Tensor, y: Tensor = None, maxk: int = 1) -> Batch[Type]:
        """
        Algorithm details:
        - if no prediction of eos let's see if we can reconstruct the a type anyway.

        :param y_logits:
        :param maxk:
        :return:
        """
        _, y_pred = y_logits.topk(k=maxk, dim=2)  # [B, T, V] -> [B, T, 1]
        y_pred = y_pred.squeeze()
        ys_pred = [rule_seq_pred for rule_seq_pred in y_pred]
        # - trim the predictions up to EOS
        trimmed_rules: list[Seq[RuleIdx]] = []
        for i in range(len(y_pred)):
            rule_seq_pred: Tensor = ys_pred[i]  # Tensor[RuleIdx]
            # - preprocess prediction
            # get the index of the first occurence of the eos, if 0 then return an error term immediately
            eos_idx = uutils.torch.index(rule_seq_pred, self.mdl.vocab_ty['<eos>'])
            eos_idx = len(rule_seq_pred) if eos_idx == -1 else eos_idx
            rule_seq_pred_trimmed: Seq[RuleIdx] = rule_seq_pred[
                                                  :eos_idx].tolist()  # trim the tensor to that length and reconstruct term if success
            trimmed_rules.append(rule_seq_pred_trimmed)
        # - do reconstruction for trimmed rule sequences
        tys_pred: list[Type] = []
        for i in range(len(trimmed_rules)):
            rule_seq_pred_trimmed: Seq[RuleIdx] = trimmed_rules[i]
            rule_seq_pred_corrected_trimmed = [rule_idx - 4 for rule_idx in rule_seq_pred_trimmed]
            # - try to make a term
            try:
                cst = rule_seq2lark_cst(rule_seq_pred_corrected_trimmed, self.target_parser)
                ty = lark_cst2ty(cst)
            except Exception as e:
                ty = ErrorType(str(e))
            # append predicted term
            tys_pred.append(ty)
        return tys_pred, ys_pred

    def eq_accuracy_via_seq(self, seq_pred: Tensor, seq_truth: Tensor) -> bool:
        # return all(ys_pred[i] == y[i])
        return all(seq_pred == seq_truth)

    def equal(self, y_pred: Tensor, y: Tensor) -> bool:
        eos_idx = uutils.torch.index(y_pred, self.mdl.vocab_ty['<eos>'])
        y_pred_trimmed: Seq[RuleIdx] = y_pred[:eos_idx]
        eos_idx = uutils.torch.index(y, self.mdl.vocab_ty['<eos>'])
        y_trimmed: Seq[RuleIdx] = y[:eos_idx]
        if len(y_pred_trimmed) != len(y_trimmed):
            return False
        equal = all(y_trimmed == y_pred_trimmed)
        return equal

    def log_train_stats(self, it: int, train_loss: float, train_acc: float,
                        save_val_ckpt: bool = False, val_iterations: int = 0,
                        epoch_print: bool = False):
        """

        :param it:
        :param train_loss:
        :param train_acc:
        :param save_val_ckpt: should we save the validation checkpoint?
        :param val_iterations:
        :param epoch_print:
        :return:
        """
        it_or_epoch = 'epoch_num' if epoch_print else 'it'
        val_loss, val_acc = self.valid(self.args.it, val_iterations=val_iterations, val_ckpt=save_val_ckpt)
        self.log_tb(it=it, tag1='train loss', loss=float(train_loss), tag2='train acc', acc=float(train_acc))
        self.log_tb(it=it, tag1='val loss', loss=float(val_loss), tag2='val acc', acc=float(val_acc))

        self.log(f"\n{it_or_epoch}={it}: {train_loss=} {train_acc=}")
        self.log(f"{it_or_epoch}={it}: {val_loss=} {val_acc=}")

    def save(self, epoch_num: int, dirname=None, ckpt_filename='mdl.pt'):
        if is_lead_worker(self.args.rank):
            self._save(epoch_num, dirname, ckpt_filename)

    def _save(self, epoch_num: int, dirname=None, ckpt_filename='mdl.pt'):
        """
        Saves checkpoint for any worker.
        Intended use is to save by worker that got a val loss that improved.

        todo - goot save ckpt always with epoch and it, much sure epoch_num and it are
        what you intended them to be...
        """
        from torch.nn.parallel.distributed import DistributedDataParallel
        dirname = self.args.log_root if dirname is None else dirname
        pickable_args = make_args_pickable(self.args)
        import dill
        mdl = self.mdl.module if type(self.mdl) is DistributedDataParallel else self.mdl
        # self.remove_term_encoder_cache()
        torch.save({'state_dict': self.mdl.state_dict(),
                    'epoch_num': epoch_num,
                    'it': self.args.it,
                    'optimizer': self.optimizer.state_dict(),
                    'args': pickable_args,
                    'mdl': mdl},
                   pickle_module=dill,
                   f=dirname / ckpt_filename)  # f'mdl_{epoch_num:03}.pt'

    def log(self, string, flush=True):
        """ logs only if you are rank 0"""
        if is_lead_worker(self.args.rank):
            print(string, flush=flush)

    def log_tb(self, it, tag1, loss, tag2, acc):
        if is_lead_worker(self.args.rank):
            uutils.torch.log_2_tb(self.tb, self.args, it, tag1, loss, tag2, acc)

    def print_process_info(self):
        print_process_info(self.args.rank)

    def is_lead_worker(self) -> bool:
        """
        Returns True if it's the lead worker (i.e. the slave meant to do the
        printing, logging, tb_logging, checkpointing etc.). This is useful for debugging.

        :return:
        """
        return is_lead_worker(self.args.rank)

# - tests

# def get_args():
#     merge = get_merge()
#     args = Namespace(rank=-1, world_size=1, merge=, ctx=ctx, batch_size=4)
#     args.device = 'cpu'
#     args.max_term_len = 1024
#     args.max_rule_seq_len = 2024
#     return args
#
# def test1():
#     args = get_args()
#     # -- pass data through the model
#     dataloaders = get_dataloaders(args, args.rank, args.world_size, args.merge)
#     parser = dataloaders['train'].dataset.extended_term_parser
#     # -- model
#     mdl = get_mdl()
#     # -- agent
#     agent = Agent(args, mdl)
#     for batch_idx, batch in enumerate(dataloaders['train']):
#         print(f'{batch_idx=}')
#         x_batch, y_batch = batch
#         y_pred = mdl(x_batch, y_batch)
#         print(f'{x_batch=}')
#         print(f'{y_batch=}')
#         if batch_idx > 5:
#             break
#
# if __name__ == "__main__":
#     test1()