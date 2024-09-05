#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python
"""
We define agent as the python class that has a tactic_predictor/model for proving and does the proving, triaing, validation, etc.

"""
import math
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from torch.utils.tensorboard import SummaryWriter

from progressbar import ProgressBar
# from tqdm import tqdm

import gc

from uutils import make_args_pickable, set_system_wide_force_flush
from uutils.torch_uu import AverageMeter, accuracy, print_dataloaders_info
from uutils.torch_uu.distributed import is_running_serially, is_lead_worker, print_process_info, process_batch_ddp_tactic_prediction

from uutils.torch_uu.tensorboard import log_2_tb

from pdb import set_trace as st

# -- Hash Pred agent

# @profile
def _next(x):
    return next(x)

class HashPredictorAgent:
    """Agent to test embedding capability on lasse's dataset

    Note: you cannot pickle this object because of tensorboard object, you need to make that into string.
    It's in opts, just in case tactic predictor or anyone with a handle to opts want to write to tensorboard.
    """

    def __init__(self, tactic_predictor, optimizer, dataloaders, opts):
        self.tactic_predictor = tactic_predictor
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.opts = opts
        self.best_val_loss = float('inf')
        if is_lead_worker(self.opts.rank):
            self.tb = SummaryWriter(log_dir=opts.tb_dir)

    def train(self, n_epoch):
        """
        Train in the generic epoch way. Note the logging has to be done outside this code.

        :param n_epoch:
        :return:
        """
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')

        bar = ProgressBar(max_value=len(self.dataloaders['train']))
        self.print_dataloaders_info(split='train')
        for i, data_batch in enumerate(self.dataloaders['train']):
            self.tactic_predictor.train()
            # self.tactic_predictor.term_encoder.cache = {}
            data_batch = process_batch_ddp_tactic_prediction(self.opts, data_batch)
            loss, logits = self.tactic_predictor(data_batch)
            acc = accuracy(output=logits, target=data_batch['tac_label'])
            avg_loss.update(loss.item(), self.opts.batch_size)
            avg_acc.update(acc.item(), self.opts.batch_size)

            self.optimizer.zero_grad()
            loss.backward()  # each process synchronizes it's gradients in the backward pass
            self.optimizer.step()  # the right update is done since all procs have the right synced grads

            if self.is_lead_worker() and i % 100 == 0:
                bar.update(i) if self.is_lead_worker() else None

            gc.collect()

        return avg_loss.item(), avg_acc.item()

    def valid(self, n_epoch, val_iterations=0, val_ckpt=True):
        """

        :param n_epoch:
        :param val_iterations: 0 default value means it does 1 iteration - so only sees 1 batch.
        :param val_ckpt:
        :return:
        """
        avg_loss = AverageMeter('val loss')
        avg_acc = AverageMeter('val accuracy')

        # bar = ProgressBar(max_value=len(self.dataloaders['val']))
        for i, data_batch in enumerate(self.dataloaders['val']):
            self.tactic_predictor.eval()
            data_batch = process_batch_ddp_tactic_prediction(self.opts, data_batch)
            loss, logits = self.tactic_predictor(data_batch)
            acc = accuracy(output=logits, target=data_batch['tac_label'])
            avg_loss.update(loss, self.opts.batch_size)
            avg_acc.update(acc, self.opts.batch_size)

            gc.collect()
            if i >= val_iterations:
                break
            if self.opts.debug and i > 2:
                break

        if avg_loss.item() < self.best_val_loss and val_ckpt:
            self.best_val_loss = loss
            self._save(n_epoch, ckpt_name='tactic_predictor_best_val.pt')
        return avg_loss.item(), avg_acc.item()

    def train_test_debug(self, split):
        """
        Same as train but prints and tb_logs every iteration.

        :param n_epoch:
        :return:
        """
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')

        data_loader = self.dataloaders[split]
        freq = int(math.ceil(len(data_loader) / 3))  # to log approximately 2-3 times.
        self.log(f'{freq=} (freq such that we log every 2-3 times)')
        for i, data_batch in enumerate(data_loader):
            self.tactic_predictor.train()
            data_batch = process_batch_ddp_tactic_prediction(self.opts, data_batch)
            train_loss, logits = self.tactic_predictor(data_batch)
            acc = accuracy(output=logits, target=data_batch['tac_label'])
            avg_loss.update(train_loss.item(), self.opts.batch_size)
            avg_acc.update(acc.item(), self.opts.batch_size)

            self.optimizer.zero_grad()
            train_loss.backward()  # each process synchronizes it's gradients in the backward pass
            self.optimizer.step()  # the right update is done since all procs have the right synced grads

            if self.is_lead_worker() and (i % freq == 0 or i >= len(data_loader) - 1):
                self.log_train_stats(self.opts.it, train_loss.item(), acc.item())

            self.opts.it += 1
            gc.collect()

        return avg_loss.item(), avg_acc.item()

    def log_train_stats(self, it: int, train_loss: float, train_acc: float, val_ckpt=False, val_iterations=0):
        val_loss, val_acc = self.valid(self.opts.n_epoch, val_iterations=val_iterations, val_ckpt=val_ckpt)
        self.log_tb(it=it, tag1='train loss', loss=float(train_loss), tag2='train acc', acc=float(train_acc))
        self.log_tb(it=it, tag1='val loss', loss=float(val_loss), tag2='val acc', acc=float(val_acc))

        self.log(f"\n{it=}: {train_loss=} {train_acc=}")
        self.log(f"{it=}: {val_loss=} {val_acc=}")

    def predict_tactic_hashes(self, prediction_mode):
        if prediction_mode == 'train':
            self.tactic_predictor.train()
        else:
            self.tactic_predictor.eval()
        data_loader = iter(self.dataloaders[self.opts.split])
        data_batch = next(data_loader)
        data_batch = process_batch_ddp_tactic_prediction(self.opts, data_batch)
        train_loss, logits = self.tactic_predictor(data_batch)
        acc = accuracy(output=logits, target=data_batch['tac_label'])
        return train_loss.item(), acc.item()

    def train_single_batch(self, acc_tolerance=1.0, train_loss_tolerance=0.01):
        """
        Train untils the accuracy on the specified batch has perfect interpolation in loss and accuracy.
        It also prints and tb logs every iteration.

        :param acc_tolerance:
        :param train_loss_tolerance:
        :return:
        """
        print('train_single_batch')
        set_system_wide_force_flush()
        avg_loss = AverageMeter('train loss')
        avg_acc = AverageMeter('train accuracy')

        def forward_one_batch(data_batch, training):
            self.tactic_predictor.train() if training else self.tactic_predictor.eval()
            data_batch = process_batch_ddp_tactic_prediction(self.opts, data_batch)
            loss, logits = self.tactic_predictor(data_batch)
            acc = accuracy(output=logits, target=data_batch['tac_label'])
            avg_loss.update(loss.item(), self.opts.batch_size)
            avg_acc.update(acc.item(), self.opts.batch_size)
            return loss, acc

        def log_train_stats(it: int, train_loss: float, acc: float):
            val_loss, val_acc = forward_one_batch(val_batch, training=False)
            self.log_tb(it=it, tag1='train loss', loss=float(train_loss), tag2='train acc', acc=float(acc))
            self.log_tb(it=it, tag1='val loss', loss=float(val_loss), tag2='val acc', acc=float(val_acc))

            self.log(f"\n{it=}: {train_loss=} {acc=}")
            self.log(f"{it=}: {val_loss=} {val_acc=}")

        # train_acc = 0.0; train_loss = float('inf')
        data_batch = next(iter(self.dataloaders['train']))
        val_batch = next(iter(self.dataloaders['val']))
        self.opts.it = 0
        while True:
            train_loss, train_acc = forward_one_batch(data_batch, training=True)

            self.optimizer.zero_grad()
            train_loss.backward()  # each process synchronizes it's gradients in the backward pass
            self.optimizer.step()  # the right update is done since all procs have the right synced grads

            if self.is_lead_worker() and self.opts.it % 10 == 0:
                log_train_stats(self.opts.it, train_loss, train_acc)
                self.save(self.opts.it)  # very expensive! since your only fitting one batch its ok to save it every time you log - but you might want to do this left often.

            self.opts.it += 1
            gc.collect()
            # if train_acc >= acc_tolerance and train_loss <= train_loss_tolerance:
            if train_acc >= acc_tolerance:
                log_train_stats(self.opts.it, train_loss, train_acc)
                self.save(self.opts.it)  # very expensive! since your only fitting one batch its ok to save it every time you log - but you might want to do this left often.
                break  # halt once both the accuracy is high enough AND train loss is low enough

        return avg_loss.item(), avg_acc.item()

    def evaluate(self, filename, proof_name=None):
        pass

    def prove_one_tactic(self, proof_env, tac):
        pass

    def prove(self, proof_env):
        pass

    def prove_DFS(self, proof_env, tac_template):
        pass

    def prove_IDDFS(self, proof_env, tac_template):
        pass

    def save(self, n_epoch, dirname=None, ckpt_name='tactic_predictor.pt'):
        if is_lead_worker(self.opts.rank):
            self._save(n_epoch, dirname, ckpt_name)

    def _save(self, n_epoch, dirname=None, ckpt_name='tactic_predictor.pt'):
        """
        Saves checkpoint for any worker.
        Intended use is to save by worker that got a val loss that improved.
        """
        dirname = self.opts.log_root if dirname is None else dirname
        pickable_opts = make_args_pickable(self.opts)
        import dill
        tactic_predictor = self.tactic_predictor.module if type(self.tactic_predictor) is DistributedDataParallel else self.tactic_predictor
        # self.remove_term_encoder_cache()
        torch.save({'state_dict': self.tactic_predictor.state_dict(),
                    'n_epoch': n_epoch,
                    'optimizer': self.optimizer.state_dict(),
                    'opts': pickable_opts,
                    'tactic_predictor': tactic_predictor},
                   pickle_module=dill,
                   f=dirname / ckpt_name)  # f'tactic_predictor_{n_epoch:03}.pt'

    def save_every_x(self, n_epoch):
        if self.is_lead_worker():
            current_dime_duration_hours = (time.time() - self.opts.start) / (60*60)
            if current_dime_duration_hours >= self.opts.next_time_to_ckpt_in_hours:
                self.opts.next_time_to_ckpt_in_hours += self.opts.ckpt_freq_in_hours
                self.save(n_epoch)

    def remove_term_encoder_cache(self):
        # this creates a bug with DDP fix later
        if hasattr(self.tactic_predictor.term_encoder, 'cache'):
            self.tactic_predictor.term_encoder.cache = {}

    def log(self, string, flush=True):
        """ logs only if you are rank 0"""
        if is_lead_worker(self.opts.rank):
            print(string, flush=flush)

    def log_tb(self, it, tag1, loss, tag2, acc):
        if is_lead_worker(self.opts.rank):
            log_2_tb(self.tb, self.opts, it, tag1, loss, tag2, acc)

    def print_process_info(self):
        print_process_info(self.opts.rank)

    def is_lead_worker(self):
        """
        Returns True if it's the lead worker (i.e. the slave meant to do the
        printing, logging, tb_logging, checkpointing etc.). This is useful for debugging.

        :return:
        """
        return is_lead_worker(self.opts.rank)

    def print_dataloaders_info(self, split):
        if self.is_lead_worker():
            print_dataloaders_info(self.opts, self.dataloaders, split)
