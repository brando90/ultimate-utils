from argparse import Namespace
from typing import Any, Union

import torch
import torch.nn as nn

# import matplotlib as mpl
from matplotlib import pyplot as plt

import json

from pathlib import Path

from torch.nn.functional import mse_loss
from torch.optim import Optimizer

import uutils
# from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu import r2_score_from_torch
from uutils.torch_uu.distributed import is_lead_worker
from uutils.torch_uu.models import get_simple_model
from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning


def save_model_as_string(args: Namespace, model: nn.Module):
    """ save model as as string. """
    with open(args.log_root / 'model_as_str.txt', 'w+') as f:
        # json.dump({'base_model': str(base_model), 'meta_learner': str(meta_learner)}, f, indent=4)
        f.write(str(model))


def save_model_and_meta_learner_as_string(args: Namespace, model: nn.Module, meta_learner: Any = None):
    """ save model and meta-learner as string. """
    with open(args.log_root / 'model_as_str.txt', 'w+') as f:
        # json.dump({'base_model': str(base_model), 'meta_learner': str(meta_learner)}, f, indent=4)
        f.write(str(model))
        if meta_learner is not None:
            f.write(str(meta_learner))


class Logger:

    def __init__(self, args: Namespace, log_with_logger: bool = False):
        """
        Main functionality is to print to console (and optionally to a file in the log_root too), track experiment
        loss, acc values/stats and save them, use those to generate my svg, png, pdf plots, print/log only if the rank
        is for the lead worker when using ddp.

        Note: - Log/print info to log_root / log_filename. See: uutils.setup_args_for_experiment for details.
                - log_root path is usually ~/data/logs/logs_date_expt
              - Also save plots to log_root path.

        Decided not to include tb tensorboard stuff here (arbitarily). To seperate tb from my code and keep tb stuff
        as functional as possible.

        For python library tutorial see: https://realpython.com/python-logging/
        """
        self.args = args
        assert hasattr(args, 'log_root'), 'You need to set a log_root for this to work.'

        assert log_with_logger == False, 'Logging with logger not implemented.'
        if log_with_logger:
            raise ValueError('Not implemented')
            self.set_up_logger()

        # - init experiment stats e.g. loss values, acc values. For reg acc should be R2.
        self.experiment_stats: dict = {'train': {'its': [], 'loss': [], 'acc': []},
                                       'val': {'its': [], 'loss': [], 'acc': []},
                                       'test': {'its': [], 'loss': [], 'acc': []},
                                       }

    def log(self, msg: str, flush: bool = False, log_file_name: Union[None, str] = None):
        """
        If lead worker, prints to console. If log_file_name present then logs to log_file in experiment folder too e.g.
        at location log_root / log_file_name e.g. ~/data/logs/logs_data_../log_file.log

        Suggested log_file_name = 'log_file.log'. Note it will be loged at log_root / log_file_name.

        :param msg:
        :param flush:
        :param log_file_name:
        :return:
        """
        if is_lead_worker(self.args.rank):
            # - guarantees it prints to console
            print(msg, flush=flush)
            # print(msg, file=sys.stdout, flush=flush)
            # - to make sure it prints to the logger file too not just to console
            if log_file_name is not None:
                print(msg, file=self.args.log_root / log_file_name, flush=flush)

    def record_train_stats_stats_collector(self, it: int, loss: float, acc: float):
        """
        Records the loss, acc and the iteration it happened.

        Note: recording the iteration it happened helps for plotting.
        Note: self.experiment_stats[train][loss/acc][it] is the loss/acc for the train phase at iteration it.
        Note: for each training iteration you log time. For each epoch you log at the end of each epoch.
        :param it:
        :param loss:
        :param acc:
        :return:
        """
        if is_lead_worker(self.args.rank):
            self._record(phase='train', it=it, loss=loss, acc=acc)

    def record_val_stats_stats_collector(self, it: int, loss: float, acc: float):
        """
        Note: recording the iteration it happened helps for plotting.
        Note: self.experiment_stats[train][loss/acc][it] is the loss/acc for the train phase at iteration it.
        Note: for each training iteration you log time. For each epoch you log at the end of each epoch.
        :param it:
        :param loss:
        :param acc:
        :return:
        """
        if is_lead_worker(self.args.rank):
            self._record(phase='val', it=it, loss=loss, acc=acc)

    def _record(self, phase: str, it: int, loss: float, acc: float):
        """"""
        if is_lead_worker(self.args.rank):
            self.experiment_stats[phase]['its'].append(int(it))
            self.experiment_stats[phase]['loss'].append(float(loss))
            self.experiment_stats[phase]['acc'].append(float(acc))

    # def _logdebug(self, msg, *args, **kwargs):
    #     if is_lead_worker(self.args.rank):
    #         self.logger.debug(msg)
    #
    # def _loginfo(self, msg, *args, **kwargs):
    #     if is_lead_worker(self.args.rank):
    #         self.logger.info(msg)
    #
    # def _logerror(self, msg, *args, **kwargs):
    #     if is_lead_worker(self.args.rank):
    #         self.logger.error(msg)
    #
    # def _log(self, level, msg, *args, **kwargs):
    #     if is_lead_worker(self.args.rank):
    #         self.logger.log(level, msg, *args, **kwargs)

    def save_experiment_stats_to_json_file(self):
        """ save experiment stats to json file. """
        # torch_uu.save(self.experiment_stats, current_logs_path / 'experiment_stats')
        with open(self.args.log_root / 'experiment_stats.json', 'w+') as f:
            json.dump(self.experiment_stats, f, indent=4, sort_keys=True)

    def set_up_logger(self):
        pass
        # logger = logging.getLogger()
        # # default logging.UNSET: https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716
        # logger.setLevel(logging.INFO)
        #
        # ## log to my_stdout.log file
        # file_handler = logging.FileHandler(filename=args.my_stdout_filepath)
        # # file_handler.setLevel(logging.INFO) # not setting it means it inherits the logger. It will log everything from DEBUG upwards in severity to this handler.
        # log_format = "{name}:{levelname}:{asctime}:{filename}:{funcName}:lineno {lineno}:->   {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        # formatter = logging.Formatter(fmt=log_format, style='{')
        # file_handler.setFormatter(fmt=formatter)
        #
        # ## log to stdout/screen
        # stdout_stream_handler = logging.StreamHandler(
        #     stream=sys.stdout)  # default stderr, though not sure the advatages of logging to one or the other
        # # stdout_stream_handler.setLevel(logging.INFO) # Note: having different set levels means that we can route using a threshold what gets logged to this handler
        # log_format = "{name}:{levelname}:{filename}:{funcName}:lineno {lineno}:->   {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        # formatter = logging.Formatter(fmt=log_format, style='{')
        # stdout_stream_handler.setFormatter(fmt=formatter)
        #
        # logger.addHandler(hdlr=file_handler)  # add this file handler to top the logger
        # logger.addHandler(hdlr=stdout_stream_handler)  # add this file handler to the top logger
        #
        # self.logger = logger

    def save_current_plots_and_stats(
            self,
            title='Learnig & Evaluation Curves',

            grid: bool = True,
            show: bool = False):
        if is_lead_worker(self.args.rank):
            # plt.style.use('default')
            # self.save_experiment_stats_to_json_file()

            if not hasattr(self.args, 'target_type'):
                tag1 = f'Train loss'
                tag2 = f'Train accuracy/R2'
                tag3 = f'Val loss'
                tag4 = f'Val accuracy/R2'
                ylabel_acc = 'Accuracy/R2'
                # raise ValueError(f'Error: args.target_type = {self.args.target_type} not valid.')
            elif self.args.target_type == 'regression':
                tag1 = f'Train loss'
                tag2 = f'Train R2'
                tag3 = f'Val loss'
                tag4 = f'Val R2'
                ylabel_acc = 'R2'
            elif self.args.target_type == 'classification':
                tag1 = f'Train loss'
                tag2 = f'Train accuracy'
                tag3 = f'Val loss'
                tag4 = f'Val accuracy'
                ylabel_acc = 'Accuracy'
            else:
                raise ValueError(f'Not implemented {self.args.target_type}')

            # - get figure with two axis, loss above and accuracy bellow
            fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

            # - plot stuff into loss axis
            loss_ax1.plot(self.experiment_stats['train']['its'], self.experiment_stats['train']['loss'],
                          label=tag1, linestyle='-', marker='o', color='r', linewidth=1)
            loss_ax1.plot(self.experiment_stats['val']['its'], self.experiment_stats['val']['loss'],
                          label=tag3, linestyle='-', marker='o', color='m', linewidth=1)

            loss_ax1.legend()
            loss_ax1.set_title(title)
            loss_ax1.set_ylabel('Loss')
            loss_ax1.grid(grid)

            # - plot stuff into acc axis
            acc_ax2.plot(self.experiment_stats['train']['its'], self.experiment_stats['train']['acc'],
                          label=tag2, linestyle='-', marker='o', color='b', linewidth=1)
            acc_ax2.plot(self.experiment_stats['val']['its'], self.experiment_stats['val']['acc'],
                          label=tag4, linestyle='-', marker='o', color='c', linewidth=1)

            acc_ax2.legend()
            x_axis_label: str = self.args.training_mode  # epochs or iterations
            acc_ax2.set_xlabel(x_axis_label)
            acc_ax2.set_ylabel(ylabel_acc)
            acc_ax2.grid(grid)

            plt.tight_layout()

            plt.show() if show else None

            fig.savefig(self.args.log_root / 'train_eval.svg')
            fig.savefig(self.args.log_root / 'train_eval.pdf')
            fig.savefig(self.args.log_root / 'train_eval.png')
            plt.close('all')  # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

# - tests

def log_train_val_stats_example(args: Namespace,
                    it: int,

                    train_loss: float,
                    train_acc: float,

                    valid,

                    save_val_ckpt: bool = False,
                    log_to_tb: bool = False,
                    log_to_wandb: bool = False
                    ):
    """

    log train and val stats.

    Note: Unlike save ckpt, this one does need it to be passed explicitly (so it can save it in the stats collector).
    """
    from uutils.torch_uu.tensorboard import log_2_tb

    # - is it epoch or iteration
    it_or_epoch = 'epoch_num' if args.training_mode == 'epochs' else args.training_mode == 'iteration'

    # - get eval stats
    val_loss, val_acc = valid(args, args.mdl, save_val_ckpt=save_val_ckpt)

    # - print
    args.logger.log('\n')
    args.logger.log(f"{it_or_epoch}={it}: {train_loss=}, {train_acc=}")
    args.logger.log(f"{it_or_epoch}={it}: {val_loss=}, {val_acc=}")

    # - record into stats collector
    args.logger.record_train_stats_stats_collector(it, train_loss, train_acc)
    args.logger.record_val_stats_stats_collector(it, val_loss, val_acc)
    args.logger.save_experiment_stats_to_json_file()
    args.logger.save_current_plots_and_stats()

    # - log to wandb
    if log_to_wandb:
        pass

    # - log to tensorboard
    if log_to_tb:
        log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'train')
        log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'val')
        # log_2_tb(args, it, val_loss, val_acc, 'train')
        # log_2_tb(args, it, val_loss, val_acc, 'val')

def save_ckpt(args: Namespace, mdl: nn.Module, optimizer: torch.optim.Optimizer,
              dirname: Union[None, Path] = None, ckpt_name: str = 'ckpt.pt'):
    """
    Saves checkpoint for any worker.
    Intended use is to save by worker that got a val loss that improved.


    """
    import dill

    dirname = args.log_root if (dirname is None) else dirname
    # - pickle ckpt
    assert uutils.xor(args.training_mode == 'epochs', args.training_mode == 'iterations')
    pickable_args = uutils.make_args_pickable(args)
    torch.save({'state_dict': mdl.state_dict(),
                'epoch_num': args.epoch_num,
                'it': args.it,
                'optimizer': optimizer.state_dict(),
                'args': pickable_args,
                'mdl': mdl},
               pickle_module=dill,
               f=dirname / ckpt_name)  # f'mdl_{epoch_num:03}.pt'

def get_args() -> Namespace:
    args = uutils.parse_args_synth_agent()
    args = uutils.setup_args_for_experiment(args)
    return args

def valid_for_test(args: Namespace, mdl: nn.Module, save_val_ckpt: bool = False):
    import torch

    for t in range(1):
        x = torch.randn(args.batch_size, 5)
        y = (x**2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        val_loss, val_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

    if val_loss.item() < args.best_val_loss and save_val_ckpt:
        args.best_val_loss = val_loss.item()
        save_ckpt(args, args.mdl, args.optimizer, ckpt_name='ckpt_best_val.pt')
    return val_loss, val_acc

def train_for_test(args: Namespace, mdl: nn.Module, optimizer: Optimizer, scheduler = None):
    for it in range(50):
        x = torch.randn(args.batch_size, 5)
        y = (x**2 + x + 1).sum(dim=1)

        y_pred = mdl(x).squeeze(dim=1)
        train_loss, train_acc = mse_loss(y_pred, y), r2_score_from_torch(y_true=y, y_pred=y_pred)

        optimizer.zero_grad()
        train_loss.backward()  # each process synchronizes it's gradients in the backward pass
        optimizer.step()  # the right update is done since all procs have the right synced grads
        scheduler.step()

        if it % 2 == 0 and is_lead_worker(args.rank):
            log_train_val_stats_example(args, it, train_loss, train_acc, valid_for_test, save_val_ckpt=True, log_to_tb=True)
            if it % 10 == 0:
                save_ckpt(args, args.mdl, args.optimizer)

    return train_loss, train_acc

def debug_test():
    args: Namespace = get_args()

    # - get mdl, opt, scheduler, etc
    args.mdl = get_simple_model(in_features=5, hidden_features=20, out_features=1, num_layer=2)
    args.optimizer = torch.optim.Adam(args.mdl.parameters(), lr=1e-1)
    args.scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.999, verbose=False)

    # - train
    train_loss, train_acc = train_for_test(args, args.mdl, args.optimizer, args.scheduler)
    print(f'{train_loss=}, {train_loss=}')

    # - eval
    val_loss, val_acc = valid_for_test(args, args.mdl)

    print(f'{val_loss=}, {val_acc=}')


if __name__ == '__main__':
    debug_test()
    print('Done!\a')
