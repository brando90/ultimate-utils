import logging
import os
import sys
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

from pdb import set_trace as st


def print_acc_loss_from_training_curve(path: Union[str, Path],
                                       learning_stats_fname: str = 'experiment_stats.json',
                                       metrics: list[str] = ['acc', 'loss'],
                                       splits: list[str] = ['train', 'val'],
                                       # test missing cuz we don't eval on test during training, no cheating!
                                       idx: int = -1,  # last value
                                       clear_accidental_ckpt_str_in_path: str = '/ckpt.pt'
                                       ) -> dict:
    """ print acc and loss from training curve ckpt.

    Intention:
    # path ~ log_root / 'experiment_stats.json'
    path = '~/data/logs/logs_Feb04_17-38-21_jobid_855372_pid_2723881_wandb_True/experiment_stats.json'
    # get dict and print acc and loss, split

    """
    if 'debug_5cnn_2filters' in str(path):
        data: dict = dict()
        for split in splits:
            for metric in metrics:
                data[f'{split}_{metric}_{idx}'] = -1
        return data
    # hopefully something like '~/data/logs/logs_Feb04_17-38-21_jobid_855372_pid_2723881_wandb_True/experiment_stats.json'
    path: Path = uutils.expanduser(path)
    if learning_stats_fname != 'experiment_stats.json':
        path = path / learning_stats_fname
    # - clear accidental ckpt str in path, remove it, then following code adds the right name to experiment json file
    if clear_accidental_ckpt_str_in_path in str(path):
        path: str = str(path).replace(clear_accidental_ckpt_str_in_path, '')
        path: Path = uutils.expanduser(path)
    # if end of string is not json, then user forgot to put the path to experiment run, so put your best guess based on how uutils works by default
    if not str(path).endswith('.json'):
        path = path / learning_stats_fname
    # - get loss/accs/metrics from training learning curve stats stored
    data: dict = dict()
    with open(path, 'r') as f:
        experiment_stats: dict = json.load(f)
        for split in splits:
            for metric in metrics:
                print(f'{split} {metric}: {experiment_stats[split][metric][idx]}')
                data[f'{split}_{metric}_{idx}'] = experiment_stats[split][metric][idx]
    return data


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
        loss, acc values/stats and save them, use those to generate my svg, png, pdf MI_plots_sl_vs_maml_1st_attempt, print/log only if the rank
        is for the lead worker when using ddp.

        Note: - Log/print info to log_root / log_filename. See: uutils.setup_args_for_experiment for details.
                - log_root path is usually ~/data/logs/logs_date_expt
              - Also save MI_plots_sl_vs_maml_1st_attempt to log_root path.

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
            # from pprint import pprint
            # pprint(msg)
            # st()
            # print(msg, flush=flush, file=sys.stdout)
            # uncomment two lines bellow perhaps to debug vit issue
            # print(f'{sys.stdout=}')
            # print(f'{os.path.realpath(sys.stdout.name)=}')
            print(msg, flush=flush)
            # print(msg, file=sys.stdout, flush=flush)
            # - to make sure it prints to the logger file too not just to console
            if log_file_name is not None:
                print(msg, file=self.args.log_root / log_file_name, flush=flush)

    # def pretty_log(self, obj: Any):
    #     """
    # better to use
    #     Good for printing dictionaries.
    #     :param obj:
    #     :return:
    #     """
    #     from pprint import pprint
    #     if is_lead_worker(self.args.rank):
    #         pprint(obj)
    #         # # - to make sure it prints to the logger file too not just to console
    #         # if log_file_name is not None:
    #         #     print(msg, file=self.args.log_root / log_file_name, flush=True)

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
            show: bool = False,

            wandb_log_fig: bool = False
    ):
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
                          label=tag1, linestyle='-', marker='x', color='r', linewidth=1)
            loss_ax1.plot(self.experiment_stats['val']['its'], self.experiment_stats['val']['loss'],
                          label=tag3, linestyle='-', marker='x', color='m', linewidth=1)

            loss_ax1.legend()
            loss_ax1.set_title(title)
            loss_ax1.set_ylabel('Loss')
            loss_ax1.grid(grid)

            # - plot stuff into acc axis
            acc_ax2.plot(self.experiment_stats['train']['its'], self.experiment_stats['train']['acc'],
                         label=tag2, linestyle='-', marker='x', color='b', linewidth=1)
            acc_ax2.plot(self.experiment_stats['val']['its'], self.experiment_stats['val']['acc'],
                         label=tag4, linestyle='-', marker='x', color='c', linewidth=1)

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

            if wandb_log_fig:
                assert False, 'Not tested'
                import wandb

                wandb.log(data={'fig': fig}, step=args.it, commit=True)
            # careful: even if you return the figure it seems it needs to be closed inside here anyway...so if you close it
            # but return it who knows what might happen.
            plt.close('all')  # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures


def save_current_plots_and_stats(
        title='Learnig & Evaluation Curves',

        grid: bool = True,
        show: bool = False,

        wandb_log_fig=False
):
    """
    TODO - adapt so that logger doesn't have this function as a method attached to the object

    :param title:
    :param grid:
    :param show:
    :param wandb_log_fig:
    :return:
    """
    # plt.style.use('default')
    # self.save_experiment_stats_to_json_file()

    tag1 = f'Train loss'
    # tag2 = f'Train accuracy/R2'
    tag3 = f'Eval loss'
    experiment_stats = uutils.load_json(
        '~/Desktop/paper_figs/logs_Nov23_11-39-21_jobid_438713.iam-pbs/experiment_stats.json')

    # - get figure with two axis, loss above and accuracy bellow
    fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # - plot stuff into loss axis
    loss_ax1.plot(experiment_stats['train']['its'], experiment_stats['train']['loss'],
                  label=tag1, linestyle='-', marker='x', color='r', linewidth=1)
    loss_ax1.plot(experiment_stats['val']['its'], experiment_stats['val']['loss'],
                  label=tag3, linestyle='-', marker='x', color='m', linewidth=1)

    loss_ax1.legend()
    loss_ax1.set_title(title)
    loss_ax1.set_ylabel('Loss')
    loss_ax1.grid(grid)

    # - plot stuff into acc axis
    # acc_ax2.plot(self.experiment_stats['train']['its'], self.experiment_stats['train']['acc'],
    #               label=tag2, linestyle='-', marker='x', color='b', linewidth=1)
    # acc_ax2.plot(self.experiment_stats['val']['its'], self.experiment_stats['val']['acc'],
    #               label=tag4, linestyle='-', marker='x', color='c', linewidth=1)

    # acc_ax2.legend()
    # x_axis_label: str = self.args.training_mode  # epochs or iterations
    # acc_ax2.set_xlabel(x_axis_label)
    # acc_ax2.set_ylabel(ylabel_acc)
    # acc_ax2.grid(grid)

    plt.tight_layout()

    plt.show() if show else None

    log_root = Path('~/Desktop').expanduser()
    fig.savefig(log_root / 'train_eval.svg')
    fig.savefig(log_root / 'train_eval.pdf')
    fig.savefig(log_root / 'train_eval.png')

    # if wandb_log_fig:
    #     assert False, 'Not tested'
    #     import wandb
    #
    #     wandb.log(data={'fig': fig}, step=args.it, commit=True)
    # careful: even if you return the figure it seems it needs to be closed inside here anyway...so if you close it
    # but return it who knows what might happen.
    # plt.close('all')  # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures


# - main, tests, examples, tutorials, etc.

def test_print_acc_loss_from_training_curve():
    path = '~/data/logs/logs_Mar30_08-17-19_jobid_17733_pid_142663'
    print_acc_loss_from_training_curve(path)
    path = '~/data/logs/logs_Mar30_08-17-19_jobid_17733_pid_142663/ckpt.pt/experiment_stats.json'
    print_acc_loss_from_training_curve(path)


# - run main, tests, examples, tutorials, etc.

if __name__ == '__main__':
    import time

    start_time = time.time()
    test_print_acc_loss_from_training_curve()
    print(f'Done! Finished in {time.time() - start_time:.2f} seconds\a\n'
          f''
          f''
          f''
          f''
          f'')
