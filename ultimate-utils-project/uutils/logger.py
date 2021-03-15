# from __future__ import division, print_function, absolute_import

import os
import sys
import logging

import torch
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

import json


class Logger:

    def __init__(self, args):
        """ Logger to record:
        - messages
        - data during training

        it always logs to my_stdout.log and the console (stdout) stream.
        
        Arguments:
            args {[type]} -- [description]
        """
        # super()
        self.args = args
        self.split = args.split
        self.current_logs_path = args.current_logs_path

        # logger = logging.getLogger(__name__) # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
        logger = logging.getLogger()
        level = logging.INFO if args.logging else logging.CRITICAL
        logger.setLevel(
            level)  # note: CAREFUL with not setting it or logging.UNSET: https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716

        ## log to my_stdout.log file
        file_handler = logging.FileHandler(filename=args.my_stdout_filepath)
        # file_handler.setLevel(logging.INFO) # not setting it means it inherits the logger. It will log everything from DEBUG upwards in severity to this handler.
        log_format = "{name}:{levelname}:{asctime}:{filename}:{funcName}:lineno {lineno}:->   {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        formatter = logging.Formatter(fmt=log_format, style='{')
        file_handler.setFormatter(fmt=formatter)

        ## log to stdout/screen
        stdout_stream_handler = logging.StreamHandler(
            stream=sys.stdout)  # default stderr, though not sure the advatages of logging to one or the other
        # stdout_stream_handler.setLevel(logging.INFO) # Note: having different set levels means that we can route using a threshold what gets logged to this handler
        log_format = "{name}:{levelname}:{filename}:{funcName}:lineno {lineno}:->   {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        formatter = logging.Formatter(fmt=log_format, style='{')
        stdout_stream_handler.setFormatter(fmt=formatter)

        logger.addHandler(hdlr=file_handler)  # add this file handler to top the logger
        logger.addHandler(hdlr=stdout_stream_handler)  # add this file handler to the top logger

        self.logger = logger
        self.reset_stats()  # to intialize a clean logger

    def reset_stats(self):
        self.stats = {'train': {'loss': [], 'acc': []},
                      'eval': {'loss': [], 'acc': []},
                      'eval_stats': {
                          'mean': {'loss': [], 'acc': []},
                          'std': {'loss': [], 'acc': []}
                        }
                      }

    def reset_eval_stats(self):
        self.stats['eval'] = {'loss': [], 'acc': []}

    def log_batch_train_info(self, loss, acc):
        """

        Note: self.stats[train][loss/acc][it] is the loss/acc for the train phase at iteration it.
        Note: for each training iteration you log time. For each epoch you log at the end of each epoch.
        :param loss:
        :param acc:
        :return:
        """
        phase = 'train'
        self.stats[phase]['loss'].append(float(loss))
        self.stats[phase]['acc'].append(float(acc))

    def log_batch_eval_info(self, loss, acc):
        """
        Intended usage is to collect a bunch of evaluations on different batches and use this
        function to collect them. Then at the end (once you have all the evals values you want), you
        call evaluate_logged_eval_stats. Then the eval_stats collect a list mean, std for the
        eval performance of the model at each time .evaluate_logged_eval_stats is called
        (usually at the end of an epoch or a iteration).

        Note: self.stats[eval][loss/acc][it] is the loss/acc for the eval phase at iteration it.
        Notice that an iteration here does not correspond to training.
        Note:
        :param loss:
        :param acc:
        :return:
        """
        phase = 'eval'
        self.stats[phase]['loss'].append(float(loss))
        self.stats[phase]['acc'].append(float(acc))

    def evaluate_logged_eval_stats_and_reset(self, reset_eval_stats=True):
        """
        Evaluates the stats (mean & std) of the collected eval losses & accs (val or test).
        It also resets the current list of logged eval stats.
        This is because this function is meant to be ran at the end of 1 epoch of evaluation.
        Example usage: collect a bunch of eval errors with log_eval_batch_info after the
        required number of epochs evaluate the errors.

        Same as old 'eval_stats' function with ._log_batch_info.

        :return:
        """
        loss_mean, loss_std = np.mean(self.stats['eval']['loss']), np.std(self.stats['eval']['loss'])
        acc_mean, acc_std = np.mean(self.stats['eval']['acc']), np.std(self.stats['eval']['acc'])
        self.stats['eval_stats']['mean']['loss'].append(loss_mean)
        self.stats['eval_stats']['mean']['acc'].append(acc_mean)
        self.stats['eval_stats']['std']['loss'].append(loss_std)
        self.stats['eval_stats']['std']['acc'].append(acc_std)
        if reset_eval_stats:
            self.reset_eval_stats()
        return acc_mean, acc_std, loss_mean, loss_std

    def logdebug(self, msg, *args, **kwargs):
        self.logger.debug(msg)

    def loginfo(self, msg, *args, **kwargs):
        self.logger.info(msg)

    def logerror(self, msg, *args, **kwargs):
        self.logger.error(msg)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def log_model_and_meta_learner_as_string(self, base_model, meta_learner, current_logs_path=None):
        """
        Note: we are NOT checkpointing here because log_validation hadles that
        """
        try:
            current_logs_path = self.current_logs_path if current_logs_path is None else current_logs_path
            # DO NOT checkpoint here, torch.save(model, current_logs_path / '')
            with open(current_logs_path / 'base_model_and_meta_learner', 'w+') as f:
                # json.dump({'base_model': str(base_model), 'meta_learner': str(meta_learner)}, f, indent=4)
                f.write(str(base_model))
                f.write(str(meta_learner))
        except:
            pass

    def save_stats_to_json_file(self, current_logs_path=None):
        current_logs_path = self.current_logs_path if current_logs_path is None else current_logs_path
        torch.save(self.stats, current_logs_path / 'experiment_stats')
        with open(current_logs_path / 'experiment_stats.json', 'w+') as f:
            json.dump(self.stats, f, indent=4)

    def save_current_plots_and_stats(
            self,
            title='Meta-Learnig & Evaluation Curves',
            x_axis='(meta) iterations',
            y_axis_loss='Meta-Loss',
            y_axis_acc='Meta-Accuracy',

            grid=True,
            show=False):
        # Initialize where to save and what the split of the experiment is
        current_logs_path = self.current_logs_path if current_logs_path is None else current_logs_path

        # save current stats
        self.save_stats_to_json_file(current_logs_path)
        plt.style.use('default')

        eval_label = 'Val' if split == 'meta-train' else 'Test'

        if self.args.target_type == 'regression':
            tag1 = f'Train loss'
            tag2 = f'Train R2'
            tag3 = f'{eval_label} loss'
            tag4 = f'{eval_label} R2'
        elif self.args.target_type == 'classification':
            tag1 = f'Train loss'
            tag2 = f'Train accuracy'
            tag3 = f'{eval_label} loss'
            tag4 = f'{eval_label} accuracy'
        else:
            raise ValueError(f'Error: args.target_type = {self.args.target_type} not valid.')

        train_loss_y = self.stats['train']['loss']
        train_acc_y = self.stats['train']['acc']
        assert (len(train_acc_y) == len(train_loss_y))
        # plus one so to start episode 1, since 0 is not recorded yet...
        episodes_train_x = np.array([self.args.log_train_freq * (i + 1) for i in range(len(train_loss_y))])
        assert (len(episodes_train_x) == len(train_loss_y))

        eval_loss_y = self.stats['eval_stats']['mean']['loss']
        eval_acc_y = self.stats['eval_stats']['mean']['acc']
        assert (len(eval_loss_y) == len(eval_acc_y))
        eval_loss_std = self.stats['eval_stats']['std']['loss']
        eval_acc_std = self.stats['eval_stats']['std']['acc']
        assert (len(eval_acc_std) == len(eval_loss_std) and len(eval_loss_y) == len(eval_loss_std))
        # plus one so to start episode 1, since 0 is not recorded yet...
        episodes_eval_x = np.array([self.args.log_val_freq * (i + 1) for i in range(len(eval_loss_y))])
        assert (len(episodes_eval_x) == len(eval_acc_y))

        fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        loss_ax1.plot(episodes_train_x, train_loss_y, label=tag1, linestyle='-', marker='o', color='r', linewidth=1)
        loss_ax1.plot(episodes_eval_x, eval_loss_y, label=tag2, linestyle='-', marker='o', color='m', linewidth=1)

        loss_ax1.legend()
        loss_ax1.set_title(title)
        loss_ax1.set_ylabel(y_axis_loss)
        loss_ax1.grid(grid)

        acc_ax2.plot(episodes_train_x, train_acc_y, label=tag3, linestyle='-', marker='o', color='b', linewidth=1)
        acc_ax2.errorbar(episodes_eval_x, eval_acc_y, yerr=eval_acc_std, label=tag4, linestyle='-', marker='o',
                         color='c', linewidth=1, capsize=3)
        acc_ax2.legend()
        acc_ax2.set_xlabel(x_axis)
        acc_ax2.set_ylabel(y_axis_acc)
        acc_ax2.grid(grid)

        plt.tight_layout()

        plt.show() if show else None

        fig.savefig(current_logs_path / 'meta_train_eval.svg')
        fig.savefig(current_logs_path / 'meta_train_eval.pdf')
        fig.savefig(current_logs_path / 'meta_train_eval.png')
        plt.close('all')  # https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures