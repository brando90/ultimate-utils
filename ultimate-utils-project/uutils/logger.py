#from __future__ import division, print_function, absolute_import

import os
import sys
import logging

import torch
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from pdb import set_trace as st


class Logger:

    def __init__(self, args):
        """ Logger to record:
        - messages
        - data during training

        it always logs to my_stdout.log and the console (stdout) stream.
        
        Arguments:
            args {[type]} -- [description]
        """
        #super()
        self.args = args
        self.mode = args.mode
        self.current_logs_path = args.current_logs_path
        
        # logger = logging.getLogger(__name__) # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
        logger = logging.getLogger()
        level = logging.INFO if args.logging else logging.CRITICAL
        logger.setLevel(level) # note: CAREFUL with not setting it or logging.UNSET: https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716

        ## log to my_stdout.log file
        file_handler = logging.FileHandler(filename=args.my_stdout_filepath)
        #file_handler.setLevel(logging.INFO) # not setting it means it inherits the logger. It will log everything from DEBUG upwards in severity to this handler.
        log_format = "{name}:{levelname}:{asctime}:{filename}:{funcName}:lineno {lineno}:->   {message}" # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        formatter = logging.Formatter(fmt=log_format, style='{')
        file_handler.setFormatter(fmt=formatter)

        ## log to stdout/screen
        stdout_stream_handler = logging.StreamHandler(stream=sys.stdout) # default stderr, though not sure the advatages of logging to one or the other
        #stdout_stream_handler.setLevel(logging.INFO) # Note: having different set levels means that we can route using a threshold what gets logged to this handler
        log_format = "{name}:{levelname}:{filename}:{funcName}:lineno {lineno}:->   {message}" # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
        formatter = logging.Formatter(fmt=log_format, style='{')
        stdout_stream_handler.setFormatter(fmt=formatter)

        logger.addHandler(hdlr=file_handler) # add this file handler to top the logger
        logger.addHandler(hdlr=stdout_stream_handler) # add this file handler to the top logger
        
        self.logger = logger
        self.reset_stats()

    def reset_stats(self):
        self.stats = {  'train': {'loss': [], 'acc': []},
                        'eval': {'loss': [], 'acc': []},
                        'eval_stats': {
                            'mean': {'loss': [], 'acc': []},
                            'std': {'loss': [], 'acc': []}
                        }
                    }

    def reset_eval_stats(self):
        self.stats['eval'] = {'loss': [], 'acc': []}

    def batch_info(self, args, **kwargs):
        phase = kwargs['phase']
        if phase == 'train':
            self.stats[phase]['loss'].append(float(kwargs['loss']))
            self.stats[phase]['acc'].append(float(kwargs['acc']))
        elif phase == 'eval':
            self.stats[phase]['loss'].append(float(kwargs['loss']))
            self.stats[phase]['acc'].append(float(kwargs['acc']))
            # never log eval, only at the end of the whole eval set of episodes & this gets reset
        elif phase == 'eval_stats':
            loss_mean, loss_std = np.mean(self.stats['eval']['loss']), np.std(self.stats['eval']['loss'])
            acc_mean, acc_std = np.mean(self.stats['eval']['acc']), np.std(self.stats['eval']['acc'])
            self.stats['eval_stats']['mean']['loss'].append(loss_mean)
            self.stats['eval_stats']['mean']['acc'].append(acc_mean)
            self.stats['eval_stats']['std']['loss'].append(loss_std)
            self.stats['eval_stats']['std']['acc'].append(acc_std)
            self.reset_eval_stats()
            return acc_mean, acc_std, loss_mean, loss_std
        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))

    def logdebug(self, msg, *args, **kwargs):
        self.logger.debug(msg)

    def loginfo(self, msg, *args, **kwargs):
        self.logger.info(msg)

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def log_growing_matplot(self):
        raise('Not implemented yet')
    
    def log_training_losses(self):
        raise('Not implemented')

    def pickle_stuff(self):
        raise('Not implemented')

    def save_final_plots(self, nb_plots=1, mode=None, current_logs_path=None, xkcd=False, grid=True, show=False):
        plt.xkcd() if xkcd else None
        ## Initialize where to save and what the mode of the experiment is
        mode = self.mode if mode is None else mode
        current_logs_path = self.current_logs_path if current_logs_path is None else current_logs_path

        ## https://stackoverflow.com/questions/61415955/why-dont-the-error-limits-in-my-plots-show-in-matplotlib
        ## mpl.rcParams["errorbar.capsize"] = 3

        plt.style.use('default')

        if mode == 'meta-train':
            eval_label = 'Val' # in train mode the evaluation of the model should  meta-val set
        else:
            eval_label = 'Test'
        
        train_loss_y = self.stats['train']['loss']
        train_acc_y = self.stats['train']['acc']
        assert(len(train_acc_y) == len(train_loss_y))
        # plus one so to start episode 1, since 0 is not recorded yet...
        episodes_train_x = np.arange( start=0, stop=len(train_loss_y), step=self.args.log_train_freq) + 1

        eval_loss_y = self.stats['eval_stats']['mean']['loss']
        eval_acc_y = self.stats['eval_stats']['mean']['acc']
        assert(len(eval_loss_y) == len(eval_acc_y))
        eval_loss_std = self.stats['eval_stats']['std']['loss']
        eval_acc_std = self.stats['eval_stats']['std']['acc'] 
        assert(len(eval_acc_std) == len(eval_loss_std) and len(eval_loss_y) == len(eval_loss_std))
        # plus one so to start episode 1, since 0 is not recorded yet...
        episodes_eval_x = np.arange( start=0, stop=len(eval_loss_y), step=self.args.val_freq) + 1 

        if nb_plots == 1:
            fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

            loss_ax1.plot(episodes_train_x, train_loss_y, label='Train Loss', linestyle='-', marker='o', color='r', linewidth=1)
            loss_ax1.errorbar(episodes_eval_x, eval_loss_y, yerr=eval_loss_std, label=f'{eval_label} Loss', linestyle='-', marker='o', color='m', linewidth=1, capsize=3)
            loss_ax1.legend()
            loss_ax1.set_title('Meta-Learnig & Evaluation Curves')
            loss_ax1.set_ylabel('Meta-Loss')
            loss_ax1.grid(grid)

            acc_ax2.plot(episodes_train_x, train_acc_y, label='Accuracy', linestyle='-', marker='o', color='b', linewidth=1)
            acc_ax2.errorbar(episodes_eval_x, eval_acc_y, yerr=eval_acc_std, label=f'{eval_label} Accuracy', linestyle='-', marker='o', color='c', linewidth=1, capsize=3)
            acc_ax2.legend()
            acc_ax2.set_xlabel('Episodes (Outer Epochs)')
            acc_ax2.set_ylabel('Meta-Accuracy')
            acc_ax2.grid(grid)

            plt.tight_layout()

            #plt.show() if show else None

            fig.savefig(current_logs_path / 'meta_train_eval.svg' )
            fig.savefig(current_logs_path / 'meta_train_eval.pdf' )
        elif nb_plots == 2:
            #fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True) 
            #fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
            raise ValueError(f" not implemented nb_plots = {nb_plots}")
        else:
            raise ValueError(f" not implemented nb_plots = {nb_plots}")
        return
