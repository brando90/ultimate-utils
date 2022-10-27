import os
import sys
from argparse import Namespace
from typing import Callable, Any

from progressbar import ProgressBar

import uutils
from uutils.logging_uu.wandb_logging.common import log_2_wanbd
from uutils.torch_uu.agents.common import Agent
from uutils.torch_uu.checkpointing_uu.supervised_learning import save_for_supervised_learning
from uutils.torch_uu.distributed import is_lead_worker, print_dist
from uutils.torch_uu.eval.eval import eval_sl

from pdb import set_trace as st


def log_train_val_stats_simple(args: Namespace,
                               it: int, train_loss: float, train_acc: float, bar: ProgressBar,
                               save_val_ckpt: bool = True, force_log: bool = False):
    if is_lead_worker(args.rank):
        # - get eval stats
        val_batch: Any = next(iter(args.dataloaders['val']))
        val_loss, val_loss_ci, val_acc, val_acc_ci = args.agent.eval_forward(val_batch)
        if float(val_loss - val_loss_ci) < float(args.best_val_loss) and save_val_ckpt:
            args.best_val_loss = float(val_loss)
            save_for_supervised_learning(args, ckpt_filename='ckpt_best_val.pt')

        # - log ckpt
        if it % 10 == 0 or force_log:
            save_for_supervised_learning(args, ckpt_filename='ckpt.pt')

        # - save args
        uutils.save_args(args, args_filename='args.json')

        # - update progress bar at the end
        bar.update(it)

        # - print
        print_dist(f"\n{it=}: {train_loss=} {train_acc=}", args.rank)
        print_dist(f"{it=}: {val_loss=} {val_acc=}", args.rank)

        # - for now no wandb for logging for one batch...perhaps change later
        pass


def log_train_val_stats(args: Namespace,
                        step: int,
                        step_name: str,
                        train_loss: float,
                        train_acc: float,

                        training: bool = False,  # default for SL

                        save_val_ckpt: bool = True,
                        ):
    _log_train_val_stats(args=args,
                         step=step,
                         step_name=step_name,
                         train_loss=train_loss,
                         train_acc=train_acc,

                         bar=args.bar,

                         ckpt_freq=getattr(args, 'ckpt_freq', args.log_freq),

                         training=training,

                         save_val_ckpt=save_val_ckpt,
                         log_to_tb=getattr(args, 'log_to_tb', False),
                         log_to_wandb=getattr(args, 'log_to_wandb', False),
                         uu_logger_log=getattr(args, 'log_to_wandb', False)
                         )


def _log_train_val_stats(args: Namespace,
                         step: int,
                         step_name: str,
                         train_loss: float,
                         train_acc: float,

                         bar: ProgressBar,

                         ckpt_freq: int,

                         training: bool,

                         save_val_ckpt: bool = True,
                         log_to_tb: bool = False,
                         log_to_wandb: bool = False,
                         uu_logger_log: bool = False,
                         ):
    """
    Log train and val stats every step (where step is .epoch_num or .it)
    """
    if is_lead_worker(args.rank):
        from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning
        # - print what flags are on
        if step == 0:
            print(f'{save_val_ckpt=}')
            print(f'{uu_logger_log=}')
            print(f'{log_to_wandb=}')
            print(f'{log_to_tb=}')
            print(f'{sys.stdout=}')
            print(f'{os.path.realpath(sys.stdout.name)=}')

        # - get eval stats
        val_loss, val_loss_ci, val_acc, val_acc_ci = eval_sl(args, args.agent, args.dataloaders, training=training)
        if float(val_loss - val_loss_ci) < float(args.best_val_loss) and save_val_ckpt:
            args.best_val_loss = float(val_loss)
            # if train_loss < 0.5: after 0.5, the loss has decreased enough to make this worth it.
            if step >= 20 * ckpt_freq:  # saving ckpt is expensive and at the beginning val will keep decreasing, so this hack so that a lot of training has happening, alternative we could do train loss < 0.2
                save_for_supervised_learning(args, ckpt_filename='ckpt_best_val.pt')

        # - log ckpt
        if step % ckpt_freq == 0:
            save_for_supervised_learning(args, ckpt_filename='ckpt.pt')

        # - save args
        uutils.save_args(args, args_filename='args.json')

        # - update progress bar at the end
        if bar is not None:
            bar.update(step)

        # - print, todo: move before checkpointing
        args.logger.log('\n')
        args.logger.log(f"-> {step_name}={step}: {train_loss=}, {train_acc=}")
        args.logger.log(f"-> {step_name}={step}: {val_loss=}, {val_acc=}")

        # - record into stats collector

        if uu_logger_log:
            args.logger.record_train_stats_stats_collector(step, train_loss, train_acc)
            args.logger.record_val_stats_stats_collector(step, val_loss, val_acc)
            args.logger.save_experiment_stats_to_json_file()
            args.logger.save_current_plots_and_stats()

        # - log to wandb
        if log_to_wandb:
            log_2_wanbd(step, train_loss, train_acc, val_loss, val_acc, step_name)

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, step, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, step, val_loss, val_acc, 'val')


def log_zeroth_step(args: Namespace, model: Agent):
    batch: Any = next(iter(args.dataloaders['train']))
    train_loss, train_acc = model(batch, training=True)
    step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
    log_train_val_stats(args, 0, step_name, train_loss, train_acc)
