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
from uutils.torch_uu.eval.eval import do_eval

from pdb import set_trace as st

from uutils.torch_uu.training.common import get_data


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
                        training: bool = False,  # false for SL see meta: https://stats.stackexchange.com/a/551153/28986
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
            print(f'---- <start> printing logging info for {step=}')
            print(f'{save_val_ckpt=}')
            print(f'{uu_logger_log=}')
            print(f'{log_to_wandb=}')
            print(f'{log_to_tb=}')
            print(f'{sys.stdout=}')
            print(f'{os.path.realpath(sys.stdout.name)=}')
            print(f'---- <end> printing logging info for {step=}')
            print()

        # - compute val stats for logging & determining if to ckpt val model
        print('1')
        val_loss, val_loss_ci, val_acc, val_acc_ci = do_eval(args, args.agent, args.dataloaders, training=training)

        # - print
        print('2')
        args.logger.log('\n')
        args.logger.log(f"-> {step_name}={step}: {train_loss=}, {train_acc=}")
        args.logger.log(f"-> {step_name}={step}: {val_loss=}, {val_acc=}")

        # - get eval stats
        print('3')
        if float(val_loss - val_loss_ci) < float(args.best_val_loss) and save_val_ckpt:
            args.best_val_loss = float(val_loss)
            # if train_loss < 0.5: after 0.5, the loss has decreased enough to make this worth it. TODO: put loss value once you know lowest train loss FMs get
            if step >= 20 * ckpt_freq:  # saving ckpt is expensive and at the beginning val will keep decreasing, so this hack so that a lot of training has happening, alternative we could do train loss < 0.2
                save_for_supervised_learning(args, ckpt_filename='ckpt_best_val.pt')

        # - log ckpt, note: ckpt_freq = getattr(args, 'ckpt_freq', args.log_freq)
        print('4')
        if step % ckpt_freq == 0:
            save_for_supervised_learning(args, ckpt_filename='ckpt.pt')
        if hasattr(args, 'smart_logging_ckpt'):
            # e.g. logging more often after train acc is high e.g. 0.9 & save ckpt with all losses in name ckpt filename
            smart_logging_ckpt(args, step, step_name, train_loss, train_acc, val_loss, val_acc, ckpt_freq)

        # - save args
        print('5')
        uutils.save_args(args, args_filename='args.json')
        # save_args_as_dict_in_pickle_file(args, args_filename='args.pt')

        # - update progress bar at the end
        if bar is not None:
            bar.update(step)

        # - record into stats collector
        if uu_logger_log:
            args.logger.record_train_stats_stats_collector(step, train_loss, train_acc)
            args.logger.record_val_stats_stats_collector(step, val_loss, val_acc)
            args.logger.save_experiment_stats_to_json_file()
            args.logger.save_current_plots_and_stats()

        # - log to wandb
        print_dist(msg=f'{log_to_wandb=} (if True then it should displaying wanbd info & using it)', rank=args.rank,
                   flush=True)
        if log_to_wandb:
            log_2_wanbd(step, train_loss, train_acc, val_loss, val_acc, step_name)

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, step, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, step, val_loss, val_acc, 'val')


def log_zeroth_step(args: Namespace, model: Agent, split: str = 'train', training: bool = True) -> tuple:
    """
    Do the zeroth step before model has been changed during training.

    Note:
    - this is another way to get the loss & acc:
        data: Any = get_data(dataloaders, split)
        losses, accs = model.get_lists_accs_losses(data, training)
        loss = torch.stack(losses).mean()
        acc = torch.stack(accs).mean()
    """
    batch: Any = get_data(args.dataloaders, split=split)
    print(f'==> {batch=}')
    train_loss, train_acc = model(batch, training=training)
    print(f'==> {train_loss=}, {train_acc=}')
    step_name: str = 'epoch_num' if 'epochs' in args.training_mode else 'it'
    log_train_val_stats(args, 0, step_name, train_loss, train_acc)
    return train_loss, train_acc


def smart_logging_ckpt(args,
                       step, step_name,
                       train_loss, train_acc,
                       val_loss, val_acc,
                       ckpt_freq,  # note: ckpt_freq = getattr(args, 'ckpt_freq', args.log_freq)
                       ) -> None:
    """
    Do some sort of smarter logging. e.g. log more often after train acc is high e.g. 0.9 & save ckpt with all losses in name ckpt filename.

    Recommended use:
        - log_more_often_after_threshold_is_reached: is useful for when you want to log ckpt more often after train acc is high
        e.g. 0.9 & save ckpt with all losses in name ckpt filename. e.g. when your model is large enough and can actually
        reach such accuracy.
            e.g. flag args in congig function:
                args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached', metric_to_use='train_acc',
                          threshold=0.9, log_speed_up=10)
        - log_more_often_after_convg_reached: is useful when you want to cjpt

    Note:
        - see if statement for currently implemented options
    """
    if hasattr(args, 'smart_logging_ckpt'):  # extra safety for backwards compatability with old args
        smart_logging_type: str = args.smart_logging_ckpt['smart_logging_type']
        if smart_logging_type == 'log_more_often_after_threshold_is_reached':
            log_more_often_after_threshold_is_reached(args, step, step_name, train_loss, train_acc, val_loss, val_acc,
                                                      ckpt_freq)
        elif smart_logging_type == 'log_more_often_after_convg_reached':
            log_more_often_after_convg_reached(args, step, step_name, train_loss, train_acc, val_loss, val_acc,
                                               ckpt_freq)
        else:
            raise NotImplementedError(f'{smart_logging_type=}')
    else:
        # likely, args are set to old code or this flag is not set, so do nothing
        pass
    return


def log_more_often_after_threshold_is_reached(args,
                                              step, step_name,
                                              train_loss, train_acc,
                                              val_loss, val_acc,
                                              ckpt_freq,  # note: ckpt_freq = getattr(args, 'ckpt_freq', args.log_freq)
                                              ) -> None:
    """
    Logs more often after a threshold has been reached for a metric e.g. train_acc >= 0.9, then it will log more
    frequently e.g. 10 times more often (500 => 50).

    note:
        - "\\" floor division operator, e.g. 5 // 2 = 2
    """
    # - get args for logging more often after threshold is reached
    metric_to_use: str = args.smart_logging_ckpt['metric_to_use']  # e.g. train_loss or train_acc
    threshold: float = args.smart_logging_ckpt['threshold']  # e.g 0.1 or 0.9
    log_speed_up: int = args.smart_logging_ckpt['log_speed_up']  # e.g. 2 or 5  or 10 or 50 or 100
    log_freq: int = max(ckpt_freq // log_speed_up, 1)  # e.g. my usual value 500 then divde it by log_speed_up e.g. 10
    # - do smart logging according to logging more often after threshold is reached
    args.log_more_often_on = False if not hasattr(args, 'log_more_often_on') else args.log_more_often_on
    # - log more often after threshold is reached
    if metric_to_use == 'train_loss':
        cond: bool = train_loss <= threshold or args.log_more_often_on
        # this is more complicated than I wished but I think we should record more often since loss/accs for meta-learning have high variance
        args.log_more_often_on = True if cond else False  # approx train_acc >= threshold + going forward leave it on
        log_more_often_accordin_to_bool(args, step, step_name, train_loss, train_acc, val_loss, val_acc, log_freq, cond)
    elif metric_to_use == 'train_acc':
        cond: bool = train_acc >= threshold or args.log_more_often_on
        # this is more complicated than I wished but I think we should record more often since loss/accs for meta-learning have high variance
        args.log_more_often_on = True if cond else False  # approx train_acc >= threshold + going forward leave it on
        log_more_often_accordin_to_bool(args, step, step_name, train_loss, train_acc, val_loss, val_acc, log_freq, cond)
    elif metric_to_use == 'val_loss':
        cond: bool = val_loss <= threshold or args.log_more_often_on
        # this is more complicated than I wished but I think we should record more often since loss/accs for meta-learning have high variance
        args.log_more_often_on = True if cond else False  # approx train_acc >= threshold + going forward leave it on
        log_more_often_accordin_to_bool(args, step, step_name, train_loss, train_acc, val_loss, val_acc, log_freq, cond)
    elif metric_to_use == 'val_acc':
        cond: bool = val_acc >= threshold or args.log_more_often_on
        # this is more complicated than I wished but I think we should record more often since loss/accs for meta-learning have high variance
        args.log_more_often_on = True if cond else False  # approx train_acc >= threshold + going forward leave it on
        log_more_often_accordin_to_bool(args, step, step_name, train_loss, train_acc, val_loss, val_acc, log_freq, cond)
    else:
        raise NotImplementedError(f'{metric_to_use=}')
    return


def log_more_often_after_convg_reached(args,
                                       step, step_name,
                                       train_loss, train_acc,
                                       val_loss, val_acc,
                                       ckpt_freq,  # note: ckpt_freq = getattr(args, 'ckpt_freq', args.log_freq)
                                       ) -> None:
    """
    Logs more often after convergence has been reached for a metric e.g. train_acc >= 0.9, then it will log more.

    note:
        - still need metric to use since you're checking convg wrt train_acc or train_loss.
        Note: idk what it means to conv often wrt val loss. However, the convg meter implementation works if it stops
        improving, so using val statistics means your doing early stopping which is fine. Might need to write a seperate
        function to make this more clear/explicit.
    """
    # - get args for logging more often after convergence is reached
    metric_to_use: str = args.smart_logging_ckpt['metric_to_use']  # e.g. train_loss or train_acc
    log_speed_up: int = args.smart_logging_ckpt['log_speed_up']  # e.g. 2 or 5  or 10 or 50 or 100
    log_freq: int = max(ckpt_freq // log_speed_up, 1)  # e.g. my usual value 500 then divde it by log_speed_up e.g. 10
    # - do smart logging according to logging more often after convergence is reached
    args.log_more_often_on = False if not hasattr(args, 'log_more_often_on') else args.log_more_often_on
    # - log more often after threshold is reached
    # note other ones have not been implemented since train loops would need to log the right metric to the convg meter, which is not done yet
    if metric_to_use == 'train_loss':
        cond: bool = args.convg_meter.check_converged() or args.log_more_often_on
        # this is more complicated than I wished but I think we should record more often since loss/accs for meta-learning have high variance
        args.log_more_often_on = True if cond else False  # approx conv + going forward leave it on
        log_more_often_accordin_to_bool(args, step, step_name, train_loss, train_acc, val_loss, val_acc, log_freq, cond)
    elif metric_to_use == 'train_acc':
        raise NotImplementedError(f'{metric_to_use=}')
        # condition: bool = train_acc >= threshold or args.log_more_often_on
        # # this is more complicated than I wished but I think we should record more often since loss/accs for meta-learning have high variance
        # args.log_more_often_on = True
    elif metric_to_use == 'val_loss':
        raise NotImplementedError(f'{metric_to_use=}')
    elif metric_to_use == 'val_acc':
        raise NotImplementedError(f'{metric_to_use=}')
    else:
        raise NotImplementedError(f'{metric_to_use=}')
    return


def log_more_often_accordin_to_bool(args,
                                    step, step_name,
                                    train_loss, train_acc,
                                    val_loss, val_acc,
                                    log_freq,  # usually, log_freq: int = max(ckpt_freq // log_speed_up, 1)
                                    condition: bool,  # e.g. train_acc >= 0.9 or convg
                                    ) -> None:
    """ Logs more often according to a boolean condition e.g. train_acc >= 0.9 or convg, then it will log more."""
    if condition:
        if step % log_freq == 0:
            ckpt_filename: str = get_more_often_ckpting_filename(args, step, step_name, train_loss, train_acc,
                                                                 val_loss, val_acc)
            save_for_supervised_learning(args, ckpt_filename=ckpt_filename)
    return


def get_more_often_ckpting_filename(args,
                                    step, step_name,
                                    train_loss, train_acc,
                                    val_loss, val_acc,
                                    ) -> str:
    ckpt_filename: str = f'ckpt_{step_name}_{step}_train_loss_{train_loss:.3f}_train_acc_{train_acc:.3f}_val_loss_{val_loss:.3f}_val_acc_{val_acc:.3f}.pt'
    return ckpt_filename


def save_args_as_dict_in_pickle_file(args: Namespace, args_filename: str = 'args.pt') -> dict:
    """Saves args as dict in pickle file. Returns new dict without keys that might give issues"""
    import pickle
    args_dict: dict = vars(args)
    # del key "agent" "model" "opt" "scheduler"
    for key in ['agent', 'model', 'opt', 'scheduler', 'meta_learner']:
        if key in args_dict:
            del args_dict[key]
    pickle.dump(args_dict, open(args.log_root / args_filename, 'wb'))


# - tests, tutorials, examples

def log_zero_test_():
    # - usl
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.argparse_uu.supervised_learning import get_args_mi_usl_default
    from uutils.torch_uu.agents.supervised_learning import ClassificationSLAgent
    args: Namespace = get_args_mi_usl_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = ClassificationSLAgent(args, args.model)
    from uutils.torch_uu.dataloaders.helpers import get_sl_dataloader
    args.dataloaders = get_sl_dataloader(args)
    train_loss, train_acc = log_zeroth_step(args, args.agent)
    print(f'{train_loss, train_acc=}')
    # - torchmeta
    from uutils.argparse_uu.meta_learning import get_args_mi_torchmeta_default
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearner
    args: Namespace = get_args_mi_torchmeta_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = MAMLMetaLearner(args, args.model)
    from uutils.torch_uu.dataloaders.meta_learning.helpers import get_meta_learning_dataloaders
    args.dataloaders = get_meta_learning_dataloaders(args)
    train_loss, train_acc = log_zeroth_step(args, args.agent)
    print(f'{train_loss, train_acc=}')
    # - l2l
    from uutils.argparse_uu.meta_learning import get_args_mi_l2l_default
    from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    from uutils.torch_uu.mains.common import get_and_create_model_opt_scheduler_for_run
    from uutils.torch_uu.meta_learners.maml_meta_learner import MAMLMetaLearnerL2L
    args: Namespace = get_args_mi_l2l_default()
    get_and_create_model_opt_scheduler_for_run(args)
    args.agent = MAMLMetaLearnerL2L(args, args.model)
    args.dataloaders = get_l2l_tasksets(args)
    train_loss, train_acc = log_zeroth_step(args, args.agent)
    print(f'{train_loss, train_acc=}')


# - run __main__

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    log_zero_test_()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
