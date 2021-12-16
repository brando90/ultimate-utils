from argparse import Namespace

import uutils
from uutils.logging_uu.wandb_logging.common import log_2_wanbd
from uutils.torch_uu.checkpointing_uu.supervised_learning import save_for_supervised_learning
from uutils.torch_uu.distributed import is_lead_worker


def log_train_val_stats(args: Namespace,
                        it: int,

                        train_loss: float,
                        train_acc: float,

                        valid,

                        bar,

                        log_freq: int = 10,
                        ckpt_freq: int = 50,
                        mdl_watch_log_freq: int = 50,
                        force_log: bool = False,  # e.g. at the final it/epoch

                        save_val_ckpt: bool = False,
                        log_to_tb: bool = False,
                        log_to_wandb: bool = False
                        ):
    """
    Log train and val stats where it is iteration or epoch step.

    Note: Unlike save ckpt, this one does need it to be passed explicitly (so it can save it in the stats collector).
    """
    import wandb
    from uutils.torch_uu.tensorboard import log_2_tb_supervisedlearning

    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'

    # if its
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    if (it % log_freq == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        # - get eval stats
        val_loss, val_loss_ci, val_acc, val_acc_ci = valid(args, split='val')
        if float(val_loss - val_loss_ci) < float(args.best_val_loss) and save_val_ckpt:
            args.best_val_loss = float(val_loss)
            save_for_supervised_learning(args, ckpt_filename='ckpt_best_val.pt')

        # - log ckpt
        if it % ckpt_freq == 0:
            save_for_supervised_learning(args, ckpt_filename='ckpt.pt')

        # - save args
        uutils.save_args(args, args_filename='args.json')

        # - update progress bar at the end
        bar.update(it)

        # - print
        args.logger.log('\n')
        args.logger.log(f"{it_or_epoch}={it}: {train_loss=}, {train_acc=}")
        args.logger.log(f"{it_or_epoch}={it}: {val_loss=}, {val_acc=}")

        print(f'{args.it=}')
        print(f'{args.num_its=}')

        # - record into stats collector
        args.logger.record_train_stats_stats_collector(it, train_loss, train_acc)
        args.logger.record_val_stats_stats_collector(it, val_loss, val_acc)
        args.logger.save_experiment_stats_to_json_file()
        args.logger.save_current_plots_and_stats()

        # - log to wandb
        if log_to_wandb:
            log_2_wanbd(it, train_loss, train_acc, val_loss, val_acc, it_or_epoch)

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, it, val_loss, val_acc, 'val')

