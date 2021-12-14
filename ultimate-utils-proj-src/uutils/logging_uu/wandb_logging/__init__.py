
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
    from pprint import pprint
    # - is it epoch or iteration
    it_or_epoch: str = 'epoch_num' if args.training_mode == 'epochs' else 'it'

    # if its
    total_its: int = args.num_empochs if args.training_mode == 'epochs' else args.num_its

    if (it % log_freq == 0 or it == total_its - 1 or force_log) and is_lead_worker(args.rank):
        # - get eval stats
        val_loss, val_acc, val_loss_std, val_acc_std = valid(args, save_val_ckpt=save_val_ckpt)
        # - log ckpt
        if it % ckpt_freq == 0:
            save_for_meta_learning(args)

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
            if it == 0:
                wandb.define_metric("train loss", step_metric=it_or_epoch)
                wandb.define_metric("train acc", step_metric=it_or_epoch)
                wandb.define_metric("val loss", step_metric=it_or_epoch)
                wandb.define_metric("val val", step_metric=it_or_epoch)
                # if mdl_watch_log_freq == -1:
                #     wandb.watch(args.base_model, args.criterion, log="all", log_freq=mdl_watch_log_freq)
            # - log to wandb
            wandb.log(data={it_or_epoch: it,
                            # custom step: https://community.wandb.ai/t/how-is-one-suppose-to-do-custom-logging-in-wandb-especially-with-the-x-axis/1400
                            'train loss': train_loss,
                            'train acc': train_acc,
                            'val loss': val_loss,
                            'val acc': val_acc},
                      commit=True)
            # if it == total_its:  # not needed here, only needed for normal SL training
            #     wandb.finish()

        # - log to tensorboard
        if log_to_tb:
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'train')
            log_2_tb_supervisedlearning(args.tb, args, it, train_loss, train_acc, 'val')
