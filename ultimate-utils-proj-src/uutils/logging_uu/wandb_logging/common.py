import wandb


def log_2_wanbd(it: int,
                train_loss: float, train_acc: float,
                val_loss: float, val_acc: float,
                step_metric,
                mdl_watch_log_freq: int = -1):
    """

    Ref:
        - custom step: https://community.wandb.ai/t/how-is-one-suppose-to-do-custom-logging-in-wandb-especially-with-the-x-axis/1400
    """
    if it == 0:
        wandb.define_metric("train loss", step_metric=step_metric)
        wandb.define_metric("train acc", step_metric=step_metric)
        wandb.define_metric("val loss", step_metric=step_metric)
        wandb.define_metric("val val", step_metric=step_metric)
        # if wanbd_mdl_watch_log_freq == -1:
        #     wandb.watch(args.base_model, args.criterion, log="all", log_freq=mdl_watch_log_freq)
    # - log to wandb
    wandb.log(data={step_metric: it,
                    'train loss': train_loss,
                    'train acc': train_acc,
                    'val loss': val_loss,
                    'val acc': val_acc},
              commit=True)
