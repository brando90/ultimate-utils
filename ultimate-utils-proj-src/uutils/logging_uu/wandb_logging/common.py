from argparse import Namespace

import wandb


def setup_wand(args: Namespace):
    if hasattr(args, 'log_to_wandb'):  # this is here again on purpose to be extra safe
        if args.log_to_wandb:
            # os.environ['WANDB_MODE'] = 'offline'
            import wandb
            print(f'{wandb=}')

            # - set run name
            run_name = None
            # if in cluster use the cluster jobid
            if hasattr(args, 'jobid'):
                # if jobid is actually set to something, use that as the run name in ui
                if args.jobid is not None and args.jobid != -1 and str(args.jobid) != '-1':
                    run_name: str = f'jobid={str(args.jobid)}'
            # if user gives run_name overwrite that always
            if hasattr(args, 'run_name'):
                run_name = args.run_name if args.run_name is not None else run_name
            args.run_name = run_name
            # - initialize wandb
            wandb.init(project=args.wandb_project,
                       entity=args.wandb_entity,
                       # job_type="job_type",
                       name=run_name,
                       group=args.experiment_name
                       )
            wandb.config.update(args)


def cleanup_wandb(args: Namespace):
    from uutils.torch_uu.distributed import is_lead_worker

    if hasattr(args, 'log_to_wandb'):
        import wandb
        if args.log_to_wandb:
            if hasattr(args, 'rank'):
                if is_lead_worker(args.rank):
                    wandb.finish()
                else:
                    pass  # nop, your not lead so you shouldn't need to close wandb
        else:
            wandb.finish()


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
