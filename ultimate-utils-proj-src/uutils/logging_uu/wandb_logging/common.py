import os
import sys
from argparse import Namespace
from typing import Union

import wandb


def setup_wandb(args: Namespace):
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
        # set a location of where to save your local wandb stuff
        dir_wandb = None
        if 'WANDB_DIR' in os.environ.keys():
            # dir_wandb = Path('~/tmp/').expanduser()
            # dir_wandb = Path('/shared/rsaas/miranda9/tmp/').expanduser()
            print(f"{os.environ['WANDB_DIR']=}")
            dir_wandb: Union[str, None] = os.environ['WANDB_DIR'] if os.environ['WANDB_DIR'] else None
        # if hasattr(args, 'dir_wandb'):
        #     # if user forces where to save
        #     dir_wandb = args.dir_wandb
        # else:
        #     args.dir_wandb: Path = args.log_root.expanduser()
        #     dir_wandb = args.dir_wandb
        # - initialize wandb
        print('-- info about wandb setup (info meant to be here for now, when ViT runs maybe we\'remove it)')
        print(f'{dir_wandb=}')
        print(f'{sys.stdout=}')
        print(f'{os.path.realpath(sys.stdout.name)=}')
        print(f'{sys.stderr=}')
        print(f'{os.path.realpath(sys.stderr.name)=}')
        print(f'{sys.stdin=}')
        print(f'{os.path.realpath(sys.stdin.name)=}')
        wandb.init(
            dir=dir_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            # job_type="job_type",
            name=run_name,
            group=args.experiment_name
        )
        # - save args in wandb
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
