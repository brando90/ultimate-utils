from pathlib import Path

import os
import sys
from argparse import Namespace
from typing import Union, Optional

import wandb


def setup_wandb(args: Namespace):
    print("setup called")
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
        # print(f'{sys.stdout=}')
        # print(f'{os.path.realpath(sys.stdout.name)=}')
        # print(f'{sys.stderr=}')
        # print(f'{os.path.realpath(sys.stderr.name)=}')
        # print(f'{sys.stdin=}')
        # print(f'{os.path.realpath(sys.stdin.name)=}')
        wandb.init(
            dir=dir_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            # job_type="job_type",
            name=run_name,
            group=args.experiment_name
        )
        print("init called")
        # - print local run (to be deleted later https://github.com/wandb/wandb/issues/4409)
        try:
            args.wandb_run = wandb.run.dir
            print(f'{wandb.run.dir=}')
        except Exception as e:
            args.wandb_run = 'no wandb run path'
            print(f'{args.wandb_run=}')
        # - save args in wandb
        wandb.config.update(args)


def cleanup_wandb(args: Namespace, delete_wandb_dir: bool = False):
    """

    It might be a bad idea to remove the wandb folder becuase other jobs might be needing it.
    Only safe to delete it if no other job is running.
    """
    from uutils.torch_uu.distributed import is_lead_worker

    if is_lead_worker(args.rank):
        if hasattr(args, 'log_to_wandb'):
            import wandb
            if args.log_to_wandb:
                if hasattr(args, 'rank'):
                    remove_current_wandb_run_dir(call_wandb_finish=True)
                    # remove_wandb_root_dir(args) if delete_wandb_dir else None
            else:
                remove_current_wandb_run_dir(call_wandb_finish=True)
                # remove_wandb_root_dir(args) if delete_wandb_dir else None
    else:
        pass  # nop, your not lead so you shouldn't need to close wandb


def remove_wandb_root_dir(args: Optional[Namespace] = None):
    """
    Remove the main wandb dir that you set in the environment variables e.g. in env variable WANDB_DIR.
    """
    import os
    import shutil
    wandb_dir: Path = Path(os.environ['WANDB_DIR']).expanduser()
    if wandb_dir.exists():
        print(f'deleting wanbd_dir at: WANDB_DIR={wandb_dir}')
        shutil.rmtree(wandb_dir)
        print(f'deletion successful≈ wanbd_dir at: WANDB_DIR={wandb_dir}')

    # delete it


def remove_current_wandb_run_dir(args: Optional[Namespace] = None, call_wandb_finish: bool = True):
    """
    Delete the current wandb run dir.

    note: I think it's better to only delete the current wandb run so that the other wandb runs are unaffected.

    Yep, wandb.run.dir in python only after wandb.init(...) has been called and before wandb.finish() is called. Every run has it's own unique directory so deleting this won't impact other runs. Here's some psuedo code:
    Pseudo code:
        import wandb
        dir_to_delete = None
        wandb.init()
        #...
        dir_to_delete = wandb.run.dir
        wandb.finish()
        if dir_to_delete is not None:
          # delete it
    """
    import wandb
    wandb_dir_to_delete = None
    # wandb.init() this should have been ran already
    print(f'{wandb_dir_to_delete=} {type(wandb_dir_to_delete)=}')
    if call_wandb_finish:
        wandb.finish()  # seem we should finish wandb first before deleting the dir todo; why? https://github.com/wandb/wandb/issues/4409
    if wandb_dir_to_delete is not None:
        wandb_dir_to_delete = wandb.run.dir
        import shutil
        shutil.rmtree(wandb_dir_to_delete)


# delete it


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


def log_2_wanbd_half_loss(it: int,
                train_loss: float, train_acc: float,
                val_loss: float, val_acc: float,
                val_loss_h: float, val_acc_h: float,
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
        wandb.define_metric("val loss half", step_metric=step_metric)
        wandb.define_metric("val acc half", step_metric=step_metric)
        # if wanbd_mdl_watch_log_freq == -1:
        #     wandb.watch(args.base_model, args.criterion, log="all", log_freq=mdl_watch_log_freq)
    # - log to wandb
    wandb.log(data={step_metric: it,
                    'train loss': train_loss,
                    'train acc': train_acc,
                    'val loss': val_loss,
                    'val acc': val_acc,
                    'val loss half': val_loss_h,
                    'val acc half': val_acc_h},
              commit=True)
