from pathlib import Path

import os
import sys
from argparse import Namespace
from typing import Union, Optional

import wandb


def setup_wandb(args: Namespace):
    args.wandb_run_url = None
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
        # print('-- info about wandb setup (info meant to be here for now, when ViT runs maybe we\'remove it)')
        # print(f'{dir_wandb=}')
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
        # - print local run (to be deleted later https://github.com/wandb/wandb/issues/4409)
        print_wanbd_run_info(args)
        # - save args in wandb
        wandb.config.update(args)
        # - save wandb run url in args
        print(f'{try_printing_wandb_url(args.log_to_wandb)=}')
        args.wandb_run_url: str = try_printing_wandb_url(args.log_to_wandb)


def print_wanbd_run_info(args: Namespace):
    try:
        args.wandb_run = wandb.run.dir
        args.wandb_run_url = wandb.run.get_url()
        print(f'{wandb.run.dir=}')
        print(f'{args.wanbd_url=}')
    except Exception as e:
        args.wandb_run = 'no wandb run path (yet?)'
        args.wandb_run_url = 'no wandb run url (yet?)'
        print(f'{wandb.run.dir=}')
        print(f'{args.wandb_run_url=}')


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
                print(f'{try_printing_wandb_url(args.log_to_wandb)=}')
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


def log_2_wandb(it: int,
                train_loss: float, train_acc: float,
                val_loss: float, val_acc: float,
                step_metric,
                mdl_watch_log_freq: int = 500,
                ):
    """

    Ref:
        - custom step: https://community.wandb.ai/t/how-is-one-suppose-to-do-custom-logging-in-wandb-especially-with-the-x-axis/1400
    """
    if it == 0:
        wandb.define_metric("train loss", step_metric=step_metric)
        wandb.define_metric("train acc", step_metric=step_metric)
        wandb.define_metric("val loss", step_metric=step_metric)
        wandb.define_metric("val val", step_metric=step_metric)
    # - log to wandb
    wandb.log(data={step_metric: it,
                    'train loss': train_loss,
                    'train acc': train_acc,
                    'val loss': val_loss,
                    'val acc': val_acc},
              commit=True)
    # - print to make explicit in console/terminal it's using wandb
    # note usually this func is callaed with a args.rank check outside, so using print vs print_dist should be fine
    try_printing_wandb_url(log_to_wandb=True)


def try_printing_wandb_url(log_to_wandb: bool = False) -> str:
    """
    Try to print the wandb url and return it as a string if it succeeds.
    If it fails, return the error message as a string.
    """
    if log_to_wandb:
        try:
            # print(f'{wandb.run.dir=}')
            print(f'{wandb.run.get_url()=}')
            print(_get_sweep_url_hardcoded())
            print(f'{wandb.get_sweep_url()=}')
            return str(wandb.run.get_url())
        except Exception as e:
            err_msg: str = f'Error from wandb url get {try_printing_wandb_url=}: {e=}'
            print(err_msg)
            import logging
            logging.warning(err_msg)
            return str(e)


def hook_wandb_watch_model(args: Namespace,
                           model,  # could be model or agent
                           mdl_watch_log_freq: int = 500,
                           log: str = 'all',  # ['gradients', 'parameters', 'all']
                           ):
    """
    Hook wandb.watch to the model.

    ref:
        - docs for wandb.watch: https://docs.wandb.ai/ref/python/watch
        - 5 min youtube: https://www.youtube.com/watch?v=k6p-gqxJfP4
    """
    # - if model is None do nothing
    if model is None:
        return
    # - for now default to watch model same as logging loss so use args.log_freq, later perhaps customize in argparse as a flag
    mdl_watch_log_freq: int = args.log_freq if hasattr(args, 'log_freq') else mdl_watch_log_freq
    # mdl_watch_log_freq: int = args.log_freq if hasattr(args, 'mdl_watch_log_freq') else mdl_watch_log_freq
    # - get model if it's agent
    if hasattr(model, 'model'):  # if it's an agent then this is how you get the model
        model = model.model
    # - get loss/criterion (for now decided to use loss everyone instead of criterion
    loss = None
    if hasattr(args, 'loss'):
        loss = args.loss
    # give priority to the one the model has
    if hasattr(model, 'loss'):
        loss = model.loss
    # - watch model
    log: str = 'all' if log is None else log
    wandb.watch(model, loss, log=log, log_freq=mdl_watch_log_freq)


def _get_sweep_url_hardcoded(entity: str, project: str, sweep_id) -> str:
    """
    ref:
        - hoping for an official answer here:
            - SO: https://stackoverflow.com/questions/75852199/how-do-i-print-the-wandb-sweep-url-in-python
            - wandb discuss: https://community.wandb.ai/t/how-do-i-print-the-wandb-sweep-url-in-python/4133
    """
    return f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"

def watch_activations():
    """
    ref:
        - https://community.wandb.ai/t/how-to-watch-the-activations-of-a-model/4101, https://github.com/wandb/wandb/issues/5218
    """
    pass


def watch_update_step():
    """
    ref:
        - https://community.wandb.ai/t/how-to-watch-the-activations-of-a-model/4101, https://github.com/wandb/wandb/issues/5218
    """
    pass
