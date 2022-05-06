import os
from argparse import Namespace
from pathlib import Path
from typing import Optional

from uutils import find_free_port, load_cluster_jobids_to, try_to_get_git_revision_hash_short, \
    try_to_get_git_revision_hash_long

from datetime import datetime


def create_default_log_root(args: Namespace):
    """
    Create the default place where we save things to.
    """
    args.log_root: Path = Path('~/data/logs/').expanduser() if not hasattr(args, 'log_root') else args.log_root
    args.log_root: Path = Path(args.log_root).expanduser() if isinstance(args.log_root, str) else args.log_root
    args.log_root: Path = args.log_root.expanduser()
    args.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    args.log_root = args.log_root / f'logs_{args.current_time}_jobid_{args.jobid}'
    args.log_root.mkdir(parents=True, exist_ok=True)


def setup_args_for_experiment(args: Namespace,
                              num_workers: Optional[int] = 0,
                              use_tb: bool = False  # not needed anymore since wandb exists
                              ) -> Namespace:
    """
    Sets up programming details for experiment to run but it should not affect the machine learning results.
    The pretty print & save the args in alphabetically sorted order.

    Note:
        - This function assume the arguments for the experiment have been given already (e.g. through the terminal
        or manually hardcoded in the main/run script).

    Example/Recommended pattern to use:
    1. From terminal (e.g. by running a bash main.sh script that runs python main_experiment.py):
        args = parse_basic_meta_learning_args_from_terminal()
        args = uutils.setup_args_for_experiment(args)
        main(args)
        wandb.finish() if is_lead_worker(args.rank) and args.log_to_wandb else None
        print("Done with experiment, success!\a")

    2. From script (also called, manual or hardcoded)
        args = parse_basic_meta_learning_args_from_terminal()
        args = manual_values_for_run(args)  # e.g. args.batch_size = 64, or ckpt
        args = uutils.setup_args_for_experiment(args)
        main(args)
        wandb.finish() if is_lead_worker(args.rank) and args.log_to_wandb else None
        print("Done with experiment, success!\a")

    Things it sets up:
        - 1. makes sure it has intial index (or use the one provided by user/checkpoint
        - gets rank, port for DDP
        - sets determinism or not
        - device name
        - dirs & filenames for logging
        - cluster job id stuff
        - pid, githash
        - best_val_loss as -infinity
        - wandb be stuff based on your options
    todo:
        - decide where to put annealing, likely in the parse args from terminal and set the default there. Likely
        no annealing as default, or some rate that is done 5 times during entire training time or something like
        that is based on something that tends to work.
    """
    import torch
    import logging
    import uutils
    from uutils.logger import Logger as uuLogger

    # - set the iteration/epoch number to start training from
    if 'iterations' in args.training_mode:
        # set the training iteration to start from beginning or from specified value (e.g. from ckpt iteration index).
        args.it = 0 if not hasattr(args, 'it') else args.it
        assert args.it >= 0, f'Iteration to train has to be start at zero or above but got: {args.it}'
        args.epoch_num = -1
    elif 'epochs' in args.training_mode:
        # set the training epoch number to start from beginning or from specified value e.g. from ckpt epoch_num index.
        args.epoch_num = 0 if not hasattr(args, 'epoch_num') else args.epoch_num
        assert args.epoch_num >= 0, f'Epoch number to train has to be start at zero or above but got: {args.epoch_num}'
        args.it = -1
    elif args.training_mode == 'fit_single_batch' or args.training_mode == 'meta_train_agent_fit_single_batch':
        args.it = 0 if not hasattr(args, 'it') else args.it
        assert args.it >= 0, f'Iteration to train has to be start at zero or above but got: {args.it}'
        args.epoch_num = -1
    else:
        raise ValueError(f'Invalid training mode: {args.training_mode}')
    # - logging frequencies
    if 'iterations' in args.training_mode:
        args.log_freq = 100 if args.log_freq == -1 else args.log_freq
        # similar to epochs, we don't want to anneal more often than what we plot, otherwise it will be harder to
        # see if it was due to the scheduler or not, but when the scheduler is called we might see a dip in the
        # learning curve - like with Qianli's plots
        log_scheduler_freq = 20 * args.log_freq
        ckpt_freq = args.log_freq
    elif 'epochs' in args.training_mode:
        args.log_freq = 1 if args.log_freq == -1 else args.log_freq
        # same as log freq so that if you schedule more often than you log you might miss the scheduler decaying
        # too quickly. It also approximates a scheduler of "on per epoch"
        log_scheduler_freq = 1 * args.log_freq
        ckpt_freq = args.log_freq
    elif args.training_mode == 'fit_single_batch' or args.training_mode == 'meta_train_agent_fit_single_batch':
        args.log_freq = 5 if args.log_freq == -1 else args.log_freq
        log_scheduler_freq = 1
        ckpt_freq = args.log_freq
    else:
        raise ValueError(f'Invalid training mode: {args.training_mode}')
    if hasattr(args, 'log_scheduler_freq'):  # if log_scheduler_freq exists, then replace it by the user or the default
        args.log_scheduler_freq = log_scheduler_freq if args.log_scheduler_freq == -1 else args.log_scheduler_freq
    if hasattr(args, 'ckpt_freq'):  # if ckpt_freq exists, then replace it by the user or the default
        args.ckpt_freq = ckpt_freq if args.ckpt_freq == -1 else args.ckpt_freq

    # - default augment train set
    if not hasattr(args, 'augment_train'):
        args.augment_train = True if not hasattr(args, 'not_augment_train') else not args.not_augment_train

    # NOTE: this should be done outside cuz flags have to be declared first then parsed, args = parser.parse_args()
    if hasattr(args, 'no_validation'):
        args.validation = not args.no_validation

    # - distributed params
    args.rank = -1  # should be written by each worker with their rank, if not we are running serially
    args.master_port = find_free_port()

    # - determinism
    if hasattr(args, 'always_use_deterministic_algorithms'):
        if args.always_use_deterministic_algorithms:
            uutils.torch_uu.make_code_deterministic(args.seed)
            logging.warning(f'Seed being ignored, seed value: {args.seed=}')

    # - get device name
    print(f'{args.seed=}')
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from uutils.torch_uu.distributed import set_devices
    set_devices(args)  # args.device = rank or .device
    print(f'device: {args.device}')

    # - get cluster info (including hostname)
    load_cluster_jobids_to(args)

    # - get log_root
    # usually in options: parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())
    create_default_log_root(args)
    # create tb in log_root
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
        args.tb_dir = args.log_root / 'tb'
        args.tb_dir.mkdir(parents=True, exist_ok=True)
        args.tb = SummaryWriter(log_dir=args.tb_dir)

    # - setup expanded path to checkpoint
    if hasattr(args, 'path_to_checkpoint'):
        # str -> Path(path_to_checkpoint).expand()
        if args.path_to_checkpoint is not None:
            if isinstance(args.path_to_checkpoint, str):
                args.path_to_checkpoint = Path(args.path_to_checkpoint).expanduser()
            elif isinstance(args.path_to_checkpoint, Path):
                args.path_to_checkpoint.expanduser()
            else:
                raise ValueError(f'Path to checkpoint is not of the right type: {type(args.path_to_checkpoint)=},'
                                 f'with value: {args.path_to_checkpoint=}')

    # - get device name if possible
    try:
        args.gpu_name = torch.cuda.get_device_name(0)
    except:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.gpu_name = args.device
    print(f'\nargs.gpu_name = {args.gpu_name}\n')  # print gpu_name if available else cpu

    # - save PID
    args.PID = str(os.getpid())
    if torch.cuda.is_available():
        args.nccl = torch.cuda.nccl.version()

    # - email option (warning, not recommended! very unreliable, especially from cluster to cluster)
    # logging.warning('uutils email is not recommended! very unreliable, especially from cluster to cluster)
    # args.mail_user = 'brando.science@gmail.com'
    # args.pw_path = Path('~/pw_app.config.json').expanduser()

    # - try to get githash, might fail and return error string in field but whatever
    args.githash: str = try_to_get_git_revision_hash_short()
    args.githash_short: str = try_to_get_git_revision_hash_short()
    args.githash_long: str = try_to_get_git_revision_hash_long()

    # - get my logger, its set at the agent level
    args.logger: uuLogger = uutils.logger.Logger(args)

    # - best val loss
    args.best_val_loss: float = float('inf')

    # - wandb
    if hasattr(args, 'log_to_wandb'):
        if not hasattr(args, 'dist_option'):  # backwards compatibility, if no dist_option just setup wandb as "normal"
            setup_wandb(args)
        elif args.dist_option != 'l2l_dist':
            #  in this case wandb has to be setup in the train code, since it's not being ran with ddp, instead the
            # script itself is distributed and pytorch manages it (i.e. pytorch torch.distributed.run manages the
            # mp.spawn or spwaning processes somehow.
            pass
    # - for debugging
    # args.environ = [str(f'{env_var_name}={env_valaue}, ') for env_var_name, env_valaue in os.environ.items()]

    # - to avoid pytorch multiprocessing issues with CUDA: https://pytorch.org/docs/stable/data.html
    args.pin_memory = False
    args.num_workers = num_workers if num_workers is not None else args.num_workers

    # - return
    uutils.print_args(args)
    uutils.save_args(args)
    return args


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
        # - initialize wandb
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   # job_type="job_type",
                   name=run_name,
                   group=args.experiment_name
                   )
        # - save args in wandb
        wandb.config.update(args)
