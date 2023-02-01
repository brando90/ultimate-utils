import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

from uutils import find_free_port, load_cluster_jobids_to, try_to_get_git_revision_hash_short, \
    try_to_get_git_revision_hash_long

from datetime import datetime

from uutils.logging_uu.wandb_logging.common import setup_wandb

from pdb import set_trace as st


def create_default_log_root(args: Namespace):
    """
    Create the default place where we save things to.
    """
    # - make sure prefix $HOME/data/logs/ is in args.log_root Path
    print(f'{args.log_root=}')
    assert 'jobid' not in str(args.log_root), f"jobid should not be in log_root but it is in it {args.log_root=}"
    args.log_root: Path = Path('~/data/logs/').expanduser() if not hasattr(args, 'log_root') else args.log_root
    args.log_root: Path = Path(args.log_root).expanduser() if isinstance(args.log_root, str) else args.log_root
    args.log_root: Path = args.log_root.expanduser()
    assert isinstance(args.log_root, Path), f'Error, it is not of type Path: {args.log_root=}'
    args.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # wandb_str helps to identify which wandb runs are more likely to be important (since they used wandb)
    wandb_str: str = f'_wandb_{args.log_to_wandb}' if hasattr(args, 'log_to_wandb') else 'code_without_wandb'
    args.log_root = args.log_root / f'logs_{args.current_time}_jobid_{args.jobid}_pid_{args.PID}{wandb_str}'
    args.log_root.mkdir(parents=True, exist_ok=True)
    print(f'{args.log_root=}')


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

    # - default "empty" distributed params, for now each dist main sets up their own dist rank e.g. ddp, l2l, etc in their main
    args.rank = -1  # should be written by each worker with their rank, if not we are running serially
    args.master_port = find_free_port()
    # note, currently the true distributed args are set up in the main train file/func todo: perhaps move here? might have to make cases, one for ddp, pytorch mp, one serial, one l2l...worth it?

    # - determinism?
    print(f'Original seed from args: {args.seed=}')
    if hasattr(args, 'always_use_deterministic_algorithms'):
        # this does set the seed but it also does much more e.g. tries to make convs etc deterministic if possible.
        fully_deterministic: bool = False
        if args.always_use_deterministic_algorithms:
            fully_deterministic: bool = uutils.torch_uu.make_code_deterministic(args.seed)
        # todo fix, warn only if code is not fully deterministic
        if not fully_deterministic:
            logging.warning(f'Seed possibly being ignored, seed value: {args.seed=}')
    # - seed only if the user chose a seed, else set a truly different seed. Reason we decided this 1. if we set the seed & make sure it's always different we can always have the code act differently 2. by always acting randomly AND choosing a seed and saving it in args it makes our code (hopefully fully reproducible, but this details about torch might not make it but at least we are trying really hard to be reproducible)
    if args.seed != -1:  # if seed set (-1 is not set), args.seed != -1 means seed not set
        uutils.seed_everything(args.seed)
    else:
        # if seed not seed then set a truly random seed, this should be different even if python is re-ran given it uses the OS
        args.seed = uutils.get_truly_random_seed_through_os(rand_size=3)
    print(f'Seed after code tries to setup args: {args.seed=}')

    # - get device name
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from uutils.torch_uu.distributed import set_devices
    set_devices(args)  # args.device = rank or .device
    print(f'device: {args.device}')

    # - get cluster info (including hostname)
    load_cluster_jobids_to(args)

    # - save PID
    args.PID = str(os.getpid())
    print(f'{args.PID=}')
    if torch.cuda.is_available():
        args.nccl = torch.cuda.nccl.version()

    # - get device name if possible
    try:
        args.gpu_name = torch.cuda.get_device_name(0)
    except:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.gpu_name = args.device
    print(f'\nargs.gpu_name = {args.gpu_name}\n')  # print gpu_name if available else cpu

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

    # -- wandb.  note, nice safety property of my logging when it does log to wandb it always checks rank.
    if hasattr(args, 'log_to_wandb'):
        if not hasattr(args, 'dist_option'):  # backwards compatibility, if no dist_option just setup wandb as "normal"
            # read above 2 comments and one bellow
            setup_wandb(args)
        else:  # hasattr(args, 'dist_option')
            # todo this might be set up cleaner if we setup args in this function and not in main train
            # if custom dist_option needed then we might need more careful starting & setting up wandb e.g. if multiple python processes are used
            if args.dist_option != 'l2l_dist':
                # todo this might be set up cleaner if we setup args in this function and not in main train
                # in this case wandb has to be setup in the train code, since it's not being ran with ddp, instead the
                # script itself is distributed and pytorch manages it i.e. pytorch torch.distributed.run manages the
                # mp.spawn or spwaning processes somehow.
                pass
            elif args.dist_option == 'l2l_dist':
                # todo this might be set up cleaner if we setup args in this function and not in main train
                # setup of wandb is done in main train after args.rank is set up properly,
                pass
        # # - set up wandb
        # # thought this was justified to ignore above since the main (singe) python process sets up wandb once and then the spawned processes always check their rank before logging to wandb
        # # BUT, due to multiple python process thing in the != l2l_dist it might mean this setsup wandb multiple times, so main or train code needs to check rank to avoid this
        # ### WARNING, don't use in this func, see l2l_dist case, setup_wandb(args)  # already checks args.log_to_wandb inside of it

    # - for debugging
    # args.environ = [str(f'{env_var_name}={env_valaue}, ') for env_var_name, env_valaue in os.environ.items()]

    # - to avoid pytorch multiprocessing issues with CUDA: https://pytorch.org/docs/stable/data.html
    args.pin_memory = False
    args.num_workers = num_workers if num_workers is not None else args.num_workers

    # - run re-auth if in stanford cluster
    from socket import gethostname
    if 'stanford' in gethostname():
        print(f'In stanford hostname: {gethostname()=}, about to do stanford reauth in python')
        from uutils import stanford_reauth
        stanford_reauth()
        print(f'finished calling stanford reauth inside python')

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

    # - return
    uutils.print_args(args)
    uutils.save_args(args)
    return args
