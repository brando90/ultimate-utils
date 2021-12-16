from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from uutils import load_cluster_jobids_to, load_json_path2file, merge_args


def parse_args_standard_sl() -> Namespace:
    import argparse

    # - great terminal argument parser
    parser = argparse.ArgumentParser()

    # -- create argument options
    parser.add_argument('--debug', action='store_true', help='if debug')
    # parser.add_argument('--serial', action='store_true', help='if running serially')
    parser.add_argument('--parallel', action='store_true', help='if running in parallel')

    # - data set & dataloader options
    parser.add_argument('--split', type=str, default='train', help="possiboe values: "
                                                                   "'train', val', test'")
    parser.add_argument('--data_set_path', type=str, default=Path('~/data/mnist/').expanduser(),
                        help='path to data set splits. The code will assume everything is saved in'
                             'the uutils standard place in ~/data/, ~/data/logs, etc. see the setup args'
                             'setup method and log_root.')
    parser.add_argument('--log_root', type=str, default=Path('/logs/').expanduser())

    # - training options
    parser.add_argument('--training_mode', type=str, default='iterations', help='valid values: '
                                                                                'iterations,'
                                                                                'epochs, '
                                                                                'train_until_perfect_train_accuracy, '
                                                                                'fit_single_batch, '
                                                                                'fit_until_convergence'
                        )
    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    # parser.add_argument('--no_validation', action='store_true', help='no validation is performed')

    # model & loss function options
    parser.add_argument('--model_type', type=str, help='Options:')
    parser.add_argument('--loss', type=str, help='loss/criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--optimizer_option', type=str, default='Adafactor.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Warning: use a learning rate according to'
                                                                          'how previous work trains your model.'
                                                                          'Otherwise, tuning might be needed.'
                                                                          'Vision resnets usually use 1e-3'
                                                                          'and transformers have a smaller'
                                                                          'learning 1e-4 or 1e-5.'
                                                                          'It might be a good start to have the'
                                                                          'Adafactor optimizer with lr=None and'
                                                                          'a its defualt scheduler called'
                                                                          'every epoch or every '
                                                                          '1(1-beta_2)^-1=2000 iterations.'
                                                                          'Doing a hp search with with wanbd'
                                                                          'a good idea.')
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--scheduler_option', type=str, default='AdafactorSchedule', help='Its strongly recommended')
    # parser.add_argument('--l2', type=float, default=0.0)
    # parser.add_argument('--lr_reduce_steps', default=3, type=int,
    #                     help='the number of steps before reducing the learning rate \
    #                     (only applicable when no_validation == True)')
    # parser.add_argument('--lr_reduce_patience', type=int, default=10)

    # - checkpoint options
    parser.add_argument('--path_to_checkpoint', type=str, default=None, help='the path to the model checkpoint to '
                                                                             'resume training.'
                                                                             'e.g. path: '
                                                                             '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668/ckpt.pt')

    # - miscellaneous arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully deterministic')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='the number of data_lib-loading threads (when running serially')
    parser.add_argument('--args_hardcoded_in_script', action='store_true', default=False,
                        help='If part of the args are '
                             'hardcoded in the python script.')

    # - parse arguments
    args: Namespace = parser.parse_args()
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)
    return args


def make_args_from_supervised_learning_checkpoint(args: Namespace,
                                                  precedence_to_args_checkpoint: bool = True,
                                                  ) -> Namespace:
    """
    Get args from supervised learning and merge it with current args. Default precedence is the checkpoint args
    since you want to keep training from the checkpoint value.

    Note:
        - model, opt, scheduler objs won't be present but will be loaded later.
        - To create a new path to checkpoint the checkpointed model this code overwrites
    """
    ckpt: dict = torch.load(args.path_to_checkpoint, map_location=torch.device('cpu'))
    # ckpt: dict = torch.load(args.path_to_checkpoint, map_location=args.device)
    args_dict: dict = ckpt['args_dict']
    del ckpt  # since ckpt might have lots of data from previous training, not needed her when recovering only args
    args_ckpt: Namespace = Namespace(**args_dict)

    # - updater args has precedence.
    if precedence_to_args_checkpoint:
        args: Namespace = merge_args(starting_args=args, updater_args=args_ckpt)
    else:
        args: Namespace = merge_args(starting_args=args_ckpt, updater_args=args)

    # - create a correct path to save the new model that will be checkpointe and not break the previous checkpoint
    from uutils.argparse_uu import create_default_log_root
    create_default_log_root()  # creates a new log root with the current job number etc
    return args


def make_args_from_supervised_learning_checkpoint_from_args_json_file(args: Namespace,
                                                                      path2args_file: str,
                                                                      precedence_to_args_checkpoint: bool = True,
                                                                      ) -> Namespace:
    """
    Get args from supervised learning and merge it with current args. Default precedence is the checkpoint args
    since you want to keep training from the checkpoint value.

    READ: if needed later, load checkpoint from args.json, process it to make sure it works then use it.
    The meta-learning one worked so this one should hopefully work too with the same preprocessing ow. extend it,
    likely not needed.

    Note:
        - you could get the it or epoch_num from the ckpt pickle file here too and set it in args but I decided
        to not do this here and do it when processing the actualy pickle file to seperate what the code is doing
        and make it less confusing (in one place you process the args.json and the other you process the pickled
        file with the actual model, opt, meta_learner etc).

    :param args:
    :param path2args_file: path & filneame e.g. path/args.json
    :param precedence_to_args_checkpoint:
    """
    # path2args: Path = Path(path2args_file).expanduser() if isinstance(path2args_file,
    #                                                                   str) else path2args_file.expanduser()
    # args_ckpt: Namespace = load_json_path2file(path2args=path2args)
    # args_ckpt: Namespace = map_args_fields_from_string_to_usable_value(args_ckpt)
    # # args.num_its = args.train_iters
    # # - for forward compatibility, but ideally getting the args and the checkpoint will be all in one place in the future
    # args.training_mode = 'iterations'
    # args.it = it
    # # - updater args has precedence
    # if precedence_to_args_checkpoint:
    #     args: Namespace = merge_args(starting_args=args, updater_args=args_ckpt)
    # else:
    #     args: Namespace = merge_args(starting_args=args_ckpt, updater_args=args)
    # # - always overwrite the path to the checkpoint with the one given by the user
    # # (since relitive paths aren't saved properly since they are usually saved as expanded paths)
    # args.path_to_checkpoint: Path = path2args
    # args.log_root: Path = Path('~/data/logs/')
    # return args
    raise NotImplementedError
