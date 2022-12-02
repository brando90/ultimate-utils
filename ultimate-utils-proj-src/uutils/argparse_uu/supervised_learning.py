from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from uutils import load_cluster_jobids_to, merge_args


def parse_args_standard_sl() -> Namespace:
    import argparse

    # - great terminal argument parser
    parser = argparse.ArgumentParser()

    # -- create argument options
    parser.add_argument('--debug', action='store_true', help='if debug', default=False)
    # parser.add_argument('--serial', action='store_true', help='if running serially', default=False)
    parser.add_argument('--parallel', action='store_true', help='if running in parallel', default=False)

    # - path to log_root
    parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())

    # - training options
    parser.add_argument('--training_mode', type=str, default='epochs_train_convergence',
                        help='valid/possible values: '
                             'fit_single_batch'
                             'iterations'
                             'epochs'
                             'iterations_train_convergence'
                             'epochs_train_convergence'
                             '- Note: since the code checkpoints the best validation model anyway, it is already doing'
                             'early stopping, so early stopping criterion is not implemented. You can kill the job'
                             'if you see from the logs in wanbd that you are done.'
                        )
    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    # parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    parser.add_argument('--train_convergence_patience', type=int, default=10, help='How long to wait for converge of'
                                                                                  'training. Note this code should'
                                                                                  'be saving the validation ckpt'
                                                                                  'so you are automatically doing '
                                                                                  'early stopping already.')

    # model & loss function options
    parser.add_argument('--model_option',
                        type=str,
                        default="5CNN_opt_as_model_for_few_shot_sl",
                        help="Options: e.g."
                             "5CNN_opt_as_model_for_few_shot_sl"
                             "resnet12_rfs"
                        )
    parser.add_argument('--loss', type=str, help='loss/criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--opt_option', type=str, default='AdafactorDefaultFair')
    parser.add_argument('--lr', type=float, default=None, help='Warning: use a learning rate according to'
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
    parser.add_argument('--grad_clip_mode', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    parser.add_argument('--scheduler_option', type=str, default='AdafactorSchedule', help='Its recommended')
    parser.add_argument('--log_scheduler_freq', type=int, default=-1, help='default is to put the epochs or iterations '
                                                                           'default either log every epoch or log ever '
                                                                           '~100 iterations.')

    # - data set args
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_eval', type=int, default=128)
    parser.add_argument('--split', type=str, default='train', help="possible values: "
                                                                   "'train', val', test'")
    parser.add_argument('--data_path', type=str, default=Path('~/data/mnist/').expanduser(),
                        help='path to data set splits. The code will assume everything is saved in'
                             'the uutils standard place in ~/data/, ~/data/logs, etc. see the setup args'
                             'setup method and log_root.')
    parser.add_argument('--data_augmentation', type=str, default=None)
    parser.add_argument('--not_augment_train', action='store_false', default=True)
    parser.add_argument('--augment_val', action='store_true', default=False)
    parser.add_argument('--augment_test', action='store_true', default=False)
    # parser.add_argument('--l2', type=float, default=0.0)

    # - checkpoint options
    parser.add_argument('--path_to_checkpoint', type=str, default=None, help='the path to the model checkpoint to '
                                                                             'resume training.'
                                                                             'e.g. path: '
                                                                             '~/data_folder_fall2020_spring2021/logs/nov_all_mini_imagenet_expts/logs_Nov05_15-44-03_jobid_668/ckpt.pt')
    parser.add_argument('--ckpt_freq', type=int, default=-1)

    # - dist/distributed options
    parser.add_argument('--init_method', type=str, default=None)

    # - miscellaneous arguments
    parser.add_argument('--log_freq', type=int, default=-1, help='default is to put the epochs or iterations default'
                                                                 'either log every epoch or log ever ~100 iterations')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully deterministic', default=False)
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='the number of data_lib-loading threads (when running serially')
    parser.add_argument('--pin_memory', action='store_true', default=False, help="Using pinning is an"
                                                                                 "advanced tip according to"
                                                                                 "pytorch docs, so will "
                                                                                 "leave it False as default"
                                                                                 "use it at your own risk"
                                                                                 "of further debugging and"
                                                                                 "spending time on none"
                                                                                 "essential, likely over"
                                                                                 "optimizing. See:"
                                                                                 "https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning")
    parser.add_argument('--log_to_tb', action='store_true', help='store to weights and biases', default=False)

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases', default=False)
    parser.add_argument('--wandb_project', type=str, default='meta-learning-playground')
    parser.add_argument('--wandb_entity', type=str, default='brando')
    # parser.add_argument('--wandb_project', type=str, default='test-project')
    # parser.add_argument('--wandb_entity', type=str, default='brando-uiuc')
    parser.add_argument('--wandb_group', type=str, default='experiment_debug', help='helps grouping experiment runs')
    # parser.add_argument('--wandb_log_freq', type=int, default=10)
    # parser.add_argument('--wandb_ckpt_freq', type=int, default=100)
    # parser.add_argument('--wanbd_mdl_watch_log_freq', type=int, default=-1)

    # - manual loads
    parser.add_argument('--manual_loads_name', type=str, default='None')

    # - 5CNN args
    parser.add_argument('--filter_size', type=int, default=-1, help="Filter size for 5CNN.")

    # - parse arguments
    args: Namespace = parser.parse_args()
    args.criterion = args.loss
    assert args.criterion is args.loss
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
    from uutils.argparse_uu.common import create_default_log_root
    create_default_log_root(args)  # creates a new log root with the current job number etc
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
