from argparse import Namespace
from pathlib import Path

from torch import nn

from uutils import load_cluster_jobids_to


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
    parser.add_argument('--log_root', type=str, default=Path('~/data/logs/').expanduser())

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
    parser.add_argument('--model_type', type=str, help='model type', help='Options:')
    parser.add_argument('--criterion', type=str, help='loss criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--optimizer', type=str, default='Adafactor.')
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
    parser.add_argument('--scheduler', type=str, default='AdafactorSchedule', help='Its strongly recommended')
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

    # - parse arguments
    args: Namespace = parser.parse_args()
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)
    return args
