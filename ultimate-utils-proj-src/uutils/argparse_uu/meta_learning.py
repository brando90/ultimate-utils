from argparse import Namespace
from pathlib import Path

from uutils import load_cluster_jobids_to


def parse_basic_meta_learning_args_from_terminal() -> Namespace:
    """
    Parse the arguments from the terminal so that the experiment runs as the user specified there.

    Example/Recommended pattern to use:
        See setup_args_for_experiment(...) to avoid copy pasting example/only mantain it in one place.

    Note:
        - Strongly recommended to see setup_args_for_experiment(...)
    """
    import argparse
    # import torch.nn as nn

    parser = argparse.ArgumentParser()

    # experimental setup
    parser.add_argument('--debug', action='store_true', help='if debug')
    parser.add_argument('--force_log', action='store_true', help='to force logging')
    parser.add_argument('--serial', action='store_true', help='if running serially')
    parser.add_argument('--args_hardcoded_in_script', action='store_true',
                        help='set to true if the args will be set from the script manually'
                             'e.g. by hardcoding them in the script.')

    parser.add_argument('--split', type=str, default='train', help=' train, val, test')
    # this is the name used in synth agent, parser.add_argument('--data_set_path', type=str, default='', help='path to data set splits')
    parser.add_argument('--path_to_data_set', type=str, default='VALUE SET IN MAIN Meta-L SCRIPT',
                        help='path to data set splits')

    parser.add_argument('--log_root', type=str, default=Path('/logs/').expanduser())

    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    parser.add_argument('--training_mode', type=str, default='iterations')
    # parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    # parser.add_argument('--embed_dim', type=int, default=256)
    # parser.add_argument('--nhead', type=int, default=8)
    # parser.add_argument('--num_layers', type=int, default=1)
    # parser.add_argument('--criterion', type=str, help='loss criterion', default=nn.CrossEntropyLoss())

    # - optimization
    # parser.add_argument('--optimizer', type=str, default='Adam')
    # parser.add_argument('--learning_rate', type=float, default=1e-5)
    # parser.add_argument('--num_warmup_steps', type=int, default=-1)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--l2', type=float, default=0.0)
    # parser.add_argument('--lr_reduce_steps', default=3, type=int,
    #                     help='the number of steps before reducing the learning rate \
    #                     (only applicable when no_validation == True)')
    # parser.add_argument('--lr_reduce_patience', type=int, default=10)

    # - miscellaneous
    parser.add_argument('--path_to_checkpoint', type=str, default=None,
                        help='the model checkpoint path to resume from.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully determinsitic')
    # for now default is 4 since meta-learning code is not parallizable right now.
    parser.add_argument('--num_workers', type=int, default=4,
                        help='the number of data_lib-loading threads (when running serially')

    # - sims/dists computations
    parser.add_argument('--sim_compute_parallel', action='store_true', help='compute sim or dist in parallel.')
    parser.add_argument('--metrics_as_dist', action='store_true', help='')
    parser.add_argument('--show_layerwise_sims', action='store_true', help='show sim/dist values per layer too')

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases')
    parser.add_argument('--wandb_project', type=str, default='meta-learning-playground')
    parser.add_argument('--wandb_entity', type=str, default='brando')
    parser.add_argument('--wandb_group', type=str, default='experiment_debug', help='helps grouping experiment runs')
    # parser.add_argument('--wandb_log_freq', type=int, default=10)
    # parser.add_argument('--wandb_ckpt_freq', type=int, default=100)
    # parser.add_argument('--wanbd_mdl_watch_log_freq', type=int, default=-1)

    # - parse arguments
    args = parser.parse_args()
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)
    return args
