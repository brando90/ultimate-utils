from pathlib import Path


def parse_args_synth_agent():
    import argparse
    import torch.nn as nn

    parser = argparse.ArgumentParser()

    # experimental setup
    parser.add_argument('--reproduce_10K', action='store_true', default=False,
                        help='Unset this if you want to run'
                             'your own data set. This is not really meant'
                             'to be used if you are trying to reproduce '
                             'the simply type lambda cal experiments on the '
                             '10K dataset.')
    parser.add_argument('--debug', action='store_true', help='if debug')
    parser.add_argument('--force_log', action='store_true', help='to force logging')
    parser.add_argument('--serial', action='store_true', help='if running serially')
    vars
    parser.add_argument('--split', type=str, default='train', help=' train, val, test')
    parser.add_argument('--data_set_path', type=str, default='~/data/simply_type_lambda_calc/dataset10000/',
                        help='path to data set splits')

    parser.add_argument('--log_root', type=str, default=Path('/logs/').expanduser())

    parser.add_argument('--num_epochs', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=-1)
    parser.add_argument('--training_mode', type=str, default='iterations')
    parser.add_argument('--no_validation', action='store_true', help='no validation is performed')
    # parser.add_argument('--save_model_epochs', type=int, default=10,
    #                     help='the number of epochs between model savings')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='the number of data_lib-loading threads (when running serially')

    # term encoder
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=1)

    # tactic label classifier
    parser.add_argument('--criterion', type=str, help='loss criterion', default=nn.CrossEntropyLoss())

    # optimization
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    # parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--l2', type=float, default=0.0)
    # parser.add_argument('--lr_reduce_steps', default=3, type=int,
    #                     help='the number of steps before reducing the learning rate \
    #                     (only applicable when no_validation == True)')

    parser.add_argument('--lr_reduce_patience', type=int, default=10)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully determinsitic')

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases')
    parser.add_argument('--wandb_project', type=str, default='playground')
    parser.add_argument('--wandb_entity', type=str, default='brando')
    parser.add_argument('--experiment_name', type=str, default='experiment1', help='helps grouping experiment runs')
    # parser.add_argument('--wandb_log_freq', type=int, default=10)
    # parser.add_argument('--wandb_ckpt_freq', type=int, default=100)
    # parser.add_argument('--wanbd_mdl_watch_log_freq', type=int, default=100)

    # - parse arguments
    args = parser.parse_args()
    return args
