from argparse import Namespace
from pathlib import Path

from torch import nn

from uutils import load_cluster_jobids_to

def fix_for_backwards_compatibility(args: Namespace) -> Namespace:
    args.meta_batch_size_train = args.batch_size
    args.meta_batch_size_eval = args.batch_size

    # args.log_train_freq = 100 if not args.debug else 1
    args.log_train_freq = args.log_freq
    args.log_val_freq = 100 if not args.debug else 1  # for hyperparam tuning. note: lower the quicker the code.

    args.grad_clip_rate = None  # does no gradient clipping if None
    # args.grad_clip_mode = None  # more specific setting of the crad clipping split
    # args.grad_clip_rate = 0.25  # does no gradient clipping if None, meta-lstm used 0.25
    # args.grad_clip_mode = 'clip_all_together'  # clip all params together/the same way

    # - unsure if to remove
    args.eval_iters = 1


def parse_args_meta_learning() -> Namespace:
    """
    WARNING: SOME NAMES MIGHT HAVE COPIES TO MAKE THINGS BACKWARD COMPATIBLE.

    Parse the arguments from the terminal so that the experiment runs as the user specified there.

    Example/Recommended pattern to use:
        See setup_args_for_experiment(...) to avoid copy pasting example/only mantain it in one place.

    Note:
        - Strongly recommended to see setup_args_for_experiment(...)
    """
    import argparse
    # import torch.nn as nn

    parser = argparse.ArgumentParser()

    # -- create argument options
    parser.add_argument('--debug', action='store_true', help='if debug')
    # parser.add_argument('--serial', action='store_true', help='if running serially')
    parser.add_argument('--parallel', action='store_true', help='if running in parallel')

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
    parser.add_argument('--train_convergence_patience', type=int, default=5, help='How long to wait for converge of'
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
    parser.add_argument('--scheduler_option', type=str, default='AdafactorSchedule', help='Its strongly recommended')
    parser.add_argument('--log_scheduler_freq', type=int, default=-1, help='default is to put the epochs or iterations '
                                                                           'default either log every epoch or log ever '
                                                                           '~100 iterations.')

    # - data set args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=4)
    parser.add_argument('--split', type=str, default='train', help="possible values: "
                                                                   "'train', val', test'")
    # warning: sl name is path_to_data_set here its data_path
    parser.add_argument('--data_option', type=str, default='None')
    parser.add_argument('--data_path', type=str, default=None)
    # parser.add_argument('--data_path', type=str, default='torchmeta_miniimagenet',
    #                     help='path to data set splits. The code will assume everything is saved in'
    #                          'the uutils standard place in ~/data/, ~/data/logs, etc. see the setup args'
    #                          'setup method and log_root.')
    # parser.add_argument('--path_to_data_set', type=str, default='None')
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

    # - miscellaneous arguments
    parser.add_argument('--log_freq', type=int, default=-1, help='default is to put the epochs or iterations default'
                                                                 'either log every epoch or log ever ~100 iterations')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--always_use_deterministic_algorithms', action='store_true',
                        help='tries to make pytorch fully deterministic')
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
    parser.add_argument('--log_to_tb', action='store_true', help='store to weights and biases')

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases')
    parser.add_argument('--wandb_project', type=str, default='meta-learning-playground')
    parser.add_argument('--wandb_entity', type=str, default='brando')
    parser.add_argument('--wandb_group', type=str, default='experiment_debug', help='helps grouping experiment runs')
    # parser.add_argument('--wandb_log_freq', type=int, default=10)
    # parser.add_argument('--wandb_ckpt_freq', type=int, default=100)
    # parser.add_argument('--wanbd_mdl_watch_log_freq', type=int, default=-1)

    # - manual loads
    parser.add_argument('--manual_loads_name', type=str, default='None')

    # - meta-learner specific
    parser.add_argument('--k_shots', type=int, default=5, help="")
    parser.add_argument('--k_eval', type=int, default=100, help="")
    parser.add_argument('--n_classes', type=int, default=5, help="")

    # - parse arguments
    args = parser.parse_args()
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)
    return args
