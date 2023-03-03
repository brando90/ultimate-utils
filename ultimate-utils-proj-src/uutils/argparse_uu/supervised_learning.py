import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import *

# later, doesn't let me do

from uutils import load_cluster_jobids_to, merge_args, expanduser

from pdb import set_trace as st

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
    parser.add_argument('--training_mode', type=str, default='iterations',
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
    parser.add_argument('--data_path', type=str, default=Path('~/data/mds/records/').expanduser(),
                        # Path('~/data/mnist/').expanduser()
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

    # - stat analysis
    parser.add_argument('--stats_analysis_option', type=str,
                        default='stats_analysis_with_emphasis_on_effect_size_and_and_full_performance_comp')

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

    # ========MDS args 1/25=========#
    parser.add_argument('--image_size', type=int, default=84,
                        help='Images will be resized to this value')
    # mscoco and traffic sign are val only
    parser.add_argument('--sources', nargs="+",
                        default=['ilsvrc_2012', 'aircraft', 'cu_birds', 'dtd', 'fungi', 'omniglot',
                                 'quickdraw', 'vgg_flower', 'traffic_sign'],
                        help='List of datasets to use')

    parser.add_argument('--train_transforms', nargs="+", default=['random_resized_crop', 'random_flip'],
                        help='Transforms applied to training data', )

    parser.add_argument('--test_transforms', nargs="+", default=['resize', 'center_crop'],
                        help='Transforms applied to test data', )

    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Whether or not to shuffle data')

    # Episode configuration
    # WARNING! this min_examples_in_class should match args.k_shots + args.k_eval in MAML for a fair comparison
    parser.add_argument('--min_examples_in_class', type=int, default=20)

    # supervised learning doesnt need ways/shots
    parser.add_argument('--num_ways', type=int, default=None)
    parser.add_argument('--num_support', type=int, default=None)
    parser.add_argument('--num_query', type=int, default=None)

    parser.add_argument('--min_ways', type=int, default=2,
                        help='Minimum # of ways per task')  # doesn't matter since we fixed 5-way setting (so trivially >2 ways)

    parser.add_argument('--max_ways_upper_bound', type=int, default=1000000000000,
                        help='Maximum # of ways per task')

    parser.add_argument('--max_num_query', type=int, default=1000000000000,
                        help='Maximum # of query samples')

    parser.add_argument('--max_support_set_size', type=int, default=1000000000000,
                        help='Maximum # of support samples')

    parser.add_argument('--max_support_size_contrib_per_class', type=int, default=1000000000000,
                        help='Maximum # of support samples per class')

    parser.add_argument('--min_log_weight', type=float, default=-0.69314718055994529,
                        help='Do not touch, used to randomly sample support set')

    parser.add_argument('--max_log_weight', type=float, default=0.69314718055994529,
                        help='Do not touch, used to randomly sample support set')

    # Hierarchy options
    parser.add_argument('--ignore_bilevel_ontology', type=bool, default=False,
                        help='Whether or not to use superclass for BiLevel datasets (e.g Omniglot)')

    parser.add_argument('--ignore_dag_ontology', type=bool, default=False,
                        help='Whether to ignore ImageNet DAG ontology when sampling \
                                          classes from it. This has no effect if ImageNet is not  \
                                          part of the benchmark.')

    parser.add_argument('--ignore_hierarchy_probability', type=float, default=0.,
                        help='if using a hierarchy, this flag makes the sampler \
                                          ignore the hierarchy for this proportion of episodes \
                                          and instead sample categories uniformly.')
    # ======end MDS args 1/25=======#

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
    print('----> Recovering args & model from checkpoint')
    logging.info('-----> Recovering args & model from checkpoint')
    # -
    args.path_to_checkpoint: Path = expanduser(args.path_to_checkpoint)
    print(f'{args.path_to_checkpoint=}')
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
    # from torch.nn import CrossEntropyLoss
    # later, doesn't let me do import all with * locally :(
    args.loss = eval(args.loss)

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

# - some default args

def get_args_mi_usl_default(args: Optional[Namespace] = None, log_to_wandb: Optional[bool] = None):
    import os
    # -
    args: Namespace = parse_args_standard_sl() if args is None else args
    # - model
    args.n_cls = 64
    # args.n_cls = 64 + 1100  # mio
    # args.n_cls = 1262  # micod
    # args.n_cls = 5  # 5-way
    args.n_classes = args.n_cls
    args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.filter_size = 4
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    # args.data_option = 'hdb4_micod'
    # args.n_classes = args.n_cls
    # args.data_augmentation = 'hdb4_micod'
    args.data_path = Path('~/data/miniImageNet_rfs/miniImageNet').expanduser()
    args.data_option = str(args.data_path)


    # - training mode
    args.training_mode = 'iterations'
    # args.num_its = 100_000  # mds 50_000: https://gitA,aahub.com/google-research/meta-dataset/blob/d6574b42c0f501225f682d651c631aef24ad0916/meta_dataset/learn/gin/best/pretrain_imagenet_resnet.gin#L20
    # args.num_its = 2_000_000  # hdb4 resnetrfs about 2.5 more than above
    args.num_its = 6

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    # args.opt_option = 'AdafactorDefaultFair'
    args.opt_option = 'Adam_rfs_cifarfs'
    args.batch_size = 256
    args.lr = 1e-3
    args.opt_hps: dict = dict(lr=args.lr)

    # - scheduler
    # args.scheduler_option = 'AdafactorSchedule'
    # args.scheduler_option = 'None'
    args.scheduler_option = 'Adam_cosine_scheduler_rfs_cifarfs'
    args.log_scheduler_freq = 1
    args.T_max = args.num_its // args.log_scheduler_freq  # intended 800K/2k
    args.eta_min = 1e-5  # match MAML++
    args.scheduler_hps: dict = dict(T_max=args.T_max, eta_min=args.eta_min)
    print(f'{args.T_max=}')

    # - logging params
    args.log_freq = args.num_its // 4  # logs 4 times
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                                metric_to_use='train_acc', threshold=0.9, log_speed_up=10)

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    # args.log_to_wandb = True  # set false for the this default dummy args
    # args.log_to_wandb = False  # set false for the this default dummy args
    args.log_to_wandb = False if log_to_wandb is None else log_to_wandb

    # -- Setup up remaining stuff for experiment
    from uutils.argparse_uu.common import setup_args_for_experiment
    args: Namespace = setup_args_for_experiment(args)
    return args