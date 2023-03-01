from argparse import Namespace
from pathlib import Path
from typing import Optional

from torch import nn

from uutils import load_cluster_jobids_to


def fix_for_backwards_compatibility(args: Namespace, gpu_idx: int = 0) -> Namespace:
    args.meta_batch_size_train = args.batch_size
    args.meta_batch_size_eval = args.batch_size_eval

    args.log_train_freq = args.log_freq
    args.log_val_freq = args.log_freq

    # - unsure if to remove
    args.eval_iters = 1

    args.outer_lr = args.lr
    args.n_classes = args.n_cls

    if not hasattr(args, 'rank'):
        args.rank = -1
    if not hasattr(args, 'device'):
        from uutils.torch_uu import get_device
        args.device = get_device(gpu_idx)
    return args


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
    import os
    # import torch.nn as nn

    parser = argparse.ArgumentParser()

    # -- create argument options
    parser.add_argument('--debug', action='store_true', help='if debug')
    # parser.add_argument('--serial', action='store_true', help='if running serially')
    parser.add_argument('--parallel', action='store_true', help='if running in parallel')

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
    parser.add_argument('--scheduler_option', type=str, default='AdafactorSchedule', help='Its strongly recommended')
    parser.add_argument('--log_scheduler_freq', type=int, default=-1, help='default is to put the epochs or iterations '
                                                                           'default either log every epoch or log ever '
                                                                           '~100 iterations.')

    # - data set args
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batch_size_eval', type=int, default=2)
    parser.add_argument('--split', type=str, default='train', help="possible values: "
                                                                   "'train', val', test'")
    # warning: sl name is path_to_data_set here its data_path
    parser.add_argument('--data_option', type=str, default='None')
    parser.add_argument('--data_path', type=str, default=os.path.expanduser('~/data/mds/records/'))
    # parser.add_argument('--data_path', type=str, default=None)
    # parser.add_argument('--data_path', type=str, default='torchmeta_miniimagenet',
    #                     help='path to data set splits. The code will assume everything is saved in'
    #                          'the uutils standard place in ~/data/, ~/data/logs, etc. see the setup args'
    #                          'setup method and log_root.')
    # parser.add_argument('--path_to_data_set', type=str, default='None')
    parser.add_argument('--data_augmentation', type=str, default=None)
    parser.add_argument('--not_augment_train', action='store_false', default=True)
    parser.add_argument('--augment_val', action='store_true', default=True)
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

    # - stat analysis
    parser.add_argument('--stats_analysis_option', type=str,
                        default='stats_analysis_with_emphasis_on_effect_sizeand_and_full_performance_comp')

    # - wandb
    parser.add_argument('--log_to_wandb', action='store_true', help='store to weights and biases')
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

    # - meta-learner specific
    parser.add_argument('--k_shots', type=int, default=5, help="")
    parser.add_argument('--k_eval', type=int, default=15, help="")
    parser.add_argument('--n_cls', type=int, default=5, help="")  # n_ways
    parser.add_argument('--n_aug_support_samples', type=int, default=1,
                        help="The puzzling rfs increase in support examples")

    # - 5CNN args
    parser.add_argument('--filter_size', type=int, default=-1, help="Filter size for 5CNN.")

    # - task2vec args
    parser.add_argument('--classifier_opts', type=dict, default=None,
                        help="The classifier options to fit the final layer"
                             "when computing the embeding of a task using"
                             "task2vec. Note that the FIM is currently "
                             "being run to completion (so no break points"
                             "to exit sooner out of FIM. This flag allows"
                             "to potentially exit sooner form fitting the"
                             "final layer.")

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
    args = parser.parse_args()

    # ==set number of ways and shots in MDS to ones we specify above===#
    args.num_ways = args.n_cls
    args.num_support = args.k_shots
    args.num_query = args.k_eval
    args.min_examples_in_class = args.k_shots + args.k_eval  # should be 5+15=20 in our setting.
    # ====================#

    args.criterion = args.loss
    assert args.criterion is args.loss
    # - load cluster ids so that wandb can use it later for naming runs, experiments, etc.
    load_cluster_jobids_to(args)
    return args

# -- some default args

def get_args_mi_torchmeta_default(args: Optional[Namespace] = None, log_to_wandb: Optional[bool] = None):
    """
    Some default args to show case how code works + for unit tests too.
    """
    import os
    from pathlib import Path
    import torch
    # - args
    args: Namespace = parse_args_meta_learning() if args is None else args
    # - model
    # # args.model_option = 'resnet18_rfs'  # note this corresponds to block=(1 + 1 + 2 + 2) * 3 + 1 = 18 + 1 layers (sometimes they count the final layer and sometimes they don't)
    # args.n_cls = 5
    # # bellow seems true for all models, they do use avg pool at the global pool/last pooling layer
    # args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5,
    #                       num_classes=args.n_cls)  # dropbock_size=5 is rfs default for MI, 2 for CIFAR, will assume 5 for mds since it works on imagenet
    args.n_cls = 5
    args.model_option = '5CNN_opt_as_model_for_few_shot'
    args.filter_size = 4
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'torchmeta_miniimagenet'
    args.data_path = '~/data/torchmeta_data'

    # - training mode
    args.training_mode = 'iterations'

    # note: 75_000 used by MAML mds https://github.com/google-research/meta-dataset/blob/main/meta_dataset/learn/gin/setups/trainer_config.gin#L1
    # args.num_its = 75_000  # 7_500 in 2 days
    args.num_its = 6

    # - debug flag
    args.debug = False
    # args.debug = True

    # - opt
    # args.opt_option = 'AdafactorDefaultFair'
    # args.opt_hps: dict = dict()
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
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

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.track_higher_grads = True  # I know this is confusing but look at this ref: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch
    args.fo = True  # This is needed.

    # - outer trainer params
    args.batch_size = 3  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2

    # - logging params
    args.log_freq = args.num_its // 4  # logs 4 times
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                                metric_to_use='train_acc', threshold=0.9, log_speed_up=10)

    # -- wandb args
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.data_option} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option}: {args.jobid=} {args.manual_loads_name}'
    # args.log_to_wandb = True  # set false for the this default dummy args
    # args.log_to_wandb = False  # set false for the this default dummy args
    args.log_to_wandb = False if log_to_wandb is None else log_to_wandb

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    from uutils.argparse_uu.common import setup_args_for_experiment
    args: Namespace = setup_args_for_experiment(args)
    return args


def get_args_mi_l2l_default(args: Optional[Namespace] = None, log_to_wandb: Optional[bool] = None) -> Namespace:
    """
    Some default args to show case how code works + for unit tests too.
    """
    import os
    from pathlib import Path
    import torch
    # - args
    args: Namespace = parse_args_meta_learning() if args is None else args
    # - model
    args.n_cls = 5
    args.model_option = '5CNN_opt_as_model_for_few_shot'
    # args.model_hps = dict(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)
    args.filter_size = 4
    args.model_hps = dict(image_size=84, bn_eps=1e-3, bn_momentum=0.95, n_classes=args.n_cls,
                          filter_size=args.filter_size, levels=None, spp=False, in_channels=3)

    # - data
    args.data_option = 'mini-imagenet'
    args.data_path = Path('~/data/l2l_data/').expanduser()
    args.data_augmentation = None  # uses default of l2l, l2l ignores this flag but here for clarity in other datasets

    # - training mode
    args.training_mode = 'iterations'

    # note: 60K iterations for original maml 5CNN with adam
    args.num_its = 6
    # args.num_its = 900_000  # resnet12rfs conv with 300K, lets to 3 times to be safe

    # - debug flag
    # args.debug = True
    args.debug = False

    # - opt
    # args.opt_option = 'AdafactorDefaultFair'
    args.opt_option = 'Adam_rfs_cifarfs'
    args.lr = 1e-3  # match MAML++
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

    # -- Meta-Learner
    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1  # same as fast_lr in l2l
    args.nb_inner_train_steps = 5
    # args.first_order = True  # need to create new args that uses first order maml, leaving as is for reproducibility
    args.first_order = False  # seems I did higher order maml by accident, leaving it to not be confusing

    # - outer trainer params
    args.batch_size = 3  # decreased it to 4 even though it gives more noise but updates quicker + nano gpt seems to do that for speed up https://github.com/karpathy/nanoGPT/issues/58
    args.batch_size_eval = 2

    # - dist args
    from uutils.torch_uu.distributed import get_default_world_size
    args.world_size = get_default_world_size()
    # args.world_size = torch.cuda.device_count()
    # args.world_size = 8
    args.parallel = args.world_size > 1
    # args.seed = 42  # I think this might be important due to how tasksets works.
    args.dist_option = 'l2l_dist'  # avoid moving to ddp when using l2l
    # args.init_method = 'tcp://localhost:10001'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    # args.init_method = f'tcp://127.0.0.1:{find_free_port()}'  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works
    args.init_method = None  # <- this cannot be hardcoded here it HAS to be given as an arg due to how torch.run works

    # - logging params
    args.log_freq = args.num_its // 4  # logs 4 times
    # args.smart_logging_ckpt = dict(smart_logging_type='log_more_often_after_threshold_is_reached',
    #                                metric_to_use='train_acc', threshold=0.9, log_speed_up=10)

    # -- wandb args
    # args.wandb_project = 'playground'  # needed to log to wandb properly
    args.wandb_project = 'entire-diversity-spectrum'
    # - wandb expt args
    args.experiment_name = f'{args.manual_loads_name} {args.model_option} {args.data_option} {args.filter_size} {os.path.basename(__file__)}'
    args.run_name = f'{args.manual_loads_name} {args.model_option} {args.opt_option} {args.lr} {args.scheduler_option} {args.filter_size}: {args.jobid=}'
    # args.log_to_wandb = True  # set false for the this default dummy args
    # args.log_to_wandb = False  # set false for the this default dummy args
    args.log_to_wandb = False if log_to_wandb is None else log_to_wandb

    # - fix for backwards compatibility
    args = fix_for_backwards_compatibility(args)
    from uutils.argparse_uu.common import setup_args_for_experiment
    args: Namespace = setup_args_for_experiment(args)
    return args
