"""
Main script to set up meta-learning experiments
"""

import torch
import torch.nn as nn
# import torch.optim as optim

from argparse import Namespace

from uutils import args_hardcoded_in_script
from uutils.argparse_uu import parse_args_standard_sl
from uutils.torch_uu.checkpointing_uu import resume_from_checkpoint


def manual_load(args) -> Namespace:
    """
    Warning: hardcoding the args can make it harder to reproduce later in a main.sh script with the
    arguments to the experiment.
    """
    raise ValueError(f'Not implemented')

def load_args() -> Namespace:
    """
    1. parse args from user's terminal
    2. optionally set remaining args values (e.g. manually, hardcoded, from ckpt etc.)
    3. setup remaining args small details from previous values (e.g. 1 and 2).
    """
    # -- parse args from terminal
    args: Namespace = parse_args_standard_sl()
    args.wandb_project = 'playground'  # needed to log to wandb properly

    # - debug args
    args.experiment_name = f'debug'
    args.run_name = f'debug (Adafactor) : {args.jobid=}'
    args.force_log = True
    args.log_to_wandb = True

    # - real args
    # args.experiment_name = f'Real experiment name (Real)'
    # args.run_name = f'Real experiment run name: {args.jobid=}'
    # args.force_log = False
    # args.log_to_wandb = True
    # #args.log_to_wandb = False

    # -- set remaining args values (e.g. hardcoded, manually, checkpoint etc)
    if resume_from_checkpoint(args):
        args: Namespace = uutils.make_args_from_supervised_learning_checkpoint(args=args,
                                                                        path2args=args.path_to_checkpoint,
                                                                        filename='args.json',
                                                                        precedence_to_args_checkpoint=True)
    elif args_hardcoded_in_script(args):
        args: Namespace = manual_load(args)
    else:
        # use arguments from terminal
        pass
    # -- Setup up remaining stuff for experiment
    args: Namespace = setup_args_for_experiment(args, num_workers=4)
    return args


def main_manual(args: Namespace):
    print('-------> Inside Main <--------')

    # Set up the learner/base model
    print(f'--> args.base_model_model: {args.base_model_mode}')
    if args.base_model_mode == 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper':

    elif args.base_model_mode == 'resnet12_rfs':
        args.k_eval = 30
        args.base_model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_classes).to(
            args.device)
    elif args.base_model_mode == 'resnet18':
        args.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        _replace_bn(args.base_model, 'model')
        args.base_model.fc = torch.nn.Linear(in_features=512, out_features=args.n_classes, bias=True)
    else:
        raise ValueError(f'Invalid value: {args.model_option=}')

    args.base_model.to(args.device)
    print(f'{args.base_model=}')

    # Get data loaders for data set
    if 'miniimagenet' in str(args.data_path):
        args.meta_learner.classification()
        args.training_mode = 'iterations'
        args.dataloaders = get_miniimagenet_dataloaders_torchmeta(args)
    else:
        raise ValueError(f'Not such task: {args.data_path}')
    assert isinstance(args.dataloaders, dict)

    # -- run training
    run_training(args)

def main_distributed():
    """
    train tree_nn in parallel
    Note: end-to-end ddp example on mnist: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    :return:
    """
    # parse the options
    args = parse_args()
    [print(f'{k,v}') for k, v in vars(args).items()]

    # parallel train
    if torch.cuda.is_available():
        args.world_size = torch.cuda.device_count()
        print(f"{torch.cuda.device_count()=}")
    elif args.serial:
        args.world_size = 1
        print('RUNNING SERIALLY')
    else:
        # args.world_size = mp.cpu_count() - 1  # 1 process is main, the rest are (parallel) trainers
        args.world_size = 4

    # spawn the distributed training code
    print(f'\n{args.world_size=}')

    if args.serial:
        print('RUNNING SERIALLY')
        train(rank=-1, args=args)
    else:
        print('\nABOUT TO SPAWN WORKERS')
        set_sharing_strategy()
        mp.spawn(fn=train, args=(args,), nprocs=args.world_size)

def train(rank, args):
    print_process_info(rank, flush=True)
    args.rank = rank  # have each process save the rank
    set_devices(args)  # basically args.gpu = rank if not debugging/serially
    setup_process(args, rank, master_port=args.master_port, world_size=args.world_size)
    print(f'setup process done for rank={rank}')

    # create the dataloaders, todo
    dataloaders: dict = get_uutils_mnist_dataloaders()

    # create the model
    mdl = get_tree_gen_simple(args, dataloaders)
    mdl = move_to_ddp(rank, args, mdl)

    # start_epoch = load_checkpoint(args, optimizer, tactic_predictor)
    # start_epoch = 0
    # start_it = 0

    # get scheduler the optimizer & scheduler
    # todo
    # optimizer = Adafactor(mdl.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # optimizer = torch_uu.optim.Adam(mdl.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    # optimizer = radam.RAdam(mdl.parameters(), lr=args.learning_rate)
    # get scheduler decay/anneal learning rate wrt epochs
    # todo
    # scheduler = None
    # scheduler = ReduceLROnPlateau(optimizer, patience=args.lr_reduce_patience, verbose=True)  # temporary
    # scheduler = torch_uu.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=False)
    # scheduler = AdafactorSchedule(optimizer)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps)
    # print(f'{scheduler=}')

    # Agent does everything, proving, training, evaluate etc.
    agent = SynthAgent(args, mdl, optimizer, dataloaders, scheduler)

    # - save args
    if agent.is_lead_worker():
        save_args(args)

    # -- Start Training Loop
    agent.log('====> about to start train loop')
    if not args.reproduce_10K:  # real experiment
        agent.log('-- running experiment')
        # agent.main_train_loop_based_on_fixed_number_of_epochs(args, start_epoch)
        # agent.main_train_loop_until_convergence(args, start_it)
        agent.main_train_loop_based_on_fixed_iterations(args, start_it)
        # agent.train_single_batch()

    # -- Clean Up Distributed Processes
    print(f'\n----> about to cleanup worker with rank {rank}')
    cleanup(rank)
    print(f'clean up done successfully! {rank}')


# -- Run experiment

if __name__ == "__main__":
    import time

    start = time.time()

    # - run experiment
    args: Namespace = load_args()
    if resume_from_checkpoint(args):
        main_resume_from_checkpoint(args)
    else:
        main_manual(args)

    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
