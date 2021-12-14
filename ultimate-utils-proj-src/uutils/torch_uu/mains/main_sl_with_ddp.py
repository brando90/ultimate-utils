"""
Main script to set up meta-learning experiments
"""

import torch
import torch.nn as nn
# import torch.optim as optim

from argparse import Namespace

import pathlib
from pathlib import Path

from argparse_uu.supervised_learning import parse_args_standard_sl
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
        # args: Namespace = uutils.make_args_from_supervised_learning_checkpoint(args=args,
        #                                                                 path2args=args.path_to_checkpoint,
        #                                                                 filename='args.json',
        #                                                                 precedence_to_args_checkpoint=True)
        # # - args to overwrite or newly insert after checkpoint args has been loaded
        # args.num_its = 600_000
        # # args.k_shots = 5
        # # args.k_eval = 15
        # args.meta_batch_size_train = 4
        # args.meta_batch_size_eval = 2
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
    if args.base_model_mode == 'cnn':
        args.bn_momentum = 0.95
        args.bn_eps = 1e-3
        args.grad_clip_mode = 'clip_all_together'
        args.image_size = 84
        args.act_type = 'sigmoid'
        args.base_model = Kcnn(args.image_size, args.bn_eps, args.bn_momentum, args.n_classes,
                               filter_size=args.n_classes,
                               nb_feature_layers=6,
                               act_type=args.act_type)
    elif args.base_model_mode == 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper':
        args.k_eval = 150
        args.bn_momentum = 0.95
        args.bn_eps = 1e-3
        args.grad_clip_mode = 'clip_all_together'
        args.image_size = 84
        args.base_model = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum,
                                  n_classes=args.n_classes).to(args.device)
    elif args.base_model_mode == 'resnet12_rfs':
        args.k_eval = 30
        args.base_model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_classes).to(
            args.device)
    elif args.base_model_mode == 'resnet18_rfs':
        args.k_eval = 30
        args.base_model = resnet18(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_classes).to(
            args.device)
    elif args.base_model_mode == 'resnet18':
        args.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        _replace_bn(args.base_model, 'model')
        args.base_model.fc = torch.nn.Linear(in_features=512, out_features=args.n_classes, bias=True)
    elif args.base_model_mode == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        _replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'resnet101':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
        _replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
        _replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'rand_init_true_arch':
        db = torch.load(str(args.data_path / args.split / 'f_avg.pt'))
        args.base_model = db['f'].to(args.device)
        # re-initialize model: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
        [layer.reset_parameters() for layer in args.base_model.children() if hasattr(layer, 'reset_parameters')]
    # GPU safety check
    args.base_model.to(args.device)  # make sure it is on GPU
    if torch.cuda.is_available():
        args.base_model.cuda()
    print(f'{args.base_model=}')

    # Set up Meta-Learner
    args.scheduler = None
    if args.meta_learner_name == 'maml_fixed_inner_lr':
        args.grad_clip_rate = None
        args.meta_learner = MAMLMetaLearner(args, args.base_model, fo=args.fo, lr_inner=args.inner_lr)
        # args.outer_opt = optim.Adam(args.meta_learner.parameters(), args.outer_lr)
        # args.outer_opt = Adafactor(args.meta_learner.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        # args.scheduler = AdafactorSchedule(args.outer_opt)
    elif args.meta_learner_name == "FitFinalLayer":
        args.meta_learner = FitFinalLayer(args, args.base_model)
        args.inner_opt_name = 'PFF'
        args.outer_opt = 'None'
    else:
        raise ValueError(f"Invalid trainable opt: {args.meta_learner_name}")
    print(f'{args.outer_opt=}')

    # Get Meta-Sets for few shot learning
    if 'miniimagenet' in str(args.data_path):
        args.meta_learner.classification()
        args.training_mode = 'iterations'
        args.dataloaders = get_miniimagenet_dataloaders_torchmeta(args)
    elif 'sinusoid' in str(args.data_path):
        args.training_mode = 'iterations'
        args.criterion = nn.MSELoss()
        args.meta_learner.regression()
        args.dataloaders = get_torchmeta_sinusoid_dataloaders(args)
    elif 'fully_connected' in str(args.data_path.name):
        args.training_mode = 'iterations'
        args.criterion = nn.MSELoss()
        args.meta_learner.regression()
        args.dataloaders = get_torchmeta_rand_fnn_dataloaders(args)
    else:
        raise ValueError(f'Not such task: {args.data_path}')
    assert isinstance(args.dataloaders, dict)

    # -- load layers to do sim analysis
    args.include_final_layer_in_lst = True
    args.layer_names = get_layer_names_to_do_sim_analysis_fc(args,
                                                             include_final_layer_in_lst=args.include_final_layer_in_lst)
    # args.layer_names = get_layer_names_to_do_sim_analysis_bn(args, include_final_layer_in_lst=args.include_final_layer_in_lst)

    # -- run training
    run_training(args)


# -- Resume from checkpoint experiment

def main_resume_from_checkpoint(args: Namespace):
    print('------- Main Resume from Checkpoint  --------')

    # - load checkpoint
    mdl_ckpt, outer_opt, scheduler, meta_learner = get_model_opt_meta_learner_to_resume_checkpoint_resnets_rfs(
        args,
        path2ckpt=args.path_to_checkpoint,
        filename='ckpt_file.pt',
        device=args.device
    )
    args.base_model = mdl_ckpt
    args.meta_learner = meta_learner
    args.outer_opt = outer_opt
    args.scheduler = scheduler
    print(f'{args.base_model=}')
    print(f'{args.meta_learner=}')
    print(f'{args.outer_opt=}')
    print(f'{args.scheduler=}')

    # - data loaders
    if 'miniimagenet' in str(args.data_path):
        args.meta_learner.classification()
        args.dataloaders = get_miniimagenet_dataloaders_torchmeta(args)
    else:
        raise ValueError(f'Not such benchmark: {args.data_path}')
    assert isinstance(args.dataloaders, dict)

    # - run training
    run_training(args)


# -- Common code

def run_training(args: Namespace):
    # -- Choose experiment split
    if args.split == 'train':
        print('--------------------- META-TRAIN ------------------------')
        # if not args.trainin_with_epochs:
        meta_train_fixed_iterations(args)
        # else:
        #     meta_train_epochs(args, meta_learner, args.outer_opt, meta_train_dataloader, meta_val_dataloader)
    elif args.split == 'val':
        print('--------------------- META-Eval ------------------------')
        args.track_higher_grads = False  # so to not track intermeddiate tensors that for back-ward pass when backward pass won't be done
        acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, split=args.split)
        args.logger.loginfo(f"val loss: {loss_mean} +- {loss_std}, val acc: {acc_mean} +- {acc_std}")
    else:
        raise ValueError(f'Value error: args.split = {args.split}, is not a valid split.')
    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        print('---> about to call wandb.finish()')
        wandb.finish()
        print('---> done calling wandb.finish()')


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
