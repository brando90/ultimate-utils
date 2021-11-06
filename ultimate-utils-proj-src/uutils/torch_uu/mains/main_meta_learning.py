#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import Adafactor
# from transformers.optimization import AdafactorSchedule

import uutils

from uutils.torch_uu import get_layer_names_to_do_sim_analysis_fc

from meta_learning.training.meta_training import meta_eval, meta_train_fixed_iterations_full_epoch_possible
from meta_learning.meta_learners.maml_meta_learner import MAMLMetaLearner
from meta_learning.meta_learners.pretrain_convergence import FitFinalLayer

from meta_learning.base_models.resnet_rfs import resnet12, resnet18
from meta_learning.base_models.learner_from_opt_as_few_shot_paper import Learner
from meta_learning.base_models.kcnn import Kcnn

from meta_learning.datasets.rand_fc_nn_vec_mu_ls_gen import get_backbone

import pathlib
from pathlib import Path

from uutils.torch_uu.dataloaders import get_torchmeta_sinusoid_dataloaders, get_torchmeta_rand_fnn_dataloaders, \
    get_miniimagenet_dataloaders_torchmeta
from uutils.torch_uu.distributed import is_lead_worker


def manual_args_load() -> Namespace:
    """
    Manually load args.

    Divided into three parts (due to legacy code)
    1. parse args from terminal
    2. manually load args in this script
    3. add remaining common setup args to experiment

    :param args:
    :return:
    """
    # -- parse args from terminal
    args: Namespace = uutils.parse_basic_meta_learning_args()

    # -- manual args load
    # Config for few-shot learning
    args.k_shots = 5
    # args.k_eval = 15
    args.k_eval = 100
    args.n_classes = 5

    # - training its/epochs
    # args.num_its = 30
    # args.num_its = 4
    # args.meta_batch_size_train = 8
    args.meta_batch_size_train = 32
    args.log_train_freq = 100 if not args.debug else 1

    args.eval_iters = 1
    # args.meta_batch_size_eval = 8
    args.meta_batch_size_eval = 32
    args.log_val_freq = 100 if not args.debug else 1  # for hyperparam tuning. note: lower the quicker the code.

    # - maml
    args.meta_learner_name = 'maml_fixed_inner_lr'
    args.inner_lr = 1e-1
    args.nb_inner_train_steps = 5
    args.track_higher_grads = True  # set to false only during meta-testing, but code sets it automatically only for meta-test
    args.copy_initial_weights = False  # DONT PUT TRUE. details: set to True only if you do NOT want to train base model's initialization https://stackoverflow.com/questions/60311183/what-does-the-copy-initial-weights-documentation-mean-in-the-higher-library-for
    args.fo = True  # True, dissallows flow of higher order grad while still letting params track gradients.
    # args.fo = True
    # - outer trainer params
    args.outer_lr = 1e-5
    # args.grad_clip_rate = None  # does no gradient clipping if None
    # args.grad_clip_mode = None  # more specific setting of the crad clipping split
    args.grad_clip_rate = 0.25  # does no gradient clipping if None, meta-lstm used 0.25
    args.grad_clip_mode = 'clip_all_together'  # clip all params together/the same way
    # - pff
    # args.meta_learner_name = 'FitFinalLayer'
    # -- Data-set options
    args.split = "train"
    # args.split = 'val'
    # args.split = "test"
    # - with BN really small to really large --
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1e-16_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1e-08_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.0001_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.01_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.1_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.25_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_0.5_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_1.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_2.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_4.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_8.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_16.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_with_BN_std1_32.0_std2_1.0_noise_std0.1nb_h_layes3_out1_H15/').expanduser()
    # -- NO BN --
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_std1_0.0001_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_std1_0.1_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_std1_4_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # args.data_path = Path('~/data/dataset_LS_fully_connected_NN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15/meta_set_fully_connected_NN_std1_16_std2_1.0_noise_std0.1nb_h_layes3_out1_H15').expanduser()
    # mini-imagenet
    # args.data_path = 'torchmeta_mini_imagenet'
    # args.data_path = 'sinusoid'
    # Data loader options
    # Base model
    # args.base_model_mode = 'cnn'
    # args.base_model_mode = 'child_mdl_from_opt_as_a_mdl_for_few_shot_learning_paper'  # & MAML
    # args.base_model_mode = 'resnet12_rfs'
    # args.base_model_mode = 'resnet18_rfs'
    # args.base_model_mode = 'resnet18'
    # args.base_model_mode = 'resnet50'
    # args.base_model_mode = 'resnet101'
    # args.base_model_mode = 'resnet152'
    # args.base_model_mode = 'rand_init_true_arch'
    # args.base_model_mode = 'f_avg'
    # args.base_model_mode = 'f_avg_add_noise'
    # args.base_model_mode = 'custom_synthetic_backbone_NO_BN'
    # args.base_model_mode = 'custom_synthetic_backbone_YES_BN'
    args.base_model_mode = 'custom_synthetic_backbone_YES_BN' if '_BN' in str(args.data_path) else 'custom_synthetic_backbone_NO_BN'
    # args.base_model_mode = 'cbfinn_sinusoid'
    # args.base_model_mode = Path('~/data/logs/logs_Sep29_13-05-52_jobid_383794.iam-pbs/ckpt_file.pt').expanduser()
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov06_16-45-35_jobid_669/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov11_13-32-07_jobid_866/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov05_15-44-03_jobid_668/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov11_13-03-40_jobid_858/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov12_09-33-21_jobid_934/ckpt_file.pt'
    # args.base_model_mode = '/home/miranda9/data/logs/logs_Nov11_15-10-28_jobid_851/ckpt_file.pt'
    # args.base_model_mode = Path(args.base_model_mode).expanduser()

    # -- Setup up remaining stuff for experiment
    args: Namespace = uutils.setup_args_for_experiment(args)
    args.num_workers = 4
    args.pin_memory = False  # it is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True), which enables fast data transfer to CUDA-enabled GPUs. https://pytorch.org/docs/stable/data.html
    # load_cluster_jobids_to(args)
    return args

def main(args):
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
        args.base_model = Learner(image_size=args.image_size, bn_eps=args.bn_eps, bn_momentum=args.bn_momentum, n_classes=args.n_classes).to(args.device)
    elif args.base_model_mode == 'resnet12_rfs':
        args.k_eval = 30
        args.base_model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_classes).to(args.device)
    elif args.base_model_mode == 'resnet18_rfs':
        args.k_eval = 30
        args.base_model = resnet18(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=args.n_classes).to(
            args.device)
    elif args.base_model_mode == 'resnet18':
        args.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        # replace_bn(args.base_model, 'model')
        args.base_model.fc = torch.nn.Linear(in_features=512, out_features=args.n_classes, bias=True)
    elif args.base_model_mode == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        # replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'resnet101':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
        # replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
        # replace_bn(model, 'model')
        model.fc = torch.nn.Linear(in_features=2048, out_features=args.n_classes, bias=True)
        args.base_model = model
    elif args.base_model_mode == 'rand_init_true_arch':
        db = torch.load(str(args.data_path / args.split / 'f_avg.pt'))
        args.base_model = db['f'].to(args.device)
        # re-initialize model: https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034
        [layer.reset_parameters() for layer in args.base_model.children() if hasattr(layer, 'reset_parameters')]
    elif args.base_model_mode == 'f_avg':
        db = torch.load(str(args.data_path / args.split / 'f_avg.pt'))
        args.base_model = db['f'].to(args.device)
    elif args.base_model_mode == 'f_avg_add_noise':
        db = torch.load(str(args.data_path / args.split / 'f_avg.pt'))
        args.base_model = db['f'].to(args.device)
        # add small noise to initial weight to break symmetry
        print()
        with torch.no_grad():
            for i, w in enumerate(args.base_model.parameters()):
                mu = torch.zeros(w.size())
                std = w * 1.25e-2  # two decimal places and a little more
                noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
                w += noise
        print('>>> f_avg_add_noise')
    elif 'custom_synthetic_backbone' in args.base_model_mode:
        # - hps for backbone
        Din, Dout = 1, 1
        # H = 15*20  # 15 is the number of features of the target function
        H = 15*4
        # 10 layers, 9 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 9 layers, 8 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 8 layers, 7 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 7 layers, 6 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 6 layers, 5 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 5 layers, 4 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, Dout)]
        # 4 layers, 3 hidden layers
        hidden_dim = [(Din, H), (H, H), (H, H), (H, Dout)]
        # 3 layers, 2 hidden layers
        # hidden_dim = [(Din, H), (H, H), (H, Dout)]
        print(f'# of hidden layers = {len(hidden_dim) - 1}')
        print(f'total layers = {len(hidden_dim)}')
        section_label = [1] * (len(hidden_dim) - 1) + [2]
        # - hps for model
        target_f_name = 'fully_connected_NN_with_BN' if 'YES_BN' in args.base_model_mode else 'fully_connected_NN'
        task_gen_params = {
            'metaset_path': None,
            'target_f_name': target_f_name,
            'hidden_dim': hidden_dim,
            'section_label': section_label,
            'Din': Din, 'Dout': Dout, 'H': H
        }
        # - CUSTOM
        args.base_model = get_backbone(task_gen_params)
        # args.base_model = get_backbone(task_gen_params, act='sigmoid')
        # - save params for generating bb
        args.task_gen_params = task_gen_params
    elif args.base_model_mode == 'cbfinn_sinusoid':
        target_f_name = 'fully_connected_NN'
        # params for backbone
        Din, Dout = 1, 1
        H = 40  # original cbfinn
        # 3 layers, 2 hidden layers (origal cbfinn)
        hidden_dim = [(Din, H), (H, H), (H, Dout)]
        print(f'# of hidden layers = {len(hidden_dim) - 1}')
        print(f'total layers = {len(hidden_dim)}')
        section_label = [1] * (len(hidden_dim) - 1) + [2]
        task_gen_params = {
            'metaset_path': None,
            'target_f_name': target_f_name,
            'hidden_dim': hidden_dim,
            'section_label': section_label,
            'Din': Din, 'Dout': Dout, 'H': H
        }
        # CBFINN SINUSOID
        args.base_model = get_backbone(task_gen_params)
        # args.base_model = get_backbone(task_gen_params, act='sigmoid')
        # save params for generating bb
        args.task_gen_params = task_gen_params
    elif type(args.base_model_mode) is pathlib.PosixPath:
        # db = torch_uu.load(str(args.resume_ckpt_path))
        db = torch.load(str(args.base_model_mode))
        # meta_learner = db['meta_learner']
        args.base_model = db['f']
        # in case loading directly doesn't work
        # modules = eval(db['f_modules_str'])
        # args.base_model = torch_uu.nn.Sequential(modules)
        # f_state_dict = db['f_state_dict']
        # args.base_model.load_state_dict(f_state_dict)
        print('RUNNING FROM CHECKPOINT')
        args.logger.loginfo('RUNNING FROM CHECKPOINT')
    else:
        raise ValueError(f'Not Implemented: args.base_model_mode = {args.base_model_mode}')
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
        args.outer_opt = optim.Adam(args.meta_learner.parameters(), args.outer_lr)
        # args.outer_opt = Adafactor(args.meta_learner.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        # args.scheduler = AdafactorSchedule(args.outer_opt)
    elif args.meta_learner_name == "FitFinalLayer":
        args.meta_learner = FitFinalLayer(args, args.base_model)
        args.inner_opt_name = 'PFF'
        args.outer_opt = 'None'
    else:
        raise ValueError(f"Invalid trainable opt: {args.meta_learner_name}")

    # Get Meta-Sets for few shot learning
    if 'torchmeta_mini_imagenet' in str(args.data_path):
        args.meta_learner.classification()
        args.training_mode = 'iterations'
        meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_miniimagenet_dataloaders_torchmeta(args)
    elif 'sinusoid' in str(args.data_path):
        args.training_mode = 'iterations'
        args.criterion = nn.MSELoss()
        args.meta_learner.regression()
        meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_torchmeta_sinusoid_dataloaders(args)
    elif 'fully_connected' in str(args.data_path.name):
        args.training_mode = 'iterations'
        args.criterion = nn.MSELoss()
        args.meta_learner.regression()
        meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = get_torchmeta_rand_fnn_dataloaders(args)
    else:
        raise ValueError(f'Not such task: {args.data_path}')
    args.dataloaders = {'train': meta_train_dataloader, 'val': meta_val_dataloader, 'test': meta_test_dataloader}

    # -- load layers to do sim analysis
    args.include_final_layer_in_lst = True
    args.layer_names = get_layer_names_to_do_sim_analysis_fc(args, include_final_layer_in_lst=args.include_final_layer_in_lst)
    # args.layer_names = get_layer_names_to_do_sim_analysis_bn(args, include_final_layer_in_lst=args.include_final_layer_in_lst)

    # -- Choose experiment split
    assert 'meta' not in args.split
    if args.split == 'train':
        print('--------------------- META-TRAIN ------------------------')
        # if not args.trainin_with_epochs:
        meta_train_fixed_iterations_full_epoch_possible(args)
        # else:
        #     meta_train_epochs(args, meta_learner, args.outer_opt, meta_train_dataloader, meta_val_dataloader)
    elif args.split == 'val':
        print('--------------------- META-Eval Val ------------------------')
        # args.track_higher_grads = False  # so to not track intermeddiate tensors that for back-ward pass when backward pass won't be done
        acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, meta_test_dataloader)
        args.logger.loginfo(f"val loss: {loss_mean} +- {loss_std}, val acc: {acc_mean} +- {acc_std}")
    elif args.split == 'test':
        print('--------------------- META-Eval Test ------------------------')
        # args.track_higher_grads = False  # so to not track intermeddiate tensors that for back-ward pass when backward pass won't be done
        acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, meta_test_dataloader)
        args.logger.loginfo(f"val loss: {loss_mean} +- {loss_std}, val acc: {acc_mean} +- {acc_std}")
    else:
        raise ValueError(f'Value error: args.split = {args.split}, is not a valid split.')
    # - wandb
    if is_lead_worker(args.rank) and args.log_to_wandb:
        import wandb
        print('---> about to call wandb.finish()')
        wandb.finish()
        print('---> done calling wandb.finish()')

if __name__ == "__main__":
    import time
    start = time.time()
    # - run experiment
    args = manual_args_load()
    main(args)
    # - print success
    duration_secs = time.time() - start
    print(f"\nSuccess, time passed: hours:{duration_secs / (60 ** 2)}, minutes={duration_secs / 60}, seconds={duration_secs}")
    print('--> Success Done! (python print) \a')

