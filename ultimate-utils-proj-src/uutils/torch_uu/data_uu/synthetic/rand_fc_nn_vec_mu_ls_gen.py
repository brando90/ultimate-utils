#!/home/miranda9/miniconda3/envs/automl-meta-learning/bin/python

"""
Code for generating a benchmark/dataset were each parameter in each layer is sampled as a random vector.
If you use the other code that initializes nets with param values that are the same in all layers
you get unininteresting models (or at the very least the avg model is uninteresting).

"""

import torch
import torch.nn as nn

from collections import OrderedDict

from pathlib import Path

from sklearn.linear_model import LinearRegression

import numpy as np

from matplotlib import pyplot as plt

import json

def my_print(*args, filepath="~/my_stdout.txt"):
    """Modified print statement that prints to terminal/scree AND to a given file (or default).

    Note: import it as follows:

    from utils.utils import my_print as print

    to overwrite builtin print function

    Keyword Arguments:
        filepath {str} -- where to save contents of printing (default: {'~/my_stdout.txt'})
    """
    # https://stackoverflow.com/questions/61084916/how-does-one-make-an-already-opened-file-readable-e-g-sys-stdout
    import sys
    from builtins import print as builtin_print
    filepath = Path(filepath).expanduser()
    # do normal print
    builtin_print(*args, file=sys.__stdout__)  # prints to terminal
    # open my stdout file in update mode
    with open(filepath, "a+") as f:
        # save the content we are trying to print
        builtin_print(*args, file=f)  # saves to file


def norm(f, l=2):
    return sum([w.detach().norm(l) for w in f.parameters()])


def get_backbone(task_gen_params, act=None):
    print(f'act = {act} (if none it probably will use ReLU)')
    backbone_name = task_gen_params['target_f_name']
    if backbone_name == 'debug':
        Din, Dout = task_gen_params['Din'], task_gen_params['Dout']
        f = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(Din, Dout)),
            ('out', nn.SELU())
        ]))
        return f
    elif backbone_name == 'fully_connected_NN':
        hidden_dim = task_gen_params['hidden_dim']
        section_label = task_gen_params['section_label']
        layers = []
        for i, (Din, Dout) in enumerate(hidden_dim):
            section = section_label[i]
            idx = i + 1
            if i != len(hidden_dim) - 1:
                fc = (f'fc{idx}_l{section}', nn.Linear(Din, Dout))
                act = (f'relu{idx}', nn.ReLU())
                layers.extend([fc, act])
            else:
                fc = (f'fc{idx}_final_l{section}', nn.Linear(Din, Dout))
                layers.extend([fc])
        f = nn.Sequential(OrderedDict(layers))
        return f
    # -- with no BN
    elif backbone_name == 'fully_connected_NN' and act == 'sigmoid':
        hidden_dim = task_gen_params['hidden_dim']
        section_label = task_gen_params['section_label']
        layers = []
        for i, (Din, Dout) in enumerate(hidden_dim):
            section = section_label[i]
            idx = i + 1
            if i != len(hidden_dim) - 1:
                fc = (f'fc{idx}_l{section}', nn.Linear(Din, Dout))
                act = (f'sigmoid{idx}', nn.Sigmoid())
                layers.extend([fc, act])
            else:
                fc = (f'fc{idx}_final_l{section}', nn.Linear(Din, Dout))
                layers.extend([fc])
        f = nn.Sequential(OrderedDict(layers))
        return f
    # -- with BN
    elif backbone_name == 'fully_connected_NN_with_BN' and act is None:
        hidden_dim = task_gen_params['hidden_dim']
        section_label = task_gen_params['section_label']
        layers = []
        for i, (Din, Dout) in enumerate(hidden_dim):
            section = section_label[i]
            idx = i + 1
            if i != len(hidden_dim) - 1:
                fc = (f'fc{idx}_l{section}', nn.Linear(Din, Dout))
                bn = (f'bn{idx}_l{section}', nn.BatchNorm1d(num_features=Dout, track_running_stats=False))
                act = (f'relu{idx}', nn.ReLU())
                layers.extend([fc, bn, act])
            else:
                fc = (f'fc{idx}_final_l{section}', nn.Linear(Din, Dout))
                layers.extend([fc])
        f = nn.Sequential(OrderedDict(layers))
        return f
    # -- with BN
    elif backbone_name == 'fully_connected_NN_with_BN' and act == 'sigmoid':
        hidden_dim = task_gen_params['hidden_dim']
        section_label = task_gen_params['section_label']
        layers = []
        for i, (Din, Dout) in enumerate(hidden_dim):
            section = section_label[i]
            idx = i + 1
            if i != len(hidden_dim) - 1:
                fc = (f'fc{idx}_l{section}', nn.Linear(Din, Dout))
                bn = (f'bn{idx}_l{section}', nn.BatchNorm1d(num_features=Dout, track_running_stats=False))
                act = (f'sigmoid{idx}', nn.Sigmoid())
                layers.extend([fc, bn, act])
            else:
                fc = (f'fc{idx}_final_l{section}', nn.Linear(Din, Dout))
                layers.extend([fc])
        f = nn.Sequential(OrderedDict(layers))
        return f
    else:
        raise ValueError(f'Not implemented: backbone_name = {backbone_name}')


def save_average_model(path, f, task_gen_params):
    """
    path - is usually path_2_split
    """
    # initialize average model
    initialize_average_function(f, task_gen_params)
    # torch_uu save it
    path2avg_f = path / 'f_avg.pt'
    torch.save({'f': f,
                'f_state_dict': f.state_dict(),
                'f_str': str(f),
                'f_modules': f._modules,
                'f_modules_str': str(f._modules)
                }, path2avg_f)


def initialize_average_function(f, task_gen_params):
    init_name = task_gen_params['init_name']
    target_f_name = task_gen_params['target_f_name']
    with torch.no_grad():
        if init_name == 'debug' or target_f_name == 'debug':
            mu1 = task_gen_params['mu1']
            [nn.init.constant_(w, mu1) for w in f.parameters()]
        # elif init_name == 'uniform':
        #     mu1, std1, mu2, std2 = task_gen_params['mu1'], task_gen_params['std1'], task_gen_params['mu2'], \
        #                            task_gen_params['std2']
        #     for name, w in f.named_parameters():
        #         w = nn.init.constant_(w, mu1)
        #         if 'l1' in name:
        #             nn.init.constant_(w, mu1)  # BAD
        #         else:
        #             nn.init.constant_(w, mu2)  # BAD
        elif init_name == 'mu_vecs':
            for name, w in f.named_parameters():
                if 'fc' in name:
                    # get the mu vec (center) for the task and set it to the weights of the uninit backbone
                    mu = task_gen_params['mu_vecs'][name]
                    w.data = mu
        else:
            raise ValueError(f'Has not been implemented: init_name={init_name}')


def initialize_target_function(f, task_gen_params):
    init_name = task_gen_params['init_name']
    target_f_name = task_gen_params['target_f_name']
    with torch.no_grad():
        if init_name == 'debug':
            mu1, std1 = task_gen_params['mu1'], task_gen_params['std1']
            lb1, ub1 = mu1 - std1, mu1 + std1
            [nn.init.uniform_(w, a=lb1, b=ub1) for w in f.parameters()]
        # elif init_name == 'uniform':
        #     mu1, std1, mu2, std2 = task_gen_params['mu1'], task_gen_params['std1'], task_gen_params['mu2'], \
        #                            task_gen_params['std2']
        #     lb1, ub1, lb2, ub2 = mu1 - std1, mu1 + std1, mu2 - std2, mu2 + std2
        #     for name, w in f.named_parameters():
        #         if 'l1' in name:
        #             nn.init.uniform_(w, a=lb1, b=ub1)
        #         else:
        #             nn.init.uniform_(w, a=lb2, b=ub2)
        elif init_name == 'mu_vecs':
            std1, std2 = task_gen_params['std1'], task_gen_params['std2']
            for name, w in f.named_parameters():
                # print(f'std1 = {std1}, std2 = {std2} (name = {name})')
                if 'fc' in name:
                    D_flat = w.numel()
                    # get the center mu vec
                    mu_vec = task_gen_params['mu_vecs'][name].view(1, D_flat)
                    if 'l1' in name and 'fc' in name:
                        m = torch.distributions.MultivariateNormal(mu_vec, std1 * torch.eye(D_flat))
                        w_task = m.sample()
                        w.data = w_task.view(w.size())
                    elif 'l2' in name and 'fc' in name:
                        m = torch.distributions.MultivariateNormal(mu_vec, std2 * torch.eye(D_flat))
                        w_task = m.sample()
                        w.data = w_task.view(w.size())
        else:
            raise ValueError(f'Has not been implemented: target_f_name={target_f_name}')


def get_new_mu_vec_for_new_benchmark(f, task_gen_params):
    """
    gets the mu vec for the f_avg model (per layer)
    """
    for name, w in f.named_parameters():
        if 'fc' in name:
            loc, scale = task_gen_params['mu_vec_loc'], task_gen_params['mu_vec_scale']
            m = torch.distributions.Normal(loc=loc, scale=scale)
            mu = m.sample(w.size())
            task_gen_params['mu_vecs'][name] = mu


def get_embedding(x, f):
    # https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/3
    # apply f until the last layer, instead return that as the embedding
    out = x
    for name, m in f.named_children():
        if 'final' in name:
            return out
        if 'l2' in name and 'final' not in name:  # cuz I forgot to write final...sorry!
            return out
        out = m(out)
    raise ValueError('error in getting emmbedding')


def check_average_function(
        task_gen_params,
        num_samples_per_task,
        lb=-1, ub=1,
        plot=False,
        save_plot=False,
):
    f_rand = get_backbone(task_gen_params)
    f_target = get_backbone(task_gen_params)
    initialize_average_function(f_target, task_gen_params)
    f_init = get_backbone(task_gen_params)
    initialize_average_function(f_init, task_gen_params)
    # get input range x
    Din = task_gen_params['Din']
    Dout = task_gen_params['Dout']
    x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
    # get true target y from f_avg
    # noise = torch_uu.tensor(0.0)
    noise_std = task_gen_params['noise_std']
    noise = torch.torch.distributions.normal.Normal(loc=0, scale=noise_std).sample((num_samples_per_task, Dout))
    y = f_target(x).detach().cpu().numpy() + noise.detach().cpu().numpy()
    rand_indices = torch.randperm(x.size(0))
    x_spt = x[rand_indices[:5], :]
    x_qry = x[rand_indices[5:20], :]
    y_spt = y[rand_indices[:5], :]
    y_qry = y[rand_indices[5:20], :]
    # Do PFF
    x_embedding = get_embedding(x, f_init).detach()
    mdl = LinearRegression().fit(x_embedding, y)
    y_pred = torch.tensor(mdl.predict(x_embedding))
    # y_pred = torch_uu.tensor(mdl.predict(x_embedding), dtype=torch_uu.double)
    y = torch.tensor(y)
    loss = nn.MSELoss()(y, y_pred)
    # Do PFF with spt,qry data
    x_embedding_spt = get_embedding(x_spt, f_init).detach()
    x_embedding_qry = get_embedding(x_qry, f_init).detach()
    mdl = LinearRegression().fit(x_embedding_spt, y_spt)
    y_pred_qry = torch.tensor(mdl.predict(x_embedding_qry))
    y_qry = torch.tensor(y_qry)
    loss_qrt = nn.MSELoss()(y_qry, y_pred_qry)
    # calculate gradient estimates in interval
    # nn.MSELoss()(f_rand(x), y).backward()
    nn.MSELoss()(f_rand(x_qry), y_qry).backward()
    # override print function
    split = task_gen_params['split']
    path = task_gen_params['metaset_path'] / f"f_avg_stats_{split}.txt"
    print = lambda printable: my_print(printable, filepath=path)
    print(f'split = {split}')
    # print gradient stats of f
    for name, w in f_rand.named_parameters():
        if 'fc' in name:
            print(f'grad.mean = {w.grad.mean()} +- {w.grad.std()}, layer name: {name}, ')
    # print intermediate values of f (check relus aren't always positive
    f_init.zero_grad()
    # print counts of how many relus activate
    out = x
    for name, m in f_init.named_children():
        out = m(out)
        # print how many pre-acts are greater than zero
        if 'bn' in name:
            # if the nn has bn then that is the pre-act
            print(f'count(out>zero) = {(out > 0.0).sum()}/{out.numel()}, layer name: {name}')
        elif 'fc' in name:
            # if the nn has no bn then the pre-act is fc (assuming the target nn is not weird)
            print(f'count(out>zero) = {(out > 0.0).sum()}/{out.numel()}, layer name: {name}')
    # print info about f
    print(f'PFF loss (full dataset) = {loss}')
    print(f'PFF loss (Qry set) = {loss_qrt}')
    # print(f'cond X = {np.linalg.cond(x_embedding)} (infinity only means ill-posed)')
    print(f'y.min() = {y.min()}')
    print(f'y.max() = {y.max()}')
    print(f'y.mean() = {y.mean()}')
    print(f'np.median(y) = {np.median(y)}')
    # check number of params for target task f_i is small that total # of samples for task f_i
    num_params = sum(p.numel() for p in f_init.parameters())
    print(f'number of params of f_i = {num_params}')
    print(f'enough data? (more data than params for task f_i): {num_params < num_samples_per_task}')
    # visualize target function
    plt.figure()
    plt.scatter(x, y)
    plt.ylabel('f(x)')
    plt.xlabel('x (raw feature)')
    path = task_gen_params['metaset_path'] / f"{split}.pdf"
    plt.savefig(path) if save_plot else None
    plt.show() if plot else None
    # print(task_gen_params)


def generate_regression_tasks(
        metaset_split_path,
        task_gen_params,
        num_samples_per_task,
        num_tasks,
        target_f_name,
        lb=-1, ub=1
):
    # get empty backbone/arch
    f = get_backbone(task_gen_params)
    # save average model
    save_average_model(metaset_split_path, f, task_gen_params)
    # get input range x
    Din = task_gen_params['Din']
    Dout = task_gen_params['Dout']
    x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
    for task_id in range(num_tasks):
        # print(f'task_id = {task_id}')
        # sample target f: task ~ p(t; task_gen_params)
        norm_f = norm(f, 2)
        initialize_target_function(f, task_gen_params)
        assert (norm(f) != norm_f)  # makes sure f_i != f

        # apply target task function f_i to inputs
        y = f(x)

        # create folder for tasks f_i
        path_2_taskfolder = metaset_split_path / f'fi_{target_f_name}_norm_f_{norm(f, 2)}/'
        path_2_taskfolder.mkdir(parents=True, exist_ok=True)

        # save data & target function to it's db file
        FNAME = 'fi_db.pt'  # DO NOT CHANGE NAME OF fi_db
        task_filename = path_2_taskfolder / FNAME
        noise_std = task_gen_params['noise_std']
        noise = torch.torch.distributions.normal.Normal(loc=0, scale=noise_std).sample((num_samples_per_task, Dout))
        x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        y_noisy = y_np + noise.detach().cpu().numpy()
        db = {'x': x_np, 'y': y_noisy, 'y_no_noise': y_np, 'task_gen_params': task_gen_params,
              'f': f,
              'f_state_dict': f.state_dict(),
              'f_str': str(f),
              'f_modules': f._modules,
              'f_modules_str': str(f._modules)
              }
        torch.save(db, task_filename)
    print('Data set creation done!')


def main():
    """
    In this experiment we want to show that inner adaptation plays a big role when the meta-learning
    problem to be solve has the L,S structure i.e. the first layer has large std while the
    second does not.

    std1 is changing from small to large. As the std2 for increase the help inner adaptation improves
    until we get to some limit.
    """
    # not sure if this flag works...leave False for now
    # debug = True
    debug = False
    # root path to LS main meta-learning task
    target_f_name = 'fully_connected_NN'
    # target_f_name = 'fully_connected_NN_with_BN'
    noise_std = 1e-1
    dataset_name = f'dataset_LS_DEBUG_{target_f_name}'
    # dataset_name = f'dataset_LS_{target_f_name}_exponential_episodic'
    # dataset_name = f'dataset_LS_{target_f_name}'  # e.g. dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15

    # tasks params
    num_samples_per_task = 1_000
    num_tasks = 200

    # mu vec params, mu_vec = N(loc, scale).vec()
    mu_vec_loc = 0.0
    mu_vec_scale = 1.0

    # std1 is changing from small to large
    # stds1 = [1e-16, 1e-8, 1e-4, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 32.0]
    stds1 = [1e-4, 1e-1, 4, 16]
    data_idx = 3

    # layer part 2
    std2 = 1.0

    # params for generating the meta-learning function for tasks
    std1 = stds1[data_idx]
    print(f'std for first half (l1) = {std1}')
    task_gen_params = {'std1': std1, 'std2': std2}
    # path to folder for (collection) of meta-learning tasks (with target function data-sets = regression task)

    # params for backbone
    Din, Dout = 1, 1
    # H = 10
    H = 15
    # 5 layers, 4 hidden layers
    # hidden_dim = [(Din, H), (H, H), (H, H), (H, H), (H, Dout)]
    # 4 layers, 3 hidden layers
    hidden_dim = [(Din, H), (H, H), (H, H), (H, Dout)]
    # 3 layers, 2 hidden layers
    # hidden_dim = [(Din, H), (H, H), (H, Dout)]
    nb_hidden_layers = len(hidden_dim) - 1
    print(f'# of hidden layers = {nb_hidden_layers}')
    print(f'total layers = {len(hidden_dim)}')
    section_label = [1] * (len(hidden_dim) - 1) + [2]
    task_gen_params = {
        'dataset_name': dataset_name,
        'metaset_path': None,
        'mu_vec_loc': mu_vec_loc,
        'mu_vec_scale': mu_vec_scale,
        'target_f_name': target_f_name,
        'hidden_dim': hidden_dim,
        'section_label': section_label,
        'Din': Din, 'Dout': Dout, 'H': H,
        'init_name': 'mu_vecs',
        'noise_std': noise_std,
        'mu_vecs': {},
        **task_gen_params}

    # it's just for printing the bb before generating the tasks
    task_gen_params['split'] = 'train'
    print(get_backbone(task_gen_params))
    # generate the regressions tasks for meta-learning i.e. the target function + their examples
    # splits = ['train']
    splits = ['train', 'val', 'test']
    for split in splits:
        # generate mu vec for current split
        print(f'\n --> split = {split}')
        task_gen_params['split'] = split
        f = get_backbone(task_gen_params)
        get_new_mu_vec_for_new_benchmark(f, task_gen_params)
        # create path to current meta-set and meta-set/split
        print('about to create meta_set...')
        dataset_path = Path(f'~/data/{dataset_name}_nb_tasks{num_tasks}_data_per_task{num_samples_per_task}_l_{len(hidden_dim)}_nb_h_layes{nb_hidden_layers}_out1_H{H}').expanduser()
        metaset_path = dataset_path / f"meta_set_{target_f_name}" \
                       f"_std1_{task_gen_params['std1']}" \
                       f"_std2_{task_gen_params['std2']}" \
                       f"_noise_std{noise_std}" \
                       f"nb_h_layes{nb_hidden_layers}_out1_H{H}/"
        metaset_split_path = metaset_path / split
        task_gen_params['metaset_path'] = metaset_path
        metaset_split_path.mkdir(parents=True, exist_ok=True)
        # check stats of data set
        check_average_function(
            task_gen_params,
            num_samples_per_task,
            lb=-1, ub=1,
            plot=True,
            save_plot=True
        )
        # create dataset
        generate_regression_tasks(
            metaset_split_path,
            task_gen_params,
            num_samples_per_task,
            num_tasks,
            target_f_name)
    # save params (as strings) used to generate data set and tasks
    path = task_gen_params['metaset_path'] / f"task_gen_params.json"
    with open(path, 'w+') as file:
        task_gen_params['dataset_path'] = dataset_path
        task_gen_params['arch_of_tasks'] = f
        task_gen_params['split'] = splits
        # make all params for generating tasks string
        task_gen_params = {key: str(value) for (key, value) in task_gen_params.items()}
        json.dump(task_gen_params, file, indent=4)


if __name__ == '__main__':
    # these are the numbers we need to generate the data points
    # metaset_dataset = Sinusoid(num_samples_per_task=shots + test_shots, num_tasks=100, noise_std=None)
    # metaset_miniimagenet = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True, download=True)
    main()
    print('Main Done! \a')
