'''
DO NOT USE THIS ONE.

This code creates a data set where the average function has the same value for all the weights
instead we want a random vector to be initialized for the weights.

'''
import torch
import torch.nn as nn

from collections import OrderedDict

from pathlib import Path

from sklearn.linear_model import LinearRegression

import numpy as np

from matplotlib import pyplot as plt

def norm(f, l=2):
    return sum([w.detach().norm(l) for w in f.parameters()])

def get_backbone(task_gen_params):
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
                fc = (f'fc{idx};l{section}', nn.Linear(Din, Dout))
                act = (f'relu{idx}', nn.ReLU())
                layers.extend([fc, act])
            else:
                fc = (f'fc{idx};final;l{section}', nn.Linear(Din, Dout))
                layers.extend([fc])
        f = nn.Sequential(OrderedDict(layers))
        return f
    elif backbone_name == 'fully_connected_NN_with_BN':
        hidden_dim = task_gen_params['hidden_dim']
        section_label = task_gen_params['section_label']
        layers = []
        for i, (Din, Dout) in enumerate(hidden_dim):
            section = section_label[i]
            idx = i + 1
            if i != len(hidden_dim) - 1:
                fc = (f'fc{idx};l{section}', nn.Linear(Din, Dout))
                bn = (f'bn{idx};l{section}', nn.BatchNorm1d(num_features=Dout, track_running_stats=False))
                act = (f'relu{idx}', nn.ReLU())
                layers.extend([fc, bn, act])
            else:
                fc = (f'fc{idx};final;l{section}', nn.Linear(Din, Dout))
                layers.extend([fc])
        f = nn.Sequential(OrderedDict(layers))
        return f
    else:
        raise ValueError(f'Not implemented: backbone_name = {backbone_name}')


def save_average_model(path, f, task_gen_params):
    # initialize average model
    initialize_average_function(f, task_gen_params)
    # torch_uu save it
    path2avg_f = path / 'f_avg'
    torch.save({'f_avg': f}, path2avg_f)

def initialize_average_function(f, task_gen_params):
    init_name = task_gen_params['init_name']
    target_f_name = task_gen_params['target_f_name']
    with torch.no_grad():
        if init_name == 'debug' or target_f_name == 'debug':
            mu1 = task_gen_params['mu1']
            [nn.init.constant_(w, mu1) for w in f.parameters()]
        elif init_name == 'uniform':
            mu1, std1, mu2, std2 = task_gen_params['mu1'], task_gen_params['std1'], task_gen_params['mu2'], task_gen_params['std2']
            for name, w in f.named_parameters():
                w = nn.init.constant_(w, mu1)
                if 'l1' in name:
                    # mu1 = torch_uu.distributions.Normal(loc=0, scale=1.0).sample(sample_shape=w.size())
                    # nn.init.normal_(w, mean=0, std=1.0)
                    nn.init.constant_(w, mu1)  # BAD!
                    #val = torch_uu.distributions.MultivariateNormal(loc=mu, scale=torch_uu.eye(2))
                    #w.data = val
                else:
                    # mu2 = torch_uu.distributions.Normal(loc=0, scale=1.0).sample(sample_shape=w.size())
                    #nn.init.normal_(w, mean=0, std=1.0)
                    nn.init.constant_(w, mu2)  # BAD!
        else:
            raise ValueError(f'Has not been implemented: target_f_name={target_f_name}')


def initialize_target_function(f, task_gen_params):
    init_name = task_gen_params['init_name']
    target_f_name = task_gen_params['target_f_name']
    if init_name == 'debug' or target_f_name == 'debug':
        mu1, std1 = task_gen_params['mu1'], task_gen_params['std1']
        lb1, ub1 = mu1 - std1, mu1 + std1
        [nn.init.uniform_(w, a=lb1, b=ub1) for w in f.parameters()]
    elif init_name == 'uniform':
        mu1, std1, mu2, std2 = task_gen_params['mu1'], task_gen_params['std1'], task_gen_params['mu2'], task_gen_params['std2']
        lb1, ub1, lb2, ub2 = mu1 - std1, mu1 + std1, mu2 - std2, mu2 + std2
        for name, w in f.named_parameters():
            if 'l1' in name:
                nn.init.uniform_(w, a=lb1, b=ub1)
            else:
                nn.init.uniform_(w, a=lb2, b=ub2)
    else:
        raise ValueError(f'Has not been implemented: target_f_name={target_f_name}')

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
        plot=False
):
    f_target = get_backbone(task_gen_params)
    initialize_average_function(f_target, task_gen_params)
    f_init = get_backbone(task_gen_params)
    initialize_average_function(f_init, task_gen_params)
    # get input range x
    Din = task_gen_params['Din']
    x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
    # get true target y from f_avg
    # noise = torch_uu.tensor(0.0)
    noise_std = task_gen_params['noise_std']
    noise = torch.torch.distributions.normal.Normal(loc=0, scale=noise_std).sample((num_samples_per_task, Din))
    y = f_target(x).detach().cpu().numpy() + noise.detach().cpu().numpy()
    rand_indices = torch.randperm(x.size(0))
    x_spt = x[rand_indices[:5], :]
    x_qry = x[rand_indices[5:20], :]
    y_spt = y[rand_indices[:5], :]
    y_qry = y[rand_indices[5:20], :]
    # Do PFF on all data set
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
    # calculate average gradient in interval
    f_init(x).sum().backward()
    for name, w in f_init.named_parameters():
        if 'fc' in name:
            print(f'grad.mean = {w.grad.mean()} +- {w.grad.std()}, layer name: {name}, ')
    # print intermediate values of f (check relus aren't always positive
    f_init.zero_grad()
    out = x
    for name, m in f_init.named_children():
        out = m(out)
        if 'bn' in name:
            print(f'count(out<zero) = {(out<0.0).sum()}/{out.numel()}, layer name: {name}')
    # print info about f
    print(f'PFF loss (full dataset) = {loss}')
    print(f'PFF loss (Qry set) = {loss_qrt}')
    # print(f'cond X = {np.linalg.cond(x_embedding)} (infinity only means ill-posed)')
    print(f'y.min() = {y.min()}')
    print(f'y.max() = {y.max()}')
    print(f'y.mean() = {y.mean()}')
    print(f'np.median(x) = {np.median(y)}')
    #
    num_params = sum(p.numel() for p in f_init.parameters())
    print(f'number of params of f = {num_params}')
    #
    plt.scatter(x, y)
    plt.figure()
    #plt.hist(y)
    plt.show() if plot else None


def generate_regression_tasks(
        path,
        task_gen_params,
        num_samples_per_task,
        num_tasks,
        lb=-1, ub=1,
        target_f_name='debug'
):
    # get empty backbone/arch
    f = get_backbone(task_gen_params)
    # torch_uu.save({'f_backbone': f}, path / 'f_backbone')
    # save average model
    save_average_model(path, f, task_gen_params)
    # get input range x
    Din = task_gen_params['Din']
    x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
    #
    for task_id in range(num_tasks):
        #print(f'task_id = {task_id}')
        # sample target f: task ~ p(t; task_gen_params)
        norm_f = norm(f, 2)
        initialize_target_function(f, task_gen_params)
        assert(norm(f) != norm_f)  # makes sure f_i != f

        # apply target f to inputs
        y = f(x)

        # create folder for task
        path_2_taskfolder = path / f'f_{target_f_name}_norm_f_{norm(f, 2)}/'
        path_2_taskfolder.mkdir(parents=True, exist_ok=True)

        # save data & target function to it's db file
        task_filename = path_2_taskfolder / 'db'
        x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        db = {'x': x_np, 'y': y_np, 'f': f}
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
    debug = True
    # debug = False
    # root path to LS main meta-learning task
    target_f_name = 'fully_connected_NN'
    # target_f_name = 'fully_connected_NN_with_BN'
    noise_std = 1e-1
    dataset_name = f'LS_{target_f_name}_noise_std{noise_std}'
    path = Path(f'~/data/{dataset_name}/').expanduser() if not debug else Path(f'~/data/{dataset_name}_debug/').expanduser()

    # tasks params
    num_samples_per_task = 1_000
    num_tasks = 500

    # mu1 is fixed but std1 is changing from small to large
    mu1 = 1.0
    print(f'mu1 = {mu1}')
    stds1 = [0.1, 0.125, 0.1875, 0.2, 0.25]

    # layer part 2
    mu2 = 1.0
    std2 = 0.025  # fixed and small, std2 is small so that adaptation in the final layer isn't influential

    # params for generating the meta-learning function for tasks
    data_idx = 4
    std1 = stds1[data_idx]
    # mu1 = std1  # CAREFUL UNCOMMENTING THIS
    print(f'std for first half (l1) = {std1}')
    task_gen_params = {'mu1': mu1, 'std1': std1, 'mu2': mu2, 'std2': std2}
    # path to folder for (collection) of meta-learning tasks (with target function data-sets = regression task)

    # params for backbone

    Din, Dout = 1, 1
    # 5 layers, 4 hidden layers
    # hidden_dim = [(Din, 10), (10, 10), (10, 10), (10, 10), (10, Dout)]
    # 4 layers, 3 hidden layers
    hidden_dim = [(Din, 10), (10, 10), (10, 10), (10, Dout)]
    # # 3 layers, 2 hidden layers
    # hidden_dim = [(Din, 10), (10, 10), (10, Dout)]
    print(f'# of hidden layers = {len(hidden_dim) - 1}')
    print(f'total layers = {len(hidden_dim)}')
    section_label = [1]*(len(hidden_dim)-1) + [2]
    task_gen_params = {
        'target_f_name': target_f_name,
        'hidden_dim': hidden_dim,
        'section_label': section_label,
        'Din': Din, 'Dout': Dout,
        'init_name': 'uniform',
        'noise_std': noise_std,
        **task_gen_params}
    # check stats about dataset
    check_average_function(
        task_gen_params,
        num_samples_per_task,
        lb=-1, ub=1,
        plot=True
    )

    # generate the regressions tasks for meta-learning i.e. the target function + their examples
    # split = 'train'
    # split = 'val'
    # split = 'test'
    # print(f'split = {split}')
    # path = path / f"{target_f_name}_mu1_{task_gen_params['mu1']}" \
    #               f"_std1_{task_gen_params['std1']}_mu2_{task_gen_params['mu2']}" \
    #               f"_std2_{task_gen_params['std2']}_noise_std{noise_std}/"
    # path = path / split
    # path.mkdir(parents=True, exist_ok=True)
    # print('about to create dataset...')
    # generate_regression_tasks(
    #     path,
    #     task_gen_params,
    #     num_samples_per_task,
    #     num_tasks)


if __name__ == '__main__':
    print('DO NOT USE THIS CODE')
    assert(False)
    # these are the numbers we need to generate the data points
    # metaset_dataset = Sinusoid(num_samples_per_task=shots + test_shots, num_tasks=100, noise_std=None)
    # metaset_miniimagenet = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True, download=True)
    # main()
    print('Main Done! \a')
