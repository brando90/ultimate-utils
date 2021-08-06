# %%

# to test impots
import sys
from typing import List, NewType

for path in sys.path:
    print(path)


# %%
def __path_bn_layer_for_functional_eval(self, module, input):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        print(target_attr)
        if type(target_attr) == torch.nn.BatchNorm1d:
            target_attr.track_running_stats = True
            target_attr.running_mean = input.mean()
            target_attr.running_var = input.var()
            target_attr.num_batches_tracked = torch.tensor(0, dtype=torch.long)

    # "recurse" iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        self._path_bn_layer_for_functional_eval(immediate_child_module, name)


# %%

import time

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('employee.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class Employee:
    """A sample Employee class"""

    def __init__(self, first, last):
        self.first = first
        self.last = last

        logger.info('Created Employee: {} - {}'.format(self.fullname, self.email))

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)


emp_1 = Employee('John', 'Smith')
emp_2 = Employee('Corey', 'Schafer')
emp_3 = Employee('Jane', 'Doe')


######## END OF EMPLOYEE LOGGING EXAMPLE

def report_times(start, verbose=False):
    '''
    How much time has passed since the time "start"

    :param float start: the number representing start (usually time.time())
    '''
    meta_str = ''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds / 60
    hours = minutes / 60
    if verbose:
        print(f"--- {seconds} {'seconds ' + meta_str} ---")
        print(f"--- {minutes} {'minutes ' + meta_str} ---")
        print(f"--- {hours} {'hours ' + meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours


#
# def params_in_comp_graph():
#     import torch
#     import torch.nn as nn
#     # from torchviz import make_dot
#     fc0 = nn.Linear(in_features=3, out_features=1)
#     params = [('fc0', fc0)]
#     mdl = nn.Sequential(OrderedDict(params))
#
#     x = torch.randn(1, 3)
#     # x.requires_grad = True  # uncomment to put in computation graph
#     y = torch.randn(1)
#
#     l = (mdl(x) - y) ** 2
#
#     # make_dot(l, params=dict(mdl.named_parameters()))
#     params = dict(mdl.named_parameters())
#     # params = {**params, 'x':x}
#     make_dot(l, params=params).render('data/debug/test_img_l', format='png')


def check_if_tensor_is_detached():
    a = torch.tensor([2.0], requires_grad=True)
    b = a.detach()
    b.requires_grad = True
    print(a == b)
    print(a is b)
    print(a)
    print(b)

    la = (5.0 - a) ** 2
    la.backward()
    print(f'a.grad = {a.grad}')

    lb = (6.0 - b) ** 2
    lb.backward()
    print(f'b.grad = {b.grad}')


def deep_copy_issue():
    params = OrderedDict([('fc1', nn.Linear(in_features=3, out_features=1))])
    mdl0 = nn.Sequential(params)
    mdl1 = copy.deepcopy(mdl0)
    print(id(mdl0))
    print(mdl0)
    print(id(mdl1))
    print(mdl1)
    # my update
    mdl1.fc1.weight = nn.Parameter(mdl1.fc1.weight + 1)
    mdl2 = copy.deepcopy(mdl1)
    print(id(mdl2))
    print(mdl2)


def download_mini_imagenet():
    # download mini-imagenet automatically
    import torch
    import torch.nn as nn
    import torchvision.datasets.utils as utils
    from torchvision.datasets.utils import download_and_extract_archive
    from torchvision.datasets.utils import download_file_from_google_drive

    ## download mini-imagenet
    # url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename = 'miniImagenet.tgz'
    root = '~/tmp/'  # dir to place downloaded file in
    download_file_from_google_drive(file_id, root, filename)


def extract():
    from torchvision.datasets.utils import extract_archive
    from_path = os.path.expanduser('~/Downloads/miniImagenet.tgz')
    extract_archive(from_path)


def download_and_extract_miniImagenet(root):
    import os
    from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

    ## download miniImagenet
    # url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename = 'miniImagenet.tgz'
    download_file_from_google_drive(file_id, root, filename)
    fpath = os.path.join(root, filename)  # this is what download_file_from_google_drive does
    ## extract downloaded dataset
    from_path = os.path.expanduser(fpath)
    extract_archive(from_path)
    ## remove the zip file
    os.remove(from_path)


def torch_concat():
    import torch

    g1 = torch.randn(3, 3)
    g2 = torch.randn(3, 3)


#
# def inner_loop1():
#     n_inner_iter = 5
#     inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)
#
#     qry_losses = []
#     qry_accs = []
#     meta_opt.zero_grad()
#     for i in range(task_num):
#         with higher.innerloop_ctx(
#                 net, inner_opt, copy_initial_weights=False
#         ) as (fnet, diffopt):
#             # Optimize the likelihood of the support set by taking
#             # gradient steps w.r.t. the model's parameters.
#             # This adapts the model's meta-parameters to the task.
#             # higher is able to automatically keep copies of
#             # your network's parameters as they are being updated.
#             for _ in range(n_inner_iter):
#                 spt_logits = fnet(x_spt[i])
#                 spt_loss = F.cross_entropy(spt_logits, y_spt[i])
#                 diffopt.step(spt_loss)
#
#             # The final set of adapted parameters will induce some
#             # final loss and accuracy on the query dataset.
#             # These will be used to update the model's meta-parameters.
#             qry_logits = fnet(x_qry[i])
#             qry_loss = F.cross_entropy(qry_logits, y_qry[i])
#             qry_losses.append(qry_loss.detach())
#             qry_acc = (qry_logits.argmax(
#                 dim=1) == y_qry[i]).sum().item() / querysz
#             qry_accs.append(qry_acc)
#
#             # Update the model's meta-parameters to optimize the query
#             # losses across all of the tasks sampled in this batch.
#             # This unrolls through the gradient steps.
#             qry_loss.backward()
#
#     meta_opt.step()
#     qry_losses = sum(qry_losses) / task_num
#     qry_accs = 100. * sum(qry_accs) / task_num
#     i = epoch + float(batch_idx) / n_train_iter
#     iter_time = time.time() - start_time


# def inner_loop2():
#     n_inner_iter = 5
#     inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)
#
#     qry_losses = []
#     qry_accs = []
#     meta_opt.zero_grad()
#     meta_loss = 0
#     for i in range(task_num):
#         with higher.innerloop_ctx(
#                 net, inner_opt, copy_initial_weights=False
#         ) as (fnet, diffopt):
#             # Optimize the likelihood of the support set by taking
#             # gradient steps w.r.t. the model's parameters.
#             # This adapts the model's meta-parameters to the task.
#             # higher is able to automatically keep copies of
#             # your network's parameters as they are being updated.
#             for _ in range(n_inner_iter):
#                 spt_logits = fnet(x_spt[i])
#                 spt_loss = F.cross_entropy(spt_logits, y_spt[i])
#                 diffopt.step(spt_loss)
#
#             # The final set of adapted parameters will induce some
#             # final loss and accuracy on the query dataset.
#             # These will be used to update the model's meta-parameters.
#             qry_logits = fnet(x_qry[i])
#             qry_loss = F.cross_entropy(qry_logits, y_qry[i])
#             qry_losses.append(qry_loss.detach())
#             qry_acc = (qry_logits.argmax(
#                 dim=1) == y_qry[i]).sum().item() / querysz
#             qry_accs.append(qry_acc)
#
#             # Update the model's meta-parameters to optimize the query
#             # losses across all of the tasks sampled in this batch.
#             # This unrolls through the gradient steps.
#             # qry_loss.backward()
#             meta_loss += qry_loss
#
#     qry_losses = sum(qry_losses) / task_num
#     qry_losses.backward()
#     meta_opt.step()
#     qry_accs = 100. * sum(qry_accs) / task_num
#     i = epoch + float(batch_idx) / n_train_iter
#     iter_time = time.time() - start_time


def error_unexpected_way_to_by_pass_safety():
    # https://stackoverflow.com/questions/62415251/why-am-i-able-to-change-the-value-of-a-tensor-without-the-computation-graph-know

    import torch
    a = torch.tensor([1, 2, 3.], requires_grad=True)
    # are detached tensor's leafs? yes they are
    a_detached = a.detach()
    # a.fill_(2) # illegal, warns you that a tensor which requires grads is used in an inplace op (so it won't be recorded in computation graph so it wont take the right derivative of the forward path as this op won't be in it)
    a_detached.fill_(
        2)  # weird that this one is allowed, seems to allow me to bypass the error check from the previous comment...?!
    print(f'a = {a}')
    print(f'a_detached = {a_detached}')
    a.sum().backward()


def detach_playground():
    import torch

    a = torch.tensor([1, 2, 3.], requires_grad=True)
    # are detached tensor's leafs? yes they are
    a_detached = a.detach()
    print(f'a_detached.is_leaf = {a_detached.is_leaf}')
    # is doing sum on the detached tensor a leaf? no
    a_detached_sum = a.sum()
    print(f'a_detached_sum.is_leaf = {a_detached_sum.is_leaf}')
    # is detaching an intermediate tensor a leaf? yes
    a_sum_detached = a.sum().detach()
    print(f'a_sum_detached.is_leaf = {a_sum_detached.is_leaf}')
    # shows they share they same data
    print(f'a == a_detached = {a == a_detached}')
    print(f'a is a_detached = {a is a_detached}')
    a_detached.zero_()
    print(f'a = {a}')
    print(f'a_detached = {a_detached}')
    # a.fill_(2) # illegal, warns you that a tensor which requires grads is used in an inplace op (so it won't be recorded in computation graph so it wont take the right derivative of the forward path as this op won't be in it)
    a_detached.fill_(
        2)  # weird that this one is allowed, seems to allow me to bypass the error check from the previous comment...?!
    print(f'a = {a}')
    print(f'a_detached = {a_detached}')
    ## conclusion: detach basically creates a totally new tensor which cuts gradient computations to the original but shares the same memory with original
    out = a.sigmoid()
    out_detached = out.detach()
    out_detached.zero_()
    out.sum().backward()


def clone_playground():
    import torch

    a = torch.tensor([1, 2, 3.], requires_grad=True)
    a_clone = a.clone()
    print(f'a_clone.is_leaf = {a_clone.is_leaf}')
    print(f'a is a_clone = {a is a_clone}')
    print(f'a == a_clone = {a == a_clone}')
    print(f'a = {a}')
    print(f'a_clone = {a_clone}')
    # a_clone.fill_(2)
    a_clone.mul_(2)
    print(f'a = {a}')
    print(f'a_clone = {a_clone}')
    a_clone.sum().backward()
    print(f'a.grad = {a.grad}')


def clone_vs_deepcopy():
    import copy
    import torch

    x = torch.tensor([1, 2, 3.])
    x_clone = x.clone()
    x_deep_copy = copy.deepcopy(x)
    #
    x.mul_(-1)
    print(f'x = {x}')
    print(f'x_clone = {x_clone}')
    print(f'x_deep_copy = {x_deep_copy}')
    print()


def inplace_playground():
    import torch

    x = torch.tensor([1, 2, 3.], requires_grad=True)
    y = x + 1
    print(f'x.is_leaf = {x.is_leaf}')
    print(f'y.is_leaf = {y.is_leaf}')
    x += 1  # not allowed because x is a leaf, since changing the value of a leaf with an inplace forgets it's value then backward wouldn't work IMO (though its not the official response)
    print(f'x.is_leaf = {x.is_leaf}')


def copy_initial_weights_playground_original():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import higher
    import numpy as np

    np.random.seed(1)
    torch.manual_seed(3)
    N = 100
    actual_multiplier = 3.5
    meta_lr = 0.00001
    loops = 5  # how many iterations in the inner loop we want to do

    x = torch.tensor(np.random.random((N, 1)), dtype=torch.float64)  # features for inner training loop
    y = x * actual_multiplier  # target for inner training loop
    model = nn.Linear(1, 1, bias=False).double()  # simplest possible model - multiple input x by weight w without bias
    meta_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)

    def run_inner_loop_once(model, verbose, copy_initial_weights):
        lr_tensor = torch.tensor([0.3], requires_grad=True)
        momentum_tensor = torch.tensor([0.5], requires_grad=True)
        opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
        with higher.innerloop_ctx(model, opt, copy_initial_weights=copy_initial_weights,
                                  override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
            for j in range(loops):
                if verbose:
                    print('Starting inner loop step j=={0}'.format(j))
                    print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(
                        list(fmodel.parameters(time=j)))))
                    print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (
                            list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
                out = fmodel(x)
                if verbose:
                    print(
                        '    Notice how `out` is `x` multiplied by the latest version of weight: {0:.4} * {1:.4} == {2:.4}'.format(
                            x[0, 0].item(), list(fmodel.parameters())[0].item(), out[0].item()))
                loss = ((out - y) ** 2).mean()
                diffopt.step(loss)

            if verbose:
                # after all inner training let's see all steps' parameter tensors
                print()
                print("Let's print all intermediate parameters versions after inner loop is done:")
                for j in range(loops + 1):
                    print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                print()

            # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
            weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
            meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier) ** 2
            print('  Final meta-loss: {0}'.format(meta_loss.item()))
            meta_loss.backward()  # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
            if verbose:
                print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad,
                                                                                                momentum_tensor.grad))
                print(
                    '  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
            return meta_loss.item()

    print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
    meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
        list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
    meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
        list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
    final_meta_gradient = list(model.parameters())[0].grad.item()
    # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
    # a bunch of hand-wavy statements and difficult to read code.
    # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
    meta_opt.step()
    meta_opt.zero_grad()
    meta_step = - meta_lr * final_meta_gradient  # how much meta_opt actually shifted inital weight value
    meta_loss_val3 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)


def copy_initial_weights_playground():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import higher
    import numpy as np

    np.random.seed(1)
    torch.manual_seed(3)
    N = 100
    actual_multiplier = 3.5  # the parameters we want the model to learn
    meta_lr = 0.00001
    loops = 5  # how many iterations in the inner loop we want to do

    x = torch.randn(N, 1)  # features for inner training loop
    y = x * actual_multiplier  # target for inner training loop
    model = nn.Linear(1, 1,
                      bias=False)  # model(x) = w*x, simplest possible model - multiple input x by weight w without bias. goal is to w~~actualy_multiplier
    outer_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)

    def run_inner_loop_once(model, verbose, copy_initial_weights):
        lr_tensor = torch.tensor([0.3], requires_grad=True)
        momentum_tensor = torch.tensor([0.5], requires_grad=True)
        inner_opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=copy_initial_weights,
                                  override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
            for j in range(loops):
                if verbose:
                    print('Starting inner loop step j=={0}'.format(j))
                    print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(
                        list(fmodel.parameters(time=j)))))
                    print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (
                            list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
                out = fmodel(x)
                if verbose:
                    print(
                        f'    Notice how `out` is `x` multiplied by the latest version of weight: {x[0, 0].item()} * {list(fmodel.parameters())[0].item()} == {out[0].item()}')
                loss = ((out - y) ** 2).mean()
                diffopt.step(loss)

            if verbose:
                # after all inner training let's see all steps' parameter tensors
                print()
                print("Let's print all intermediate parameters versions after inner loop is done:")
                for j in range(loops + 1):
                    print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                print()

            # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
            weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
            meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier) ** 2
            print('  Final meta-loss: {0}'.format(meta_loss.item()))
            meta_loss.backward()  # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
            if verbose:
                print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad,
                                                                                                momentum_tensor.grad))
                print(
                    '  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
            return meta_loss.item()

    print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
    meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
        list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
    meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
        list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
    final_meta_gradient = list(model.parameters())[0].grad.item()
    # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
    # a bunch of hand-wavy statements and difficult to read code.
    # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
    outer_opt.step()
    outer_opt.zero_grad()
    meta_step = - meta_lr * final_meta_gradient  # how much meta_opt actually shifted inital weight value
    meta_loss_val3 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)

    meta_loss_gradient_approximation = (meta_loss_val3 - meta_loss_val2) / meta_step

    print()
    print(
        'Side-by-side meta_loss_gradient_approximation and gradient computed by `higher` lib: {0:.4} VS {1:.4}'.format(
            meta_loss_gradient_approximation, final_meta_gradient))


def tqdm_torchmeta():
    from torchvision.transforms import Compose, Resize, ToTensor

    import torchmeta
    from torchmeta.datasets.helpers import miniimagenet

    from pathlib import Path
    from types import SimpleNamespace

    from tqdm import tqdm

    ## get args
    args = SimpleNamespace(episodes=5, n_classes=5, k_shot=5, k_eval=15, meta_batch_size=1, n_workers=4)
    args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()

    ## get meta-batch loader
    train_transform = Compose([Resize(84), ToTensor()])
    dataset = miniimagenet(
        args.data_root,
        ways=args.n_classes,
        shots=args.k_shot,
        test_shots=args.k_eval,
        meta_split='train',
        download=False)
    dataloader = torchmeta.utils.data.BatchMetaDataLoader(
        dataset,
        batch_size=args.meta_batch_size,
        num_workers=args.n_workers)

    with tqdm(dataset):
        print(f'len(dataloader)= {len(dataloader)}')
        for episode, batch in enumerate(dataloader):
            print(f'episode = {episode}')
            train_inputs, train_labels = batch["train"]
            print(f'train_labels[0] = {train_labels[0]}')
            print(f'train_inputs.size() = {train_inputs.size()}')
            pass
            if episode >= args.episodes:
                break


# if __name__ == "__main__":
#     start = time.time()
#     print('pytorch playground!')
#     # params_in_comp_graph()
#     # check_if_tensor_is_detached()
#     # deep_copy_issue()
#     # download_mini_imagenet()
#     # extract()
#     # download_and_extract_miniImagenet(root='~/tmp')
#     # download_and_extract_miniImagenet(root='~/automl-meta-learning/data')
#     # torch_concat()
#     # detach_vs_cloe()
#     # error_unexpected_way_to_by_pass_safety()
#     # clone_playground()
#     # inplace_playground()
#     # clone_vs_deepcopy()
#     # copy_initial_weights_playground()
#     tqdm_torchmeta()
#     print('--> DONE')
#     time_passed_msg, _, _, _ = report_times(start)
#     print(f'--> {time_passed_msg}')

# %%

import sys

print(sys.version)  ##
print(sys.path)


def helloworld():
    print('helloworld')
    print('hello12345')


def union_dicts():
    d1 = {'x': 1}
    d2 = {'y': 2, 'z': 3}
    d_union = {**d1, **d2}
    print(d_union)


def get_stdout_old():
    import sys

    # contents = ""
    # #with open('some_file.txt') as f:
    # #with open(sys.stdout,'r') as f:
    # # sys.stdout.mode = 'r'
    # for line in sys.stdout.readlines():
    #     contents += line
    # print(contents)

    # print(sys.stdout)
    # with open(sys.stdout.buffer) as f:
    #     print(f.readline())

    # import subprocess

    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # stdout = []
    # while True:
    #     line = p.stdout.readline()
    #     stdout.append(line)
    #     print( line )
    #     if line == '' and p.poll() != None:
    #         break
    # print( ''.join(stdout) )

    import sys
    myfile = "input.txt"

    def print(*args):
        __builtins__.print(*args, file=sys.__stdout__)
        with open(myfile, "a+") as f:
            __builtins__.print(*args, file=f)

    print('a')
    print('b')
    print('c')

    repr(sys.stdout)


def get_stdout():
    import sys
    myfile = "my_stdout.txt"

    # redefine print
    def print(*args):
        __builtins__.print(*args, file=sys.__stdout__)  # prints to terminal
        with open(myfile, "a+") as f:
            __builtins__.print(*args, file=f)  # saves in a file

    print('a')
    print('b')
    print('c')


def logging_basic():
    import logging
    logging.warning('Watch out!')  # will print a message to the console
    logging.info('I told you so')  # will not print anything


def logging_to_file():
    import logging
    logging.basicConfig(filename='example.log', level=logging.DEBUG)
    # logging.
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')


def logging_to_file_INFO_LEVEL():
    import logging
    import sys
    format = '{asctime}:{levelname}:{name}:lineno {lineno}:{message}'
    logging.basicConfig(filename='example.log', level=logging.INFO, format=format, style='{')
    # logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=format,style='{')
    # logging.
    logging.debug('This message should NOT go to the log file')
    logging.info('This message should go to log file')
    logging.warning('This, too')


def logger_SO_print_and_write_to_my_stdout():
    """My sample logger code to print to screen and write to file (the same thing).

    Note: trying to replace this old answer of mine using a logger:
    - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced

    Credit:
    - https://www.youtube.com/watch?v=jxmzY9soFXg&t=468s
    - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
    - https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716

    Other resources:
    - https://docs.python-guide.org/writing/logging/
    - https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
    - https://stackoverflow.com/questions/61084916/how-does-one-make-an-already-opened-file-readable-e-g-sys-stdout/61255375#61255375
    """
    from pathlib import Path
    import logging
    import os
    import sys
    from datetime import datetime

    ## create directory (& its parents) if it does not exist otherwise do nothing :)
    # get current time
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logs_dirpath = Path(f'~/logs/python_playground_logs_{current_time}/').expanduser()
    logs_dirpath.mkdir(parents=True, exist_ok=True)
    my_stdout_filename = logs_dirpath / Path('my_stdout.log')
    # remove my_stdout if it exists (note you can also just create a new log dir/file each time or append to the end of the log file your using)
    # os.remove(my_stdout_filename) if os.path.isfile(my_stdout_filename) else None

    ## create top logger
    logger = logging.getLogger(
        __name__)  # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
    logger.setLevel(
        logging.DEBUG)  # note: use logging.DEBUG, CAREFUL with logging.UNSET: https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716

    ## log to my_stdout.log file
    file_handler = logging.FileHandler(filename=my_stdout_filename)
    # file_handler.setLevel(logging.INFO) # not setting it means it inherits the logger. It will log everything from DEBUG upwards in severity to this handler.
    log_format = "{asctime}:{levelname}:{lineno}:{name}:{message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{')  # set the logging format at for this handler
    file_handler.setFormatter(fmt=formatter)

    ## log to stdout/screen
    stdout_stream_handler = logging.StreamHandler(
        stream=sys.stdout)  # default stderr, though not sure the advatages of logging to one or the other
    # stdout_stream_handler.setLevel(logging.INFO) # Note: having different set levels means that we can route using a threshold what gets logged to this handler
    log_format = "{name}:{levelname}:-> {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{')  # set the logging format at for this handler
    stdout_stream_handler.setFormatter(fmt=formatter)

    logger.addHandler(hdlr=file_handler)  # add this file handler to top logger
    logger.addHandler(hdlr=stdout_stream_handler)  # add this file handler to top logger

    logger.log(logging.NOTSET, 'notset')
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')


def logging_unset_level():
    """My sample logger explaining UNSET level

    Resources:
    - https://stackoverflow.com/questions/21494468/about-notset-in-python-logging
    - https://www.youtube.com/watch?v=jxmzY9soFXg&t=468s
    - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
    """
    import logging

    logger = logging.getLogger(
        __name__)  # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
    print(f'DEFAULT VALUE: logger.level = {logger.level}')

    file_handler = logging.FileHandler(filename='my_log.log')
    log_format = "{asctime}:{levelname}:{lineno}:{name}:{message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{')
    file_handler.setFormatter(fmt=formatter)

    stdout_stream_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_stream_handler.setLevel(logging.INFO)
    log_format = "{name}:{levelname}:-> {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(fmt=log_format, style='{')
    stdout_stream_handler.setFormatter(fmt=formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=stdout_stream_handler)

    logger.log(logging.NOTSET, 'notset')
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')


def logger():
    from pathlib import Path
    import logging

    # create directory (& its parents) if it does not exist otherwise do nothing :)
    logs_dirpath = Path('~/automl-meta-learning/logs/python_playground_logs/').expanduser()
    logs_dirpath.mkdir(parents=True, exist_ok=True)
    my_stdout_filename = logs_dirpath / Path('my_stdout.log')
    # remove my_stdout if it exists (used to have this but now I decided to create a new log & file each)
    # os.remove(my_stdout_filename) if os.path.isfile(my_stdout_filename) else None

    logger = logging.getLogger(
        __name__)  # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
    logger.setLevel(logging.INFO)

    log_format = "{asctime}:{levelname}:{name}:{message}"
    formatter = logging.Formatter(fmt=log_format, style='{')

    file_handler = logging.FileHandler(filename=my_stdout_filename)
    file_handler.setFormatter(fmt=formatter)

    logger.addHandler(hdlr=file_handler)
    logger.addHandler(hdlr=logging.StreamHandler())

    for i in range(3):
        logger.info(f'i = {i}')

    logger.info(f'logger DONE')


def logging_example_from_youtube():
    """https://github.com/CoreyMSchafer/code_snippets/blob/master/Logging-Advanced/employee.py
    """
    import logging
    # import pytorch_playground  # has employee class & code
    import sys

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler('sample.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.critical('not really critical :P')

    def add(x, y):
        """Add Function"""
        return x + y

    def subtract(x, y):
        """Subtract Function"""
        return x - y

    def multiply(x, y):
        """Multiply Function"""
        return x * y

    def divide(x, y):
        """Divide Function"""
        try:
            result = x / y
        except ZeroDivisionError:
            logger.exception('Tried to divide by zero')
        else:
            return result

    logger.info(
        'testing if log info is going to print to screen. it should because everything with debug or above is printed since that stream has that level.')

    num_1 = 10
    num_2 = 0

    add_result = add(num_1, num_2)
    logger.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

    sub_result = subtract(num_1, num_2)
    logger.debug('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))

    mul_result = multiply(num_1, num_2)
    logger.debug('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))

    div_result = divide(num_1, num_2)
    logger.debug('Div: {} / {} = {}'.format(num_1, num_2, div_result))


def plot():
    """
    source:
        - https://www.youtube.com/watch?v=UO98lJQ3QGI
        - https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Matplotlib/01-Introduction/finished_code.py
    """
    from matplotlib import pyplot as plt

    plt.xkcd()

    ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

    py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000,
                71496, 75370, 83640, 84666,
                84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708,
                108423, 101407, 112542, 122870, 120000]
    plt.plot(ages_x, py_dev_y, label='Python')

    js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674,
                68745, 68746, 74583, 79000,
                78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000,
                91660, 99240, 108000, 105000, 104000]
    plt.plot(ages_x, js_dev_y, label='JavaScript')

    dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928,
             67317, 68748, 73752, 77232,
             78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988,
             100000, 108923, 105000, 103117]
    plt.plot(ages_x, dev_y, color='#444444', linestyle='--', label='All Devs')

    plt.xlabel('Ages')
    plt.ylabel('Median Salary (USD)')
    plt.title('Median Salary (USD) by Age')

    plt.legend()

    plt.tight_layout()

    plt.savefig('plot.png')

    plt.show()


def subplot():
    """https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Matplotlib/10-Subplots/finished_code.py
    """

    import pandas as pd
    from matplotlib import pyplot as plt

    plt.style.use('seaborn')

    data = read_csv('data.csv')
    ages = data['Age']
    dev_salaries = data['All_Devs']
    py_salaries = data['Python']
    js_salaries = data['JavaScript']

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.plot(ages, dev_salaries, color='#444444',
             linestyle='--', label='All Devs')

    ax2.plot(ages, py_salaries, label='Python')
    ax2.plot(ages, js_salaries, label='JavaScript')

    ax1.legend()
    ax1.set_title('Median Salary (USD) by Age')
    ax1.set_ylabel('Median Salary (USD)')

    ax2.legend()
    ax2.set_xlabel('Ages')
    ax2.set_ylabel('Median Salary (USD)')

    plt.tight_layout()

    plt.show()

    fig1.savefig('fig1.png')
    fig2.savefig('fig2.png')


#
# def import_utils_test():
#     import uutils
#     # import uutils.utils as utils
#     # from uutils.utils import logger
#
#     print(uutils)
#     print(utils)
#     print(logger)
#
#     print()


def sys_path():
    """

    python -c "import sys; print(sys.path)”

    python -c "import sys; [print(p) for p in sys.path]"
    """
    import sys

    def path():
        import sys
        [print(p) for p in sys.path]

    for path in sys.path:
        print(path)


def pycharm_playground():
    import tqdm

    print('running pycharm playground...')

    b = 0
    print(b)
    print('Intermediate print line')
    print(b)
    print(b)
    print('Done!')


if __name__ == '__main__':
    # union_dicts()
    # get_stdout()
    # logger()
    # logger_SO_print_and_write_to_my_stdout()
    # logging_basic()
    # logging_to_file()
    # logging_to_file()
    # logging_to_file_INFO_LEVEL()
    # logging_example_from_youtube()
    # logging_unset_level()
    # import_utils_test()
    pycharm_playground()
    print('\n---> DONE\a\n\n')  ## HIii

# %%

import sys

print(sys.version)

# %%

## dictionary comprehension looping

d = {'a': 0, 'b': 1}
lst1 = [f'key:{k}' for k in d]
lst2 = [f'key:{k}, value:{v}' for k, v in d.items()]

print(lst1)
print(lst2)

# %%

## merging two dictionaries

d1 = {'a': 0, 'b': 1}
d2 = {'c': 2, 'd': 3}
d3 = {'e': 4, 'f': 5, 'g': 6}
d = {**d1, **d2, **d3}

print(d)

# %%


from collections import OrderedDict

od = OrderedDict([
    ('first', 1)
])

print(od)
od['first'] = 2
print(od)

lst = sum([i for i in range(3)])
print(lst)
od3 = OrderedDict([(i, i) for i in range(3)])
print(od3)
print(3 + float('Inf'))

# %%

# import pathlib
# from pathlib import Path
#
#
# def make_dirpath_current_datetime_hostname(path=None, comment='', replace_dots=True):
#     '''
#     make dir string: runs/CURRENT_DATETIME_HOSTNAME
#     '''
#     import socket
#     import os
#     from datetime import datetime
#     # check if root is a PosixPath object
#     if type(path) != pathlib.PosixPath and path is not None:
#         path = Path(path)
#     current_time = datetime.now().strftime('%b%d_%H-%M-%S')
#     log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
#     log_dir = Path(log_dir)
#     print(log_dir._str)
#     if replace_dots:
#         log_dir = Path(log_dir._str.replace('.', '_'))
#     if path is not None:
#         log_dir = path / log_dir
#     return log_dir
#
#
# print(type(Path('~')) == pathlib.PosixPath)
# print()
#
# log_dir = make_dirpath_current_datetime_hostname()
# print(log_dir)
# log_dir = make_dirpath_current_datetime_hostname('~')
# print(log_dir)
# log_dir = make_dirpath_current_datetime_hostname('~', '_jupyter')
# print(log_dir)
# log_dir = make_dirpath_current_datetime_hostname('~').expanduser()
# print(log_dir)
#
# string = "geeks for geeks geeks geeks geeks"
# # Prints the string by replacing geeks by Geeks
# print(string.replace("geeks", "Geeks"))
#
# log_dir = make_dirpath_current_datetime_hostname('~', '_jupyter', True)
# print(log_dir)

# %%

# adding keys to empty dic

d = {}
d['a'] = 3
print(d)

# %%

# unpack list?

(a, b, c) = [1, 2, 3]
print(a)


# %%

## kwargs

def f(*args, **kwargs):
    print(args)
    print(kwargs)


f()
f(1, 2, 3, a=1, b=2, c=3)

# %%

#
# import json
#
# from pathlib import Path
#
# p = Path('~/').expanduser()
# with open(p) as f:
#     data = json.load(f)
#     print(data)
#     print(data['password'])

# %%

import subprocess

from subprocess import Popen, PIPE, STDOUT

cmd = 'ls /etc/fstab /etc/non-existent-file'
p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
output = p.stdout.read()
print(output)

# %%

import sys

print('a')

print(sys.stdout)

# %%

# from pathlib import Path
#
#
# def send_email(subject, message, destination, password_path=None):
#     """ Send an e-mail from with message to destination email.
#
#     NOTE: if you get an error with google gmails you might need to do this:
#     https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
#     To use an app password:
#     https://stackoverflow.com/questions/60975490/how-does-one-send-an-e-mail-from-python-not-using-gmail
#
#     Arguments:
#         message {str} -- message string to send.
#         destination {str} -- destination email (as string)
#     """
#     from socket import gethostname
#     from email.message import EmailMessage
#     import smtplib
#     import json
#     import sys
#
#     server = smtplib.SMTP('smtp.gmail.com', 587)
#     smtplib.stdout = sys.stdout
#     server.starttls()
#     with open(password_path) as f:
#         config = json.load(f)
#         server.login('slurm.miranda@gmail.com', config['password'])
#
#         # craft message
#         msg = EmailMessage()
#
#         # message = f'{message}\nSend from Hostname: {gethostname()}'
#         # msg.set_content(message)
#         msg['Subject'] = subject
#         msg['From'] = 'slurm.miranda@gmail.com'
#         msg['To'] = destination
#         # send msg
#         server.send_message(msg)
#
#
# ##
# print("-------> HELLOWWWWWWWW")
# p = Path('~/automl-meta-learning/automl/experiments/pw_app.config.json').expanduser()
# send_email(subject='TEST: send_email2', message='MESSAGE', destination='brando.science@gmail.com', password_path=p)

# %%

"""
Demo of the errorbar function, including upper and lower limits
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["errorbar.capsize"] = 3

# https://stackoverflow.com/questions/61415955/why-dont-the-error-limits-in-my-plots-show-in-matplotlib

# example data
x = np.arange(0.5, 5.5, 0.5)
y = np.exp(-x)
xerr = 0.1
yerr = 0.2
ls = 'dotted'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# standard error bars
plt.errorbar(x, y, xerr=xerr, yerr=yerr, ls=ls, color='blue')

# including upper limits
uplims = np.zeros(x.shape)
uplims[[1, 5, 9]] = True
plt.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims, ls=ls,
             color='green')

# including lower limits
lolims = np.zeros(x.shape)
lolims[[2, 4, 8]] = True
plt.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims, ls=ls,
             color='red')

# including upper and lower limits
plt.errorbar(x, y + 1.5, marker='o', ms=8, xerr=xerr, yerr=yerr,
             lolims=lolims, uplims=uplims, ls=ls, color='magenta')

# including xlower and xupper limits
xerr = 0.2
yerr = np.zeros(x.shape) + 0.2
yerr[[3, 6]] = 0.3
xlolims = lolims
xuplims = uplims
lolims = np.zeros(x.shape)
uplims = np.zeros(x.shape)
lolims[[6]] = True
uplims[[3]] = True
plt.errorbar(x, y + 2.1, marker='o', ms=8, xerr=xerr, yerr=yerr,
             xlolims=xlolims, xuplims=xuplims, uplims=uplims, lolims=lolims,
             ls='none', mec='blue', capsize=0, color='cyan')

ax.set_xlim((0, 5.5))
ax.set_title('Errorbar upper and lower limits')
plt.show()

# %%

from types import SimpleNamespace
from pathlib import Path
from pprint import pprint

args = SimpleNamespace()
args.data_root = "~/automl-meta-learning/data/miniImagenet"

args.data_root = Path(args.data_root).expanduser()

print(args)

# pprint(dir(args.data_root))
print(args.data_root.name)
print('miniImagenet' in args.data_root.name)

# %%

## sampling N classes for len(meta-set)
# In sampling without replacement, each sample unit of
# the population has only one chance to be selected in the sample.
# because you are NOT replacing what you removed.

import random

N = 5
len_meta_set = 64
sample = random.sample(range(0, len_meta_set), N)

print(sample)

for i, n in enumerate(sample):
    print(f'i={i}\nn={n}\n')


# %%

# iterator https://www.programiz.com/python-programming/iterator

class Counter:

    def __init__(self, max=0):
        self.max = max  # returns up to and including that number

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.max:
            current_count = self.n
            self.n += 1
            print(f'current_count = {current_count}')
            print(f'self.n = {self.n}')
            print(self.n is current_count)
            return current_count
        else:
            raise StopIteration


## test it

counter = iter(Counter(max=0))
for count in counter:
    print(f'count = {count}')

# %%

from tqdm import tqdm

print(tqdm)

lst = range(3)
print(type(lst))

with tqdm(iter(lst), total=5) as tlist:
    print(f'tlist = {type(tlist)}')
    for i in tlist:
        print(i)

# %%

from tqdm import tqdm


class Plus2:

    def __init__(self, max=0):
        self.max = max  # returns up to and including that number

    def __iter__(self):
        self.it = 0
        self.tot = 0
        return self

    def __next__(self):
        if self.it <= self.max:
            self.it += 1
            self.tot += 2
            return self.tot
        else:
            raise StopIteration

    def __len__(self):
        return self.max


##
counter = iter(Plus2(max=int(100000)))
with tqdm(counter, total=len(counter)) as tqcounter:
    for idx, pow2 in enumerate(tqcounter):
        print()
        print(f'idx = {idx}')
        print(f'powd2 = {pow2}')
        pass

# %%

from tqdm import tqdm

for i in tqdm(range(int(9e6))):
    pass

# %%

from tqdm import tqdm

import time

with tqdm(range(int(5))) as trange:
    for i in trange:
        print(f'\ni = {i}')
        print('done\n')
        time.sleep(1)
        pass

# %%

# zip, it aligns elements in one list to elements in the other

l1 = [0, 1, 2]
l2 = ['a', 'b', 'c']

print(list(zip(l1, l2)))

# %%

from tqdm import tqdm
import time

lst = range(10000000)
total = 2

with tqdm(lst, total=total) as tlst:
    i = 0
    for _, element in enumerate(tlst):
        print(f'\n->i = {i}\n')
        time.sleep(0.2)
        i += 1
        if i >= total:
            break

print('\n--> DONE \a')

# %%

from tqdm import tqdm
import time

lst = range(10000000)
total = 2

with tqdm(lst, total=total) as tlst:
    for idx, element in enumerate(tlst):
        print(f'\n->idx = {idx}\n')
        time.sleep(0.2)
        if idx >= total:
            break

print('\n--> DONE \a')

# %%

from tqdm import tqdm
import time

lst = range(10000000)
total = 2

with tqdm(range(total)) as tcounter:
    lst = iter(lst)
    for idx, element in enumerate(tcounter):
        print(f'\n->idx = {idx}\n')
        time.sleep(0.2)

print('\n--> DONE \a')

# %%

# Question: Do detached() tensors track their own gradients seperately?
# Ans: Yes!
# https://discuss.pytorch.org/t/why-is-the-clone-operation-part-of-the-computation-graph-is-it-even-differentiable/67054/11

import torch

a = torch.tensor([2.0], requires_grad=True)
b = a.detach()
b.requires_grad = True

la = (5.0 - a) ** 2
la.backward()
print(f'a.grad = {a.grad}')

lb = (6.0 - b) ** 2
lb.backward()
print(f'b.grad = {b.grad}')

# %%

import torch
import torch.nn as nn

from collections import OrderedDict

params = OrderedDict([
    ('fc0', nn.Linear(in_features=4, out_features=4)),
    ('ReLU0', nn.ReLU()),
    ('fc1', nn.Linear(in_features=4, out_features=1))
])
mdl = nn.Sequential(params)

print(params)
print(mdl._parameters)
print(params == params)
print(mdl._parameters == params)
print(mdl._modules)

print()
for name, w in mdl.named_parameters():
    print(name, w.norm(2))

print()
# mdl._modules['fc0'] = nn.Linear(10,11)
mdl._modules[0]

for name, w in mdl.named_parameters():
    print(name, w.norm(2))

# %%

## Q: are parameters are in computation graph?
# import torch
# import torch.nn as nn
# # from torchviz import make_dot
#
# from collections import OrderedDict
#
# fc0 = nn.Linear(in_features=3, out_features=1)
# params = [('fc0', fc0)]
# mdl = nn.Sequential(OrderedDict(params))
#
# x = torch.randn(1, 3)
# y = torch.randn(1)
#
# l = (mdl(x) - y) ** 2
#
# # make_dot(l,{x:'x',y:'y','fc0':fc0})
# print(fc0.weight)
# print(fc0.bias)
# print(fc0.weight.to_tens)
# print()
# # make_dot(l,{x:'x',y:'y','fc0':fc0})
# make_dot(l, {'x': x, 'y': y})
# make_dot(l)

# %%

'''
expand
'''

import torch

x = torch.randn([2, 3, 4, 5])

# h_0 of shape (num_layers * num_directions, batch, hidden_size)
h = torch.randn([1, 4, 8])

x_mean = x.mean()
print(x_mean.size())
print(x_mean)
x = x_mean.expand_as(h)
print(x.size())
print(x)

# %%

import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
type(device)
print(device == 'cpu')
device.type

# %%

# THIS WORKS

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

# log_dir (string) – Save directory location.
# Default is runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.

tb = SummaryWriter()
tb.add_scalar('loss', 111)

# %%

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path


def CURRENT_DATETIME_HOSTNAME(comment=''):
    # if not log_dir:
    import socket
    import os
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    return Path(log_dir)


# log_dir (string) – Save directory location.
# Default is runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.
# tensorboard --logdir=runs
log_dir = (Path('~/automl-meta-learning/') / CURRENT_DATETIME_HOSTNAME()).expanduser()
print(log_dir)
tb = SummaryWriter(log_dir=log_dir)
tb.add_scalar('loss', 15)

# %%

# download mini-imagenet automatically

# from torchvision.utils import download_and_extract_archive

import torchvision.utils as utils

print(utils)
# print(download_and_extract_archive)

# %%

# torch concat, https://pytorch.org/docs/stable/torch.html#torch.cat
# Concatenates the given sequence of seq tensors in the given dimension.
# All tensors must either have the same shape (except in the concatenating dimension) or be empty.
import torch

g1 = torch.randn(3, 2)
g2 = torch.randn(4, 2)

g3 = torch.randn(4, 2, 3)

grads = [g1, g2]
print(g1.view(-1).size())
print(g2.view(-1).size())
print(g3.view(-1).size())
# print(g3.view(-1))

grads = torch.cat(grads, dim=0)
print(grads)
print(grads.size())
print(grads.mean())
print(grads.std())

# torch stack, https://pytorch.org/docs/stable/torch.html#torch.stack
# Concatenates sequence of tensors along a new dimension.
# All tensors need to be of the same size.
# torch.stack([g1,g2], dim=0)

# %%

import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
a_detached = a.detach()
print(a_detached.is_leaf)
a_detached_sum = a.sum()
print(c.is_leaf)
d = c.detach()
print(d.is_leaf)

# %%

import torch

from types import SimpleNamespace
from pathlib import Path
from pprint import pprint

x = torch.empty([1, 2, 3])
print(x.size())

args = SimpleNamespace()
args.data_root = "~/automl-meta-learning/data/miniImagenet"

# n1313361300001299.jpg
args.data_root = Path(args.data_root).expanduser()

# %%

import torch

CHW = 3, 12, 12
x = torch.randn(CHW)
y = torch.randn(CHW)

new = [x, y]
new = torch.stack(new)
print(x.size())
print(new.size())

# %%

print('a');
print('b')

# %%

# conver list to tensor

import torch

x = torch.tensor([1, 2, 3.])
print(x)

# %%

from torchvision.transforms import Compose, Resize, ToTensor

import torchmeta
from torchmeta.datasets.helpers import miniimagenet

from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

## get args
args = SimpleNamespace(episodes=5, n_classes=5, k_shot=5, k_eval=15, meta_batch_size=1, n_workers=4)
args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()

## get meta-batch loader
train_transform = Compose([Resize(84), ToTensor()])
dataset = miniimagenet(
    args.data_root,
    ways=args.n_classes,
    shots=args.k_shot,
    test_shots=args.k_eval,
    meta_split='train',
    download=False)
dataloader = torchmeta.utils.data.BatchMetaDataLoader(
    dataset,
    batch_size=args.meta_batch_size,
    num_workers=args.n_workers)

with tqdm(dataset):
    print(f'len(dataloader)= {len(dataloader)}')
    for episode, batch in enumerate(dataloader):
        print(f'episode = {episode}')
        train_inputs, train_labels = batch["train"]
        print(f'train_labels[0] = {train_labels[0]}')
        print(f'train_inputs.size() = {train_inputs.size()}')
        pass
        if episode >= args.episodes:
            break

# %%

# zip tensors

import torch

x = torch.tensor([1., 2., 3.])
y = torch.tensor([1, 2, 3])

print(list(zip(x, y)))

xx = torch.randn(2, 3, 84, 84)
yy = torch.randn(2, 3, 32, 32)

print(len(list(zip(xx, yy))))

# %%

x = 2
print(x)

# %%

## sinusioid function
print('Starting Sinusioid cell')

from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter

# from tqdm import tqdm

batch_size = 16
shots = 5
test_shots = 15
# dataset = torchmeta.toy.helpers.sinusoid(shots=shots, test_shots=tes_shots)
metaset_dataset = Sinusoid(num_samples_per_task=shots + test_shots, num_tasks=100, noise_std=None)
splitter_metset_dataset = ClassSplitter(
    metaset_dataset,
    num_train_per_class=shots,
    num_test_per_class=test_shots,
    shuffle=True)
dataloader = BatchMetaDataLoader(splitter_metset_dataset, batch_size=batch_size, num_workers=4)

print(f'batch_size = {batch_size}')
print(f'len(dataset) = {len(metaset_dataset)}')
print(f'len(dataloader) = {len(dataloader)}\n')
for batch_idx, batch in enumerate(dataloader):
    print(f'batch_idx = {batch_idx}')
    train_inputs, train_targets = batch['train']
    test_inputs, test_targets = batch['test']
    print(f'train_inputs.shape = {train_inputs.shape}')
    print(f'train_targets.shape = {train_targets.shape}')
    print(f'test_inputs.shape = {test_inputs.shape}')
    print(f'test_targets.shape = {test_targets.shape}')
    if batch_idx >= 1:  # halt after 2 iterations
        break

print('DONE\a')

# %%

## notes of torchmeta

from pathlib import Path
import torchmeta

# meta-set: creates collection of data-sets, D_meta = {D_1, ... Dn}
print('\n-- Sinusoid(MetaDataset)')
metaset_sinusoid = torchmeta.toy.Sinusoid(num_samples_per_task=10, num_tasks=1_000_000, noise_std=None)
print(f'type(metaset_sinusoid) = {type(metaset_sinusoid)}')
print(f'len(metaset_sinusoid) = {len(metaset_sinusoid)}')
print(f'metaset_sinusoid = {metaset_sinusoid}')

# this is still a data set but helps implement forming D_i
# i.e. the N-way, K-shot tasks/datasets we need.
print('\n-- MiniImagenet(CombinationMetaDataset)')
data_path = Path('~/data').expanduser()
metaset_miniimagenet = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True,
                                                       download=True)
print(f'type(metaset_miniimagenet) = {type(metaset_miniimagenet)}')
print(f'len(metaset_miniimagenet) = {len(metaset_miniimagenet)}')
print(f'metaset_miniimagenet = {metaset_miniimagenet}')

# Splits the data-sets inside the meta-set into support/train & query/test sets
dataset = metaset_miniimagenet
dataset = torchmeta.transforms.ClassSplitter(dataset, num_train_per_class=1, num_test_per_class=15, shuffle=True)
print(dataset)

# %%

import torch
import torch.nn as nn
import numpy as np

x = np.random.uniform()

x = torch.rand()

print(x)

l = nn.Linear(1, 1)

y = l(x)

print(y)

# %%

# saving tensors for my data set
import torch
import torch.nn as nn

from collections import OrderedDict

from pathlib import Path

# N x's of size D=1 in an interval
Din, Dout = 3, 2
num_samples = 5
lb, ub = -1, 1
X = (ub - lb) * torch.rand([num_samples, Din]) + lb  # rand gives uniform in [0,1) range

# N y's of size D=1 (from output of NN)
f = nn.Sequential(OrderedDict([
    ('f1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))

# fill cnn with Gaussian
mu1, std1 = 5, 7.5
f.f1.weight.data.normal_(mu1, std1)
f.f1.bias.data.normal_(mu1, std1)

# get outputs
Y = f(X)
print(Y)

# save tensors and cnn
# https://stackoverflow.com/questions/1466000/difference-between-modes-a-a-w-w-and-r-in-built-in-open-function
db = {
    'X': X,
    'Y': Y
}
path = Path(f'~/data/tmp/SinData_mu1{mu1}_std1{std1}/').expanduser()
path.mkdir(parents=True, exist_ok=True)
with open(path / 'db', 'w') as file:  # create file and truncate to length 0, only writing allowed
    torch.save(db, file)

# %%

# saving data in numpy

import numpy as np
import pickle
from pathlib import Path

path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

lb, ub = -1, 1
num_samples = 5
x = np.random.uniform(low=lb, high=ub, size=(1, num_samples))
y = x ** 2 + x + 2

# using save (to npy), savez (to npz)
np.save(path / 'x', x)
np.save(path / 'y', y)
np.savez(path / 'db', x=x, y=y)
with open(path / 'db.pkl', 'wb') as db_file:
    pickle.dump(obj={'x': x, 'y': y}, file=db_file)

## using loading npy, npz files
x_loaded = np.load(path / 'x.npy')
y_load = np.load(path / 'y.npy')
db = np.load(path / 'db.npz')
with open(path / 'db.pkl', 'rb') as db_file:
    db_pkl = pickle.load(db_file)

print(x is x_loaded)
print(x == x_loaded)
print(x == db['x'])
print(x == db_pkl['x'])
print('done')

# %%

import numpy as np
from pathlib import Path

path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

lb, ub = -1, 1
num_samples = 5
x = np.random.uniform(low=lb, high=ub, size=(1, num_samples))
y = x ** 2 + x + 2

np.save(path / 'x', x)
np.save(path / 'y', y)

x_loaded = np.load(path / 'x.npy')
y_load = np.load(path / 'y.npy')

print(x is x_loaded)  # False
print(x == x_loaded)  # [[ True  True  True  True  True]]

# %%

# saving torch tensors

import torch
import torch.nn as nn
import torchvision

from pathlib import Path
from collections import OrderedDict

path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

tensor_a = torch.rand(2, 3)
tensor_b = torch.rand(1, 3)

db = {'a': tensor_a, 'b': tensor_b}

torch.save(db, path / 'torch_db')
loaded = torch.load(path / 'torch_db')
print(loaded['a'] == tensor_a)
print(loaded['b'] == tensor_b)

# testing if ToTensor() screws things up
lb, ub = -1, 1
N, Din, Dout = 3, 1, 1
x = torch.distributions.Uniform(low=lb, high=ub).sample((N, Din))
print(x)

f = nn.Sequential(OrderedDict([
    ('f1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
y = f(x)

transform = torchvision.transforms.transforms.ToTensor()
y_proc = transform(y)
print(y_proc)

# %%

# merge dict
# union dictionaries, https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python

d1 = {'a': 1, 'b': 2.5}
d2 = {'b': 2, 'c': 3, 'd': 4}
d = {**d1, **d2}
# duplicates resolved in favour of d2
print(d)

# %%

# generating uniform variables

import numpy as np

num_samples = 3
Din = 1
lb, ub = -1, 1

xn = np.random.uniform(low=lb, high=ub, size=(num_samples, Din))
print(xn)

import torch

sampler = torch.distributions.Uniform(low=lb, high=ub)
r = sampler.sample((num_samples, Din))

print(r)

r2 = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))

print(r2)

# process input
f = nn.Sequential(OrderedDict([
    ('f1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
Y = f(r2)
print(Y)

# %%

# sampling from normal distribution in torch

import torch

num_samples = 3
Din = 1
mu, std = 0, 1
x = torch.distributions.normal.Normal(loc=mu, scale=std).sample((num_samples, Din))

print(x)

# %%

# creating data and running through a nn and saving it

import torch
import torch.nn as nn

from pathlib import Path
from collections import OrderedDict

import numpy as np

import pickle

path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

num_samples = 3
Din, Dout = 1, 1
lb, ub = -1, 1

x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))

f = nn.Sequential(OrderedDict([
    ('f1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
y = f(x)

# save data torch to numpy
x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
np.savez(path / 'db', x=x_np, y=y_np)

print(x_np)
# save model
with open('db_saving_seq', 'wb') as file:
    pickle.dump({'f': f}, file)

# load model
with open('db_saving_seq', 'rb') as file:
    db = pickle.load(file)
    f2 = db['f']

# test that it outputs the right thing
y2 = f2(x)

y_eq_y2 = y == y2
print(y_eq_y2)

db2 = {'f': f, 'x': x, 'y': y}
torch.save(db2, path / 'db_f_x_y')

print('Done')

db3 = torch.load(path / 'db_f_x_y')
f3 = db3['f']
x3 = db3['x']
y3 = db3['y']
yy3 = f3(x3)

y_eq_y3 = y == y3
print(y_eq_y3)

y_eq_yy3 = y == yy3
print(y_eq_yy3)

# %%

# test for saving everything with torch.save

import torch
import torch.nn as nn

from pathlib import Path
from collections import OrderedDict

import numpy as np

import pickle

path = Path('~/data/tmp/').expanduser()
path.mkdir(parents=True, exist_ok=True)

num_samples = 3
Din, Dout = 1, 1
lb, ub = -1, 1

x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))

f = nn.Sequential(OrderedDict([
    ('f1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
y = f(x)

# save data torch to numpy
x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
db2 = {'f': f, 'x': x_np, 'y': y_np}
torch.save(db2, path / 'db_f_x_y')
# np.savetxt(path / 'output.csv', y_np)  # for csv

db3 = torch.load(path / 'db_f_x_y')
f3 = db3['f']
x3 = db3['x']
y3 = db3['y']
xx = torch.tensor(x3)
yy3 = f3(xx)

print(yy3)

# %%

# my saving code for synthetic data, nvm using torch.save for everything

# import torch
# import torch.nn as nn
#
# from pathlib import Path
# from collections import OrderedDict
#
# import numpy as np
#
# path = Path('~/data/tmp/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
#
# num_samples = 3
# Din, Dout = 1, 1
# lb, ub = -1, 1
#
# x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))
#
# f = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din,Dout)),
#     ('out', nn.SELU())
# ]))
# y = f(x)
#
# # save data torch to numpy
# x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
# np.savez(path / 'data', x=x_np, y=y_np)
#
# # save model
# torch.save(f,path / 'f')

# %%

import torch

import torch.nn as nn

from collections import OrderedDict

num_samples = 3
Din, Dout = 1, 1
lb, ub = -1, 1

x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))

hidden_dim = [(Din, 20), (20, 20), (20, 20), (20, 20), (20, Dout)]
f = nn.Sequential(OrderedDict([
    ('fc1;l1', nn.Linear(hidden_dim[0][0], hidden_dim[0][1])),
    ('relu2', nn.ReLU()),
    ('fc2;l1', nn.Linear(hidden_dim[1][0], hidden_dim[1][1])),
    ('relu2', nn.ReLU()),
    ('fc3;l1', nn.Linear(hidden_dim[2][0], hidden_dim[2][1])),
    ('relu3', nn.ReLU()),
    ('fc4;l1', nn.Linear(hidden_dim[3][0], hidden_dim[3][1])),
    ('relu4', nn.ReLU()),
    ('fc5;final;l2', nn.Linear(hidden_dim[4][0], hidden_dim[4][1]))
]))

y = f(x)

print(y)

section_label = [1] * 4 + [2]
print(section_label)

# %%

# get list of paths to task
# https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
# https://stackoverflow.com/a/44228436/1601580

from pathlib import Path
from glob import glob

meta_split = 'train'
data_path = Path('~/data/LS/debug/fully_connected_NN_mu1_1.0_std1_2.5_mu2_1.0_std2_0.5/')
data_path = (data_path / meta_split).expanduser()

# with path lib
tasks_folder = [f for f in data_path.iterdir() if f.is_dir()]

assert ('f_avg' not in tasks_folder)

len_folder = len(tasks_folder)
print(len_folder)
print(tasks_folder)
print()

# with glob
p = str(data_path) + '/*/'
print(p)
tasks_folder = glob(p)

assert ('f_avg' not in tasks_folder)

len_folder = len(tasks_folder)
print(len_folder)
print(tasks_folder)
print()

# with glob and negation
print(set(glob(str(data_path / "f_avg"))))
tasks_folder = set(glob(str(data_path / '*'))) - set(glob(str(data_path / "f_avg")))

assert ('f_avg' not in tasks_folder)

len_folder = len(tasks_folder)
print(len_folder)
print(tasks_folder)
print()

# %%

# looping through metasets

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter
from torchmeta.toy import Sinusoid

from tqdm import tqdm

# get data set
dataset = Sinusoid(num_samples_per_task=25, num_tasks=30)
shots, test_shots = 5, 15
# get metaset
metaset = ClassSplitter(
    dataset,
    num_train_per_class=shots,
    num_test_per_class=test_shots,
    shuffle=True)
# get meta-dataloader
batch_size = 16
num_workers = 0
meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
epochs = 2

print(f'batch_size = {batch_size}')
print(f'len(metaset) = {len(metaset)}')
print(f'len(meta_dataloader) = {len(meta_dataloader)}')
with tqdm(range(epochs)) as tepochs:
    for epoch in tepochs:
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'\nbatch_idx = {batch_idx}')
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            print(f'train_inputs.shape = {train_inputs.shape}')
            print(f'train_targets.shape = {train_targets.shape}')
            print(f'test_inputs.shape = {test_inputs.shape}')
            print(f'test_targets.shape = {test_targets.shape}')

# %%

from tqdm import tqdm

import time

with tqdm(range(5)) as trange:
    for t in trange:
        print(t)
        time.sleep(1)

# %%


import torch
import torch.nn as nn

l1 = torch.tensor([1, 2, 3.]) ** 0.5
l2 = torch.tensor([0, 0, 0.0])
mse = nn.MSELoss()
loss = mse(l1, l2)
print(loss)

# %%

import numpy as np

x = np.arange(0, 10)
print(x)

print(x.max())
print(x.min())
print(x.mean())
print(np.median(x))

# %%

x = torch.randn(3)
print(x)
print(x.argmax(-1))

# %%

# testing accuracy function
# https://discuss.pytorch.org/t/calculating-accuracy-of-the-current-minibatch/4308/11
# https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch

import torch
import torch.nn as nn

D = 1
true = torch.tensor([0, 1, 0, 1, 1]).reshape(5, 1)
print(f'true.size() = {true.size()}')

batch_size = true.size(0)
print(f'batch_size = {batch_size}')
x = torch.randn(batch_size, D)
print(f'x = {x}')
print(f'x.size() = {x.size()}')

mdl = nn.Linear(D, 1)
logit = mdl(x)
_, pred = torch.max(logit.data, 1)

print(f'logit = {logit}')

print(f'pred = {pred}')
print(f'true = {true}')

acc = (true == pred).sum().item()
print(f'acc = {acc}')

# %%

# https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be
# dimension
# https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/4?u=brando_miranda
"""
Dimension reduction. It collapses/reduces a specific dimension by selecting an element from that dimension to be
reduced.
Consider x is 3D tensor. x.sum(1) converts x into a tensor that is 2D using an element from D1 elements in
the 1th dimension. Thus:
x.sum(1) = x[i,k] = op(x[i,:,k]) = op(x[i,0,k],...,x[i,D1,k])
the key is to realize that we need 3 indices to select a single element. So if we use only 2 (because we are collapsing)
then we have D1 number of elements possible left that those two indices might indicate. So from only 2 indices we get a
set that we need to specify how to select. This is where the op we are using is used for and selects from this set.
In theory if we want to collapse many indices we need to indicate how we are going to allow indexing from a smaller set
of indices (using the remaining set that we'd usually need).
"""

import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(f'x.size() = {x.size()}')

# sum the 0th dimension (rows). So we get a bunch of colums that have the rows added together.
x0 = x.sum(0)
print(x0)

# sum the 1th dimension (columns)
x1 = x.sum(1)
print(x1)

x_1 = x.sum(-1)
print(x_1)

x0 = x.max(0)
print(x0.values)

y = torch.tensor([[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]],

    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]])

print(y)

# into the screen [1, 13]
print(y[:, 0, 0])
# columns [1, 5, 9]
print(y[0, :, 0])
# rows [1, 2, 3, 4]
print(y[0, 0, :])

# for each remaining index, select the largest value in the "screen" dimension
y0 = y.max(0)
print(y0.values)


# %%

# understanding making label predictions
# https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/3?u=brando_miranda

def calc_accuracy(mdl, X, Y):
    # reduce/collapse the classification dimension according to max op
    # resulting in most likely label
    max_vals, max_indices = mdl(X).max(1)
    # assumes the first dimension is batch size
    n = max_indices.size(0)  # index 0 for extracting the # of elements
    # calulate acc (note .item() to do float division)
    acc = (max_indices == Y).sum().item() / n
    return acc


import torch
import torch.nn as nn

# data dimension [batch-size, D]
D, Dout = 1, 5
batch_size = 16
x = torch.randn(batch_size, D)
y = torch.randint(low=0, high=Dout, size=(batch_size,))

mdl = nn.Linear(D, Dout)
logits = mdl(x)
print(f'y.size() = {y.size()}')
# removes the 1th dimension with a max, which is the classification layer
# which means it returns the most likely label. Also, note you need to choose .indices since you want to return the
# position of where the most likely label is (not it's raw logit value)
pred = logits.max(1).indices
print(pred)

print('--- preds vs truth ---')
print(f'predictions = {pred}')
print(f'y = {y}')

acc = (pred == y).sum().item() / pred.size(0)
print(acc)
print(calc_accuracy(mdl, x, y))

# %%

# https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/20

import torch
import torch.nn as nn

x = torch.randn(1)
mdl = nn.Linear(1, 1)

y = mdl(x)
print(mdl.weight)

print(y)

# %%

# https://discuss.pytorch.org/t/how-to-get-the-module-names-of-nn-sequential/39682
# looping through modules but get the one with a specific name

import torch
import torch.nn as nn

from collections import OrderedDict

params = OrderedDict([
    ('fc0', nn.Linear(in_features=4, out_features=4)),
    ('ReLU0', nn.ReLU()),
    ('fc1L:final', nn.Linear(in_features=4, out_features=1))
])
mdl = nn.Sequential(params)

# throws error
# mdl['fc0']

for m in mdl.children():
    print(m)

print()

for m in mdl.modules():
    print(m)

print()

for name, m in mdl.named_modules():
    print(name)
    print(m)

print()

for name, m in mdl.named_children():
    print(name)
    print(m)

# %%

# apply mdl to x until the final layer, then return the embeding

# import torch
# import torch.nn as nn
# 
# from collections import OrderedDict
# 
# Din, Dout = 1, 1
# H = 10
# 
# modules = OrderedDict([
#     ('fc0', nn.Linear(in_features=Din, out_features=H)),
#     ('ReLU0', nn.ReLU()),
# 
#     ('fc1', nn.Linear(in_features=H, out_features=H)),
#     ('ReLU1', nn.ReLU()),
# 
#     ('fc2', nn.Linear(in_features=H, out_features=H)),
#     ('ReLU2', nn.ReLU()),
# 
#     ('fc3', nn.Linear(in_features=H, out_features=H)),
#     ('ReLU3', nn.ReLU()),
# 
#     ('fc4L:final', nn.Linear(in_features=H, out_features=Dout))
# ])
# 
# mdl = nn.Sequential(modules)
# 
# out = x
# for name, m in self.base_model.named_children():
#     if 'final' in name:
#         # return out
#         break
#     out = m(out)
# 
# print(out.size())

# %%

# initializing a constant weight net
# https://discuss.pytorch.org/t/how-to-add-appropriate-noise-to-a-neural-network-with-constant-weights-so-that-back-propagation-training-works/93411

# import torch

# [layer.reset_parameters() for layer in base_model.children() if hasattr(layer, 'reset_parameters')]

# model = nn.Linear(1, 1)
# model_copy = copy.deepcopy(model)

# %%

print('start')

# f_avg: PLinReg vs MAML

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

datas_std = [0.1, 0.125, 0.1875, 0.2]

pl = [2.3078539778125768e-07,
      1.9997889411762922e-07,
      2.729681222011256e-07,
      3.2532371115080884e-07]
pl_stds = [1.4852212316567463e-08,
           5.090588920661132e-09,
           1.1424832554909115e-08,
           5.058656213138166e-08]

maml = [3.309504692539563e-07,
        4.1058904888091606e-06,
        6.8326703386053605e-06,
        7.4616147721799645e-06]
maml_stds = [4.039131189060566e-08,
             3.66839089258494e-08,
             9.20683484136399e-08,
             9.789292209743077e-08]

# fig = plt.figure()
fig, ax = plt.subplots(nrows=1, ncols=1)

ax.set_title('MAML vs Pre-Trained embedding with Linear Regression')

x = datas_std

ax.errorbar(x, pl, yerr=pl_stds, label='PLinReg', marker='o')
ax.errorbar(x, maml, yerr=maml_stds, label='MAML', marker='o')
ax.plot()
ax.legend()

ax.set_xlabel('std (of FNN Data set)')
ax.set_ylabel('meta-test loss (MSE)')

plt.show()

# path = Path('~/ultimate-utils/plot').expanduser()
# fig.savefig(path)

print('done \a')
# %%

# Torch-meta miniImagenet
# loop through meta-batches of this data set, print the size, make sure it's the size you exepct

import torchmeta
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter
# from torchmeta.toy import Sinusoid

from tqdm import tqdm

# dataset = Sinusoid(num_samples_per_task=100, num_tasks=20)
dataset = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True, download=True)
print(f'type(metaset_miniimagenet) = {type(dataset)}')
print(f'len(metaset_miniimagenet) = {len(dataset)}')
shots, test_shots = 5, 15
# get metaset
metaset = ClassSplitter(
    dataset,
    num_train_per_class=shots,
    num_test_per_class=test_shots,
    shuffle=True)
# get meta-dataloader
batch_size = 16
num_workers = 0
meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
epochs = 2

print(f'batch_size = {batch_size}')
print(f'len(metaset) = {len(metaset)}')
print(f'len(meta_dataloader) = {len(meta_dataloader)}\n')
with tqdm(range(epochs)) as tepochs:
    for epoch in tepochs:
        print(f'\n[epoch={epoch}]')
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'batch_idx = {batch_idx}')
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            print(f'train_inputs.shape = {train_inputs.shape}')
            print(f'train_targets.shape = {train_targets.shape}')
            print(f'test_inputs.shape = {test_inputs.shape}')
            print(f'test_targets.shape = {test_targets.shape}')
            print()

# %%

import torch

x = torch.tensor([1., 2, 3])
print(x.mean())

print(x * x)
print(x @ x)
print(x.matmul(x))

# x.mm(x) weird error

# %%

import torch

x = torch.randn(12, 20)
y = torch.randn(20, 30)

out = x @ y
print(out.size())

# %%
# https://www.youtube.com/watch?v=46RjXawJQgg&t=1493s

from pathlib import Path

from pandas import read_csv

read_csv(Path())

# %%

print('hello-world')
xx = 2

print(xx)

print(' ')

##
print('end!')

# %%

# let's see how big the random values from the normal are

import torch

D = 8
w = torch.tensor([0.1] * D)
print(f'w.size() = {w.size()}')
mu = torch.zeros(w.size())
std = w * 1.5e-2  # two decimal places and a little more
noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()

print('--- noise ')
print(noise.size())
print(noise)

w += noise
print('--- w')
print(w.size())
print(w)

# %%

# editing parameters in pytorch in place without error: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/41

import torch
import torch.nn as nn
from collections import OrderedDict

Din, Dout = 8, 1

base_model = nn.Sequential(OrderedDict([
    ('f1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))

with torch.no_grad():
    for i, w in enumerate(base_model.parameters()):
        print(f'--- i = {i}')
        print(w)
        w += w + 0.001
        print(w)

# %%

# pickle vs torch.save

# def log_validation(args, meta_learner, outer_opt, meta_val_set):
#     """ Log the validation loss, acc. Checkpoint the model if that flag is on. """
#     if args.save_ckpt:  # pickle vs torch.save https://discuss.pytorch.org/t/advantages-disadvantages-of-using-pickle-module-to-save-models-vs-torch-save/79016
#         # make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
#         path_to_ckpt = args.logger.current_logs_path
#         path_to_ckpt.mkdir(parents=True, exist_ok=True)  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
#         ckpt_path_plus_path = path_to_ckpt / Path('db')
#
#         args.base_model = "check the meta_learner field in the checkpoint not in the args field"  # so that we don't save the child model so many times since it's part of the meta-learner
#         # note this obj has the last episode/outer_i we ran
#         torch.save({'args': args, 'meta_learner': meta_learner}, ckpt_path_plus_path)
#     acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, meta_learner, meta_val_set)
#     if acc_mean > args.best_acc:
#         args.best_acc, args.loss_of_best = acc_mean, loss_mean
#         args.logger.loginfo(
#             f"***> Stats of Best Acc model: meta-val loss: {args.loss_of_best} +- {loss_std}, meta-val acc: {args.best_acc} +- {acc_std}")
#     return acc_mean, acc_std, loss_mean, loss_std

# %%

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression()
print(reg)
reg = LinearRegression().fit(X, y)
print(reg)
reg.score(X, y)

reg.coef_

reg.intercept_

reg.predict(np.array([[3, 5]]))

# %%

# https://stackoverflow.com/questions/63818676/what-is-the-machine-precision-in-pytorch-and-when-should-one-use-doubles
# https://discuss.pytorch.org/t/how-does-one-start-using-double-without-unexpected-bugs/95715
# https://discuss.pytorch.org/t/what-is-the-machine-precision-of-pytorch-with-cpus-or-gpus/9384

import torch

x1 = torch.tensor(1e-6)
x2 = torch.tensor(1e-7)
x3 = torch.tensor(1e-8)
x4 = torch.tensor(1e-9)

eps = torch.tensor(1e-11)

print(x1.dtype)
print(x1)
print(x1 + eps)

print(x2)
print(x2 + eps)

print(x3)
print(x3 + eps)

print(x4)
print(x4 + eps)

# %%

# python float is a C double
# NumPy's standard numpy.float is the same (so C double), also numpy.float64.
# https://www.doc.ic.ac.uk/~eedwards/compsys/float/
# https://stackoverflow.com/questions/1049722/what-is-2s-complement
# https://www.cs.cornell.edu/~tomf/notes/cps104/twoscomp.html#whyworks
# https://stackoverflow.com/questions/7524838/fixed-point-vs-floating-point-number
# https://en.wikipedia.org/wiki/Single-precision_floating-point_format
# https://www.cs.cornell.edu/~tomf/notes/cps104/twoscomp.html#whyworks

import torch

xf = torch.tensor(1e-7)
xd = torch.tensor(1e-7, dtype=torch.double)
epsf = torch.tensor(1e-11)

print(xf.dtype)
print(xf)
print(xf.item())
print(type(xf.item()))

#
print('\n> test when a+eps = a')
print(xf.dtype)
print(f'xf = {xf}')
print(f'xf + 1e-7 = {xf + 1e-7}')
print(f'xf + 1e-11 = {xf + 1e-11}')
print(f'xf + 1e-8 = {xf + 1e-8}')
print(f'xf + 1e-16 = {xf + 1e-16}')
# after seeing the above it seems that there are errors if things are small

print('\n> test when a+eps = a')
x = torch.tensor(1e-7, dtype=torch.double)
print(f'xf = {x}')
print(f'xf + 1e-7 = {x + 1e-7}')
print(f'xf + 1e-11 = {x + 1e-11}')
print(f'xf + 1e-8 = {x + 1e-8}')
print(f'xf + 1e-16 = {x + 1e-16}')
# using doubles clearly is better but still has some errors

print('\n> test when a+eps = a')
x = torch.tensor(1e-4)
print(f'xf = {x}')
print(f'xf + 1e-7 = {x + 1e-7}')
print(f'xf + 1e-11 = {x + 1e-11}')
print(f'xf + 1e-8 = {x + 1e-8}')
print(f'xf + 1e-16 = {x + 1e-16}')

# %%

# https://pytorch.org/docs/stable/torchvision/models.html

# %%

import torch

print(torch.zeros(2))
m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
x = m.sample()
print(x)

# m = torch.distributions.MultivariateNormal(torch.zeros(1, 3), torch.eye(1, 3))
# mu = m.sample()
# print(mu)

m = torch.distributions.MultivariateNormal(torch.zeros(1, 5), torch.eye(5))
y = m.sample()
print(y)

# %%

from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np

path = Path('~/data/').expanduser()

# x = np.linspace(0, 2*np.pi, 50)
x = np.random.uniform(0, 2 * np.pi, 100)
noise = np.random.normal(0.0, 0.05, 100)
print(noise)
y = np.sin(x) + noise
plt.figure()
plt.scatter(x, y)
plt.ylabel('f(x)')
plt.ylabel('x (raw feature)')
plt.savefig(path / 'test_fig.pdf')
plt.savefig(path / 'test_fig.png')
plt.show()

# %%

from socket import gethostname
from email.message import EmailMessage
import smtplib

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
print(server)

# %%

# MTA (Mail Transfer Agent)
# https://stackoverflow.com/questions/784201/is-there-a-python-mta-mail-transfer-agent
# https://www.quora.com/How-does-one-send-e-mails-from-Python-using-MTA-Mail-Transfer-Agent-rather-than-an-SMTP-library
# https://www.reddit.com/r/learnpython/comments/ixlq81/how_does_one_send_emails_from_python_using_mta/

# Q why can't I just send an email directly?
# Q why do smtp libraries exist

# %%

import smtplib

server = smtplib.SMTP('smtp.intel-research.net', 25)
server.starttls()
print(server)


# %%

# from socket import gethostname
# from email.message import EmailMessage
# import smtplib
#
# server = smtplib.SMTP('smtp.gmail.com', 587)
# server.starttls()
# # not a real email account nor password, its all ok!
# server.login('slurm.miranda@gmail.com', 'dummy123!@#$321')
#
# # craft message
# msg = EmailMessage()
#
# message = f'{message}\nSend from Hostname: {gethostname()}'
# msg.set_content(message)
# msg['Subject'] = subject
# msg['From'] = 'slurm.miranda@gmail.com'
# msg['To'] = destination
# # send msg
# server.send_message(msg)

# %%

# send email with smtp intel

def send_email(message):
    from socket import gethostname
    import smtplib
    hostname = gethostname()
    from_address = 'slurm.miranda@gmail.com'
    from_address = 'miranda9@intel-research.net.'
    # to_address = [ 'iam-alert@intel-research.net']
    to_address = ['brando.science@gmail.com']
    subject = f"Test msg from: {hostname}"
    ##
    message = f'Test msg from {hostname}: {message}'
    full_message = f'From: {from_address}\n' \
                   f'To: {to_address}\n' \
                   f'Subject: {subject}\n' \
                   f'{message}'
    server = smtplib.SMTP('smtp.intel-research.net')
    server.sendmail(from_address, to_address, full_message)
    server.quit()
    # sys.exit(1)


print('start')
send_email('HelloWorld')
print('done email test!')


# %%

def send_email2(message):
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from socket import gethostname
    import smtplib
    server = smtplib.SMTP('smtp.intel-research.net')
    # craft message
    msg = MIMEMultipart()

    message = f'{message}\nSend from Hostname: {gethostname()}'
    msg['Subject'] = 'Test email'
    msg['From'] = 'miranda9@intel-research.net.'
    msg['To'] = 'brando.science@gmail.com'
    msg.attach(MIMEText(message, "plain"))
    # send message
    server.send_message(msg)
    # server.sendmail(from_address, to_address, full_message)
    server.quit()


print('start')
send_email2('HelloWorld')
print('done email test!')

# %%

from pathlib import Path

message = 'HelloWorld'
path_to_pdf = Path('~/data/test_fig.pdf').expanduser()

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from socket import gethostname
import smtplib

server = smtplib.SMTP('smtp.intel-research.net')
# craft message
msg = MIMEMultipart()

message = f'{message}\nSend from Hostname: {gethostname()}'
msg['Subject'] = 'Test email'
msg['From'] = 'miranda9@intel-research.net.'
msg['To'] = 'brando.science@gmail.com'
msg.attach(MIMEText(message, "plain"))
# attach pdf
if path_to_pdf.exists():
    with open(path_to_pdf, "rb") as f:
        # attach = email.mime.application.MIMEApplication(f.read(),_subtype="pdf")
        attach = MIMEApplication(f.read(), _subtype="pdf")
    attach.add_header('Content-Disposition', 'attachment', filename=str(path_to_pdf))
    msg.attach(attach)

# send message
server.send_message(msg)
# server.sendmail(from_address, to_address, full_message)
server.quit()

# %%

# Here, we used "w" letter in our argument, which indicates write and will create a file if it does not exist in library
# Plus sign indicates both read and write.

# with open('data.json', 'w+') as f:
#     json.dump(self.stats, f)

# %%

import numpy as np
from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm

path = Path('~/data/logs/').expanduser()
tb = SummaryWriter(log_dir=path)
# tb = SummaryWriter(log_dir=args.current_logs_path)

for i in range(3):
    loss = i + np.random.normal(loc=0, scale=1)
    tb.add_scalar('loss', loss, i)

# %%

# https://pytorch.org/tutorials/beginner/saving_loading_models.html

# Saving & Loading Model for Inference
# Save/Load state_dict (Recommended)
# Save:
# torch.save(model.state_dict(), PATH)
#
# # Load:
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# %%

# Save:
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#               ...
#             }, PATH)
# # Load:
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
#
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval()
# # - or -
# model.train()

# %%

# https://discuss.pytorch.org/t/how-does-load-a-sequential-model-from-a-string/97648
# https://stackoverflow.com/questions/64109883/how-does-one-load-a-sequential-model-from-a-string-in-pytorch

# %%
#
# torch.save({'f': f,
#             'f_state_dict': f.state_dict(),
#             'f_str': str(f),
#             'f_modules': f._modules,
#             'f_modules_str': str(f._modules)
#             }, path2avg_f)

# %%

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np

path = Path('~/data/tb_test/').expanduser()
# path = Path('~/logs/logs_Sep29_12-38-08_jobid_-1/tb').expanduser()
writer = SummaryWriter(path)

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

print('done! \a')

# %%

db = torch.load(str(args.resume_ckpt_path))
# args.epchs = db['epoch']  # we can start counting from zero
# args.epoch += 1  # this is needed so that it starts on the next batch since it says the last batch it *did* and range counts with 0 indexing.
# meta_learner = db['meta_learner']
args.base_model = db['f']
# in case loading directly doesn't work
modules = eval(db['f_modules_str'])
args.base_model = torch.nn.Sequential(modules)
f_state_dict = db['f_state_dict']
args.base_model.load_state_dict(f_state_dict)

# %%

# Torch-meta miniImagenet

import torchmeta
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter

from pathlib import Path

from tqdm import tqdm

data_path = Path('~/data/').expanduser()
meta_split = 'train'
dataset = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_split=meta_split, download=True)
# dataset = torchmeta.datasets.Omniglot(data_path, num_classes_per_task=5, meta_split=meta_split, download=True)

print(f'type(metaset_miniimagenet) = {type(dataset)}')
print(f'len(metaset_miniimagenet) = {len(dataset)}')
shots, test_shots = 5, 15
metaset = ClassSplitter(
    dataset,
    num_train_per_class=shots,
    num_test_per_class=test_shots,
    shuffle=True)
batch_size = 16
num_workers = 0
meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
epochs = 2

print(f'batch_size = {batch_size}')
print(f'len(metaset) = {len(metaset)}')
print(f'len(meta_dataloader) = {len(meta_dataloader)}\n')
with tqdm(range(epochs)) as tepochs:
    for epoch in tepochs:
        print(f'\n[epoch={epoch}]')
        for batch_idx, batch in enumerate(meta_dataloader):
            print(f'batch_idx = {batch_idx}')
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            print(f'train_inputs.shape = {train_inputs.shape}')
            print(f'train_targets.shape = {train_targets.shape}')
            print(f'test_inputs.shape = {test_inputs.shape}')
            print(f'test_targets.shape = {test_targets.shape}')
            print()
            break
        break

# %%

from torchmeta.datasets.helpers import omniglot
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from pathlib import Path

meta_split = 'train'
data_path = Path('~/data/').expanduser()
dataset = omniglot(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

for batch in dataloader:
    train_inputs, train_targets = batch["train"]
    print('Train inputs shape: {0}'.format(train_inputs.shape))  # (16, 25, 1, 28, 28)
    print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))  # (16, 75, 1, 28, 28)
    print('Test targets shape: {0}'.format(test_targets.shape))  # (16, 75)

# %%

# replacing a module in in a pytorch model
# https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/11

import torch

from torchmeta.datasets.helpers import omniglot
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from pathlib import Path

import copy

meta_split = 'train'
data_path = Path('~/data/').expanduser()
dataset = omniglot(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)


def replace_bn(module, name):
    """
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            new_bn = torch.nn.BatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum,
                                          target_attr.affine,
                                          track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)


def convert_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.__init__(module.num_features, module.eps,
                            module.momentum, module.affine,
                            track_running_stats=False)


fc_out_features = 5

# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
# replace_bn(model, 'model')
# model.fc = torch.nn.Linear(in_features=512, out_features=fc_out_features, bias=True)
#
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
# replace_bn(model, 'model')
# model.fc = torch.nn.Linear(in_features=2048, out_features=fc_out_features, bias=True)

# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
# replace_bn(model, 'model')
# model.fc = torch.nn.Linear(in_features=2048, out_features=fc_out_features, bias=True)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
replace_bn(model, 'model')
model.fc = torch.nn.Linear(in_features=2048, out_features=fc_out_features, bias=True)

for batch in dataloader:
    train_inputs, train_targets = batch["train"]
    print('Train inputs shape: {0}'.format(train_inputs.shape))  # (16, 25, 1, 28, 28)
    print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))  # (16, 75, 1, 28, 28)
    print('Test targets shape: {0}'.format(test_targets.shape))  # (16, 75)
    first_meta_batch = train_inputs[0]  # task
    nk_task = first_meta_batch
    out = model(nk_task)
    print(f'resnet out.size(): {out.size()}')
    break

print('success\a')

# %%

import torch

import torchvision.transforms as transforms

# import torchmeta
# from torchmeta.datasets.helpers import omniglot
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from pathlib import Path

meta_split = 'train'
data_path = Path('~/data/').expanduser()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(84),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
    transforms.ToTensor(),
    normalize])
dataset = miniimagenet(data_path,
                       transform=data_augmentation_transforms,
                       ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)

print(len(dataloader))
# for batch_idx, batch in enumerate(dataloader):
#     print(f'--> batch_idx = {batch_idx}')
#     train_inputs, train_targets = batch["train"]
#     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
#     first_meta_batch = train_inputs[0]  # task
#     nk_task = first_meta_batch
#     out = model(nk_task)
#     print(f'resnet out.size(): {out.size()}')
#     break

print('success\a')

# %%

import torch

import torchvision.transforms as transforms

# import torchmeta
# from torchmeta.datasets.helpers import omniglot
from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from pathlib import Path

meta_split = 'train'
data_path = Path('~/data/').expanduser()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(84),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
    transforms.ToTensor(),
    normalize])
dataset = miniimagenet(data_path,
                       transform=data_augmentation_transforms,
                       ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
print(f'len augmented = {len(dataloader)}')

dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
print(f'len normal = {len(dataloader)}')

print('success\a')

# %%

import torch

import torchvision.transforms as transforms

from torchmeta.datasets.helpers import miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from tqdm import tqdm

from pathlib import Path

meta_split = 'train'
data_path = Path('~/data/').expanduser()

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# data_augmentation_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(84),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.2),
#     transforms.ToTensor(),
#     normalize])
# dataset = miniimagenet(data_path,
#                        transform=data_augmentation_transforms,
#                        ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
# print(f'len augmented = {len(dataloader)}')

dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
print(f'len normal = {len(dataloader)}')

num_batches = 10
with tqdm(dataloader, total=num_batches) as pbar:
    for batch_idx, batch in enumerate(pbar):
        train_inputs, train_targets = batch["train"]
        print(train_inputs.size())
        # print(batch_idx)
        if batch_idx >= num_batches:
            break

print('success\a')

# %%

from math import comb

total_classes = 64
n = 5
number_tasks = comb(total_classes, n)
print(number_tasks)

# %%

# saving a json file save json file
# human readable pretty print https://stackoverflow.com/questions/12943819/how-to-prettyprint-a-json-file

import json

data = 'data string'
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)

# json.dump(data, open('data.txt', 'w'))

# with open(current_logs_path / 'experiment_stats.json', 'w+') as f:
#     json.dump(self.stats, f)
# data_ars = {key:value for (key,value) in dictonary.items()}
# x = {key:str(value) fo# %%
#
# # to test impots
# import sys
#
# for path in sys.path:
#     print(path)
# # %%
#
# import time
#
# import logging
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
#
# file_handler = logging.FileHandler('employee.log')
# file_handler.setFormatter(formatter)
#
# logger.addHandler(file_handler)
#
#
# class Employee:
#     """A sample Employee class"""
#
#     def __init__(self, first, last):
#         self.first = first
#         self.last = last
#
#         logger.info('Created Employee: {} - {}'.format(self.fullname, self.email))
#
#     @property
#     def email(self):
#         return '{}.{}@email.com'.format(self.first, self.last)
#
#     @property
#     def fullname(self):
#         return '{} {}'.format(self.first, self.last)
#
#
# emp_1 = Employee('John', 'Smith')
# emp_2 = Employee('Corey', 'Schafer')
# emp_3 = Employee('Jane', 'Doe')
#
#
# ######## END OF EMPLOYEE LOGGING EXAMPLE
#
# def report_times(start, verbose=False):
#     '''
#     How much time has passed since the time "start"
#
#     :param float start: the number representing start (usually time.time())
#     '''
#     meta_str = ''
#     ## REPORT TIMES
#     start_time = start
#     seconds = (time.time() - start_time)
#     minutes = seconds / 60
#     hours = minutes / 60
#     if verbose:
#         print(f"--- {seconds} {'seconds ' + meta_str} ---")
#         print(f"--- {minutes} {'minutes ' + meta_str} ---")
#         print(f"--- {hours} {'hours ' + meta_str} ---")
#         print('\a')
#     ##
#     msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
#     return msg, seconds, minutes, hours
#
#
# def params_in_comp_graph():
#     import torch
#     import torch.nn as nn
#     from torchviz import make_dot
#     fc0 = nn.Linear(in_features=3, out_features=1)
#     params = [('fc0', fc0)]
#     mdl = nn.Sequential(OrderedDict(params))
#
#     x = torch.randn(1, 3)
#     # x.requires_grad = True  # uncomment to put in computation graph
#     y = torch.randn(1)
#
#     l = (mdl(x) - y) ** 2
#
#     # make_dot(l, params=dict(mdl.named_parameters()))
#     params = dict(mdl.named_parameters())
#     # params = {**params, 'x':x}
#     make_dot(l, params=params).render('data/debug/test_img_l', format='png')
#
#
# def check_if_tensor_is_detached():
#     a = torch.tensor([2.0], requires_grad=True)
#     b = a.detach()
#     b.requires_grad = True
#     print(a == b)
#     print(a is b)
#     print(a)
#     print(b)
#
#     la = (5.0 - a) ** 2
#     la.backward()
#     print(f'a.grad = {a.grad}')
#
#     lb = (6.0 - b) ** 2
#     lb.backward()
#     print(f'b.grad = {b.grad}')
#
#
# def deep_copy_issue():
#     params = OrderedDict([('fc1', nn.Linear(in_features=3, out_features=1))])
#     mdl0 = nn.Sequential(params)
#     mdl1 = copy.deepcopy(mdl0)
#     print(id(mdl0))
#     print(mdl0)
#     print(id(mdl1))
#     print(mdl1)
#     # my update
#     mdl1.fc1.weight = nn.Parameter(mdl1.fc1.weight + 1)
#     mdl2 = copy.deepcopy(mdl1)
#     print(id(mdl2))
#     print(mdl2)
#
#
# def download_mini_imagenet():
#     # download mini-imagenet automatically
#     import torch
#     import torch.nn as nn
#     import torchvision.datasets.utils as utils
#     from torchvision.datasets.utils import download_and_extract_archive
#     from torchvision.datasets.utils import download_file_from_google_drive
#
#     ## download mini-imagenet
#     # url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
#     file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
#     filename = 'miniImagenet.tgz'
#     root = '~/tmp/'  # dir to place downloaded file in
#     download_file_from_google_drive(file_id, root, filename)
#
#
# def extract():
#     from torchvision.datasets.utils import extract_archive
#     from_path = os.path.expanduser('~/Downloads/miniImagenet.tgz')
#     extract_archive(from_path)
#
#
# def download_and_extract_miniImagenet(root):
#     import os
#     from torchvision.datasets.utils import download_file_from_google_drive, extract_archive
#
#     ## download miniImagenet
#     # url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
#     file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
#     filename = 'miniImagenet.tgz'
#     download_file_from_google_drive(file_id, root, filename)
#     fpath = os.path.join(root, filename)  # this is what download_file_from_google_drive does
#     ## extract downloaded dataset
#     from_path = os.path.expanduser(fpath)
#     extract_archive(from_path)
#     ## remove the zip file
#     os.remove(from_path)
#
#
# def torch_concat():
#     import torch
#
#     g1 = torch.randn(3, 3)
#     g2 = torch.randn(3, 3)
#
#
# def inner_loop1():
#     n_inner_iter = 5
#     inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)
#
#     qry_losses = []
#     qry_accs = []
#     meta_opt.zero_grad()
#     for i in range(task_num):
#         with higher.innerloop_ctx(
#                 net, inner_opt, copy_initial_weights=False
#         ) as (fnet, diffopt):
#             # Optimize the likelihood of the support set by taking
#             # gradient steps w.r.t. the model's parameters.
#             # This adapts the model's meta-parameters to the task.
#             # higher is able to automatically keep copies of
#             # your network's parameters as they are being updated.
#             for _ in range(n_inner_iter):
#                 spt_logits = fnet(x_spt[i])
#                 spt_loss = F.cross_entropy(spt_logits, y_spt[i])
#                 diffopt.step(spt_loss)
#
#             # The final set of adapted parameters will induce some
#             # final loss and accuracy on the query dataset.
#             # These will be used to update the model's meta-parameters.
#             qry_logits = fnet(x_qry[i])
#             qry_loss = F.cross_entropy(qry_logits, y_qry[i])
#             qry_losses.append(qry_loss.detach())
#             qry_acc = (qry_logits.argmax(
#                 dim=1) == y_qry[i]).sum().item() / querysz
#             qry_accs.append(qry_acc)
#
#             # Update the model's meta-parameters to optimize the query
#             # losses across all of the tasks sampled in this batch.
#             # This unrolls through the gradient steps.
#             qry_loss.backward()
#
#     meta_opt.step()
#     qry_losses = sum(qry_losses) / task_num
#     qry_accs = 100. * sum(qry_accs) / task_num
#     i = epoch + float(batch_idx) / n_train_iter
#     iter_time = time.time() - start_time
#
#
# def inner_loop2():
#     n_inner_iter = 5
#     inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)
#
#     qry_losses = []
#     qry_accs = []
#     meta_opt.zero_grad()
#     meta_loss = 0
#     for i in range(task_num):
#         with higher.innerloop_ctx(
#                 net, inner_opt, copy_initial_weights=False
#         ) as (fnet, diffopt):
#             # Optimize the likelihood of the support set by taking
#             # gradient steps w.r.t. the model's parameters.
#             # This adapts the model's meta-parameters to the task.
#             # higher is able to automatically keep copies of
#             # your network's parameters as they are being updated.
#             for _ in range(n_inner_iter):
#                 spt_logits = fnet(x_spt[i])
#                 spt_loss = F.cross_entropy(spt_logits, y_spt[i])
#                 diffopt.step(spt_loss)
#
#             # The final set of adapted parameters will induce some
#             # final loss and accuracy on the query dataset.
#             # These will be used to update the model's meta-parameters.
#             qry_logits = fnet(x_qry[i])
#             qry_loss = F.cross_entropy(qry_logits, y_qry[i])
#             qry_losses.append(qry_loss.detach())
#             qry_acc = (qry_logits.argmax(
#                 dim=1) == y_qry[i]).sum().item() / querysz
#             qry_accs.append(qry_acc)
#
#             # Update the model's meta-parameters to optimize the query
#             # losses across all of the tasks sampled in this batch.
#             # This unrolls through the gradient steps.
#             # qry_loss.backward()
#             meta_loss += qry_loss
#
#     qry_losses = sum(qry_losses) / task_num
#     qry_losses.backward()
#     meta_opt.step()
#     qry_accs = 100. * sum(qry_accs) / task_num
#     i = epoch + float(batch_idx) / n_train_iter
#     iter_time = time.time() - start_time
#
#
# def error_unexpected_way_to_by_pass_safety():
#     # https://stackoverflow.com/questions/62415251/why-am-i-able-to-change-the-value-of-a-tensor-without-the-computation-graph-know
#
#     import torch
#     a = torch.tensor([1, 2, 3.], requires_grad=True)
#     # are detached tensor's leafs? yes they are
#     a_detached = a.detach()
#     # a.fill_(2) # illegal, warns you that a tensor which requires grads is used in an inplace op (so it won't be recorded in computation graph so it wont take the right derivative of the forward path as this op won't be in it)
#     a_detached.fill_(
#         2)  # weird that this one is allowed, seems to allow me to bypass the error check from the previous comment...?!
#     print(f'a = {a}')
#     print(f'a_detached = {a_detached}')
#     a.sum().backward()
#
#
# def detach_playground():
#     import torch
#
#     a = torch.tensor([1, 2, 3.], requires_grad=True)
#     # are detached tensor's leafs? yes they are
#     a_detached = a.detach()
#     print(f'a_detached.is_leaf = {a_detached.is_leaf}')
#     # is doing sum on the detached tensor a leaf? no
#     a_detached_sum = a.sum()
#     print(f'a_detached_sum.is_leaf = {a_detached_sum.is_leaf}')
#     # is detaching an intermediate tensor a leaf? yes
#     a_sum_detached = a.sum().detach()
#     print(f'a_sum_detached.is_leaf = {a_sum_detached.is_leaf}')
#     # shows they share they same data
#     print(f'a == a_detached = {a == a_detached}')
#     print(f'a is a_detached = {a is a_detached}')
#     a_detached.zero_()
#     print(f'a = {a}')
#     print(f'a_detached = {a_detached}')
#     # a.fill_(2) # illegal, warns you that a tensor which requires grads is used in an inplace op (so it won't be recorded in computation graph so it wont take the right derivative of the forward path as this op won't be in it)
#     a_detached.fill_(
#         2)  # weird that this one is allowed, seems to allow me to bypass the error check from the previous comment...?!
#     print(f'a = {a}')
#     print(f'a_detached = {a_detached}')
#     ## conclusion: detach basically creates a totally new tensor which cuts gradient computations to the original but shares the same memory with original
#     out = a.sigmoid()
#     out_detached = out.detach()
#     out_detached.zero_()
#     out.sum().backward()
#
#
# def clone_playground():
#     import torch
#
#     a = torch.tensor([1, 2, 3.], requires_grad=True)
#     a_clone = a.clone()
#     print(f'a_clone.is_leaf = {a_clone.is_leaf}')
#     print(f'a is a_clone = {a is a_clone}')
#     print(f'a == a_clone = {a == a_clone}')
#     print(f'a = {a}')
#     print(f'a_clone = {a_clone}')
#     # a_clone.fill_(2)
#     a_clone.mul_(2)
#     print(f'a = {a}')
#     print(f'a_clone = {a_clone}')
#     a_clone.sum().backward()
#     print(f'a.grad = {a.grad}')
#
#
# def clone_vs_deepcopy():
#     import copy
#     import torch
#
#     x = torch.tensor([1, 2, 3.])
#     x_clone = x.clone()
#     x_deep_copy = copy.deepcopy(x)
#     #
#     x.mul_(-1)
#     print(f'x = {x}')
#     print(f'x_clone = {x_clone}')
#     print(f'x_deep_copy = {x_deep_copy}')
#     print()
#
#
# def inplace_playground():
#     import torch
#
#     x = torch.tensor([1, 2, 3.], requires_grad=True)
#     y = x + 1
#     print(f'x.is_leaf = {x.is_leaf}')
#     print(f'y.is_leaf = {y.is_leaf}')
#     x += 1  # not allowed because x is a leaf, since changing the value of a leaf with an inplace forgets it's value then backward wouldn't work IMO (though its not the official response)
#     print(f'x.is_leaf = {x.is_leaf}')
#
#
# def copy_initial_weights_playground_original():
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     import higher
#     import numpy as np
#
#     np.random.seed(1)
#     torch.manual_seed(3)
#     N = 100
#     actual_multiplier = 3.5
#     meta_lr = 0.00001
#     loops = 5  # how many iterations in the inner loop we want to do
#
#     x = torch.tensor(np.random.random((N, 1)), dtype=torch.float64)  # features for inner training loop
#     y = x * actual_multiplier  # target for inner training loop
#     model = nn.Linear(1, 1, bias=False).double()  # simplest possible model - multiple input x by weight w without bias
#     meta_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)
#
#     def run_inner_loop_once(model, verbose, copy_initial_weights):
#         lr_tensor = torch.tensor([0.3], requires_grad=True)
#         momentum_tensor = torch.tensor([0.5], requires_grad=True)
#         opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
#         with higher.innerloop_ctx(model, opt, copy_initial_weights=copy_initial_weights,
#                                   override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
#             for j in range(loops):
#                 if verbose:
#                     print('Starting inner loop step j=={0}'.format(j))
#                     print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(
#                         list(fmodel.parameters(time=j)))))
#                     print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (
#                             list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
#                 out = fmodel(x)
#                 if verbose:
#                     print(
#                         '    Notice how `out` is `x` multiplied by the latest version of weight: {0:.4} * {1:.4} == {2:.4}'.format(
#                             x[0, 0].item(), list(fmodel.parameters())[0].item(), out[0].item()))
#                 loss = ((out - y) ** 2).mean()
#                 diffopt.step(loss)
#
#             if verbose:
#                 # after all inner training let's see all steps' parameter tensors
#                 print()
#                 print("Let's print all intermediate parameters versions after inner loop is done:")
#                 for j in range(loops + 1):
#                     print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
#                 print()
#
#             # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
#             weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
#             meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier) ** 2
#             print('  Final meta-loss: {0}'.format(meta_loss.item()))
#             meta_loss.backward()  # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
#             if verbose:
#                 print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad,
#                                                                                                 momentum_tensor.grad))
#                 print(
#                     '  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
#             return meta_loss.item()
#
#     print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
#     meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
#     print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
#         list(model.parameters())[0].grad))
#
#     print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
#     meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
#     print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
#         list(model.parameters())[0].grad))
#
#     print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
#     final_meta_gradient = list(model.parameters())[0].grad.item()
#     # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
#     # a bunch of hand-wavy statements and difficult to read code.
#     # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
#     meta_opt.step()
#     meta_opt.zero_grad()
#     meta_step = - meta_lr * final_meta_gradient  # how much meta_opt actually shifted inital weight value
#     meta_loss_val3 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
#
#
# def copy_initial_weights_playground():
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim
#     import higher
#     import numpy as np
#
#     np.random.seed(1)
#     torch.manual_seed(3)
#     N = 100
#     actual_multiplier = 3.5  # the parameters we want the model to learn
#     meta_lr = 0.00001
#     loops = 5  # how many iterations in the inner loop we want to do
#
#     x = torch.randn(N, 1)  # features for inner training loop
#     y = x * actual_multiplier  # target for inner training loop
#     model = nn.Linear(1, 1,
#                       bias=False)  # model(x) = w*x, simplest possible model - multiple input x by weight w without bias. goal is to w~~actualy_multiplier
#     outer_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)
#
#     def run_inner_loop_once(model, verbose, copy_initial_weights):
#         lr_tensor = torch.tensor([0.3], requires_grad=True)
#         momentum_tensor = torch.tensor([0.5], requires_grad=True)
#         inner_opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
#         with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=copy_initial_weights,
#                                   override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
#             for j in range(loops):
#                 if verbose:
#                     print('Starting inner loop step j=={0}'.format(j))
#                     print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(
#                         list(fmodel.parameters(time=j)))))
#                     print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (
#                             list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
#                 out = fmodel(x)
#                 if verbose:
#                     print(
#                         f'    Notice how `out` is `x` multiplied by the latest version of weight: {x[0, 0].item()} * {list(fmodel.parameters())[0].item()} == {out[0].item()}')
#                 loss = ((out - y) ** 2).mean()
#                 diffopt.step(loss)
#
#             if verbose:
#                 # after all inner training let's see all steps' parameter tensors
#                 print()
#                 print("Let's print all intermediate parameters versions after inner loop is done:")
#                 for j in range(loops + 1):
#                     print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
#                 print()
#
#             # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
#             weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
#             meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier) ** 2
#             print('  Final meta-loss: {0}'.format(meta_loss.item()))
#             meta_loss.backward()  # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
#             if verbose:
#                 print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad,
#                                                                                                 momentum_tensor.grad))
#                 print(
#                     '  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
#             return meta_loss.item()
#
#     print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
#     meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
#     print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
#         list(model.parameters())[0].grad))
#
#     print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
#     meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
#     print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(
#         list(model.parameters())[0].grad))
#
#     print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
#     final_meta_gradient = list(model.parameters())[0].grad.item()
#     # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
#     # a bunch of hand-wavy statements and difficult to read code.
#     # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
#     outer_opt.step()
#     outer_opt.zero_grad()
#     meta_step = - meta_lr * final_meta_gradient  # how much meta_opt actually shifted inital weight value
#     meta_loss_val3 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
#
#     meta_loss_gradient_approximation = (meta_loss_val3 - meta_loss_val2) / meta_step
#
#     print()
#     print(
#         'Side-by-side meta_loss_gradient_approximation and gradient computed by `higher` lib: {0:.4} VS {1:.4}'.format(
#             meta_loss_gradient_approximation, final_meta_gradient))
#
#
# def tqdm_torchmeta():
#     from torchvision.transforms import Compose, Resize, ToTensor
#
#     import torchmeta
#     from torchmeta.datasets.helpers import miniimagenet
#
#     from pathlib import Path
#     from types import SimpleNamespace
#
#     from tqdm import tqdm
#
#     ## get args
#     args = SimpleNamespace(episodes=5, n_classes=5, k_shot=5, k_eval=15, meta_batch_size=1, n_workers=4)
#     args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
#
#     ## get meta-batch loader
#     train_transform = Compose([Resize(84), ToTensor()])
#     dataset = miniimagenet(
#         args.data_root,
#         ways=args.n_classes,
#         shots=args.k_shot,
#         test_shots=args.k_eval,
#         meta_split='train',
#         download=False)
#     dataloader = torchmeta.utils.data.BatchMetaDataLoader(
#         dataset,
#         batch_size=args.meta_batch_size,
#         num_workers=args.n_workers)
#
#     with tqdm(dataset):
#         print(f'len(dataloader)= {len(dataloader)}')
#         for episode, batch in enumerate(dataloader):
#             print(f'episode = {episode}')
#             train_inputs, train_labels = batch["train"]
#             print(f'train_labels[0] = {train_labels[0]}')
#             print(f'train_inputs.size() = {train_inputs.size()}')
#             pass
#             if episode >= args.episodes:
#                 break
#
#
# # if __name__ == "__main__":
# #     start = time.time()
# #     print('pytorch playground!')
# #     # params_in_comp_graph()
# #     # check_if_tensor_is_detached()
# #     # deep_copy_issue()
# #     # download_mini_imagenet()
# #     # extract()
# #     # download_and_extract_miniImagenet(root='~/tmp')
# #     # download_and_extract_miniImagenet(root='~/automl-meta-learning/data')
# #     # torch_concat()
# #     # detach_vs_cloe()
# #     # error_unexpected_way_to_by_pass_safety()
# #     # clone_playground()
# #     # inplace_playground()
# #     # clone_vs_deepcopy()
# #     # copy_initial_weights_playground()
# #     tqdm_torchmeta()
# #     print('--> DONE')
# #     time_passed_msg, _, _, _ = report_times(start)
# #     print(f'--> {time_passed_msg}')
#
# # %%
#
# import sys
#
# print(sys.version)  ##
# print(sys.path)
#
#
# def helloworld():
#     print('helloworld')
#     print('hello12345')
#
#
# def union_dicts():
#     d1 = {'x': 1}
#     d2 = {'y': 2, 'z': 3}
#     d_union = {**d1, **d2}
#     print(d_union)
#
#
# def get_stdout_old():
#     import sys
#
#     # contents = ""
#     # #with open('some_file.txt') as f:
#     # #with open(sys.stdout,'r') as f:
#     # # sys.stdout.mode = 'r'
#     # for line in sys.stdout.readlines():
#     #     contents += line
#     # print(contents)
#
#     # print(sys.stdout)
#     # with open(sys.stdout.buffer) as f:
#     #     print(f.readline())
#
#     # import subprocess
#
#     # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     # stdout = []
#     # while True:
#     #     line = p.stdout.readline()
#     #     stdout.append(line)
#     #     print( line )
#     #     if line == '' and p.poll() != None:
#     #         break
#     # print( ''.join(stdout) )
#
#     import sys
#     myfile = "input.txt"
#
#     def print(*args):
#         __builtins__.print(*args, file=sys.__stdout__)
#         with open(myfile, "a+") as f:
#             __builtins__.print(*args, file=f)
#
#     print('a')
#     print('b')
#     print('c')
#
#     repr(sys.stdout)
#
#
# def get_stdout():
#     import sys
#     myfile = "my_stdout.txt"
#
#     # redefine print
#     def print(*args):
#         __builtins__.print(*args, file=sys.__stdout__)  # prints to terminal
#         with open(myfile, "a+") as f:
#             __builtins__.print(*args, file=f)  # saves in a file
#
#     print('a')
#     print('b')
#     print('c')
#
#
# def logging_basic():
#     import logging
#     logging.warning('Watch out!')  # will print a message to the console
#     logging.info('I told you so')  # will not print anything
#
#
# def logging_to_file():
#     import logging
#     logging.basicConfig(filename='example.log', level=logging.DEBUG)
#     # logging.
#     logging.debug('This message should go to the log file')
#     logging.info('So should this')
#     logging.warning('And this, too')
#
#
# def logging_to_file_INFO_LEVEL():
#     import logging
#     import sys
#     format = '{asctime}:{levelname}:{name}:lineno {lineno}:{message}'
#     logging.basicConfig(filename='example.log', level=logging.INFO, format=format, style='{')
#     # logging.basicConfig(stream=sys.stdout,level=logging.INFO,format=format,style='{')
#     # logging.
#     logging.debug('This message should NOT go to the log file')
#     logging.info('This message should go to log file')
#     logging.warning('This, too')
#
#
# def logger_SO_print_and_write_to_my_stdout():
#     """My sample logger code to print to screen and write to file (the same thing).
#
#     Note: trying to replace this old answer of mine using a logger:
#     - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
#
#     Credit:
#     - https://www.youtube.com/watch?v=jxmzY9soFXg&t=468s
#     - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
#     - https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716
#
#     Other resources:
#     - https://docs.python-guide.org/writing/logging/
#     - https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
#     - https://stackoverflow.com/questions/61084916/how-does-one-make-an-already-opened-file-readable-e-g-sys-stdout/61255375#61255375
#     """
#     from pathlib import Path
#     import logging
#     import os
#     import sys
#     from datetime import datetime
#
#     ## create directory (& its parents) if it does not exist otherwise do nothing :)
#     # get current time
#     current_time = datetime.now().strftime('%b%d_%H-%M-%S')
#     logs_dirpath = Path(f'~/logs/python_playground_logs_{current_time}/').expanduser()
#     logs_dirpath.mkdir(parents=True, exist_ok=True)
#     my_stdout_filename = logs_dirpath / Path('my_stdout.log')
#     # remove my_stdout if it exists (note you can also just create a new log dir/file each time or append to the end of the log file your using)
#     # os.remove(my_stdout_filename) if os.path.isfile(my_stdout_filename) else None
#
#     ## create top logger
#     logger = logging.getLogger(
#         __name__)  # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
#     logger.setLevel(
#         logging.DEBUG)  # note: use logging.DEBUG, CAREFUL with logging.UNSET: https://stackoverflow.com/questions/21494468/about-notset-in-python-logging/21494716#21494716
#
#     ## log to my_stdout.log file
#     file_handler = logging.FileHandler(filename=my_stdout_filename)
#     # file_handler.setLevel(logging.INFO) # not setting it means it inherits the logger. It will log everything from DEBUG upwards in severity to this handler.
#     log_format = "{asctime}:{levelname}:{lineno}:{name}:{message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
#     formatter = logging.Formatter(fmt=log_format, style='{')  # set the logging format at for this handler
#     file_handler.setFormatter(fmt=formatter)
#
#     ## log to stdout/screen
#     stdout_stream_handler = logging.StreamHandler(
#         stream=sys.stdout)  # default stderr, though not sure the advatages of logging to one or the other
#     # stdout_stream_handler.setLevel(logging.INFO) # Note: having different set levels means that we can route using a threshold what gets logged to this handler
#     log_format = "{name}:{levelname}:-> {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
#     formatter = logging.Formatter(fmt=log_format, style='{')  # set the logging format at for this handler
#     stdout_stream_handler.setFormatter(fmt=formatter)
#
#     logger.addHandler(hdlr=file_handler)  # add this file handler to top logger
#     logger.addHandler(hdlr=stdout_stream_handler)  # add this file handler to top logger
#
#     logger.log(logging.NOTSET, 'notset')
#     logger.debug('debug')
#     logger.info('info')
#     logger.warning('warning')
#     logger.error('error')
#     logger.critical('critical')
#
#
# def logging_unset_level():
#     """My sample logger explaining UNSET level
#
#     Resources:
#     - https://stackoverflow.com/questions/21494468/about-notset-in-python-logging
#     - https://www.youtube.com/watch?v=jxmzY9soFXg&t=468s
#     - https://github.com/CoreyMSchafer/code_snippets/tree/master/Logging-Advanced
#     """
#     import logging
#
#     logger = logging.getLogger(
#         __name__)  # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
#     print(f'DEFAULT VALUE: logger.level = {logger.level}')
#
#     file_handler = logging.FileHandler(filename='my_log.log')
#     log_format = "{asctime}:{levelname}:{lineno}:{name}:{message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
#     formatter = logging.Formatter(fmt=log_format, style='{')
#     file_handler.setFormatter(fmt=formatter)
#
#     stdout_stream_handler = logging.StreamHandler(stream=sys.stdout)
#     stdout_stream_handler.setLevel(logging.INFO)
#     log_format = "{name}:{levelname}:-> {message}"  # see for logrecord attributes https://docs.python.org/3/library/logging.html#logrecord-attributes
#     formatter = logging.Formatter(fmt=log_format, style='{')
#     stdout_stream_handler.setFormatter(fmt=formatter)
#
#     logger.addHandler(hdlr=file_handler)
#     logger.addHandler(hdlr=stdout_stream_handler)
#
#     logger.log(logging.NOTSET, 'notset')
#     logger.debug('debug')
#     logger.info('info')
#     logger.warning('warning')
#     logger.error('error')
#     logger.critical('critical')
#
#
# def logger():
#     from pathlib import Path
#     import logging
#
#     # create directory (& its parents) if it does not exist otherwise do nothing :)
#     logs_dirpath = Path('~/automl-meta-learning/logs/python_playground_logs/').expanduser()
#     logs_dirpath.mkdir(parents=True, exist_ok=True)
#     my_stdout_filename = logs_dirpath / Path('my_stdout.log')
#     # remove my_stdout if it exists (used to have this but now I decided to create a new log & file each)
#     # os.remove(my_stdout_filename) if os.path.isfile(my_stdout_filename) else None
#
#     logger = logging.getLogger(
#         __name__)  # loggers are created in hierarchy using dot notation, thus __name__ ensures no name collisions.
#     logger.setLevel(logging.INFO)
#
#     log_format = "{asctime}:{levelname}:{name}:{message}"
#     formatter = logging.Formatter(fmt=log_format, style='{')
#
#     file_handler = logging.FileHandler(filename=my_stdout_filename)
#     file_handler.setFormatter(fmt=formatter)
#
#     logger.addHandler(hdlr=file_handler)
#     logger.addHandler(hdlr=logging.StreamHandler())
#
#     for i in range(3):
#         logger.info(f'i = {i}')
#
#     logger.info(f'logger DONE')
#
#
# def logging_example_from_youtube():
#     """https://github.com/CoreyMSchafer/code_snippets/blob/master/Logging-Advanced/employee.py
#     """
#     import logging
#     import pytorch_playground  # has employee class & code
#     import sys
#
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.DEBUG)
#
#     formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
#
#     file_handler = logging.FileHandler('sample.log')
#     file_handler.setLevel(logging.ERROR)
#     file_handler.setFormatter(formatter)
#
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
#
#     logger.critical('not really critical :P')
#
#     def add(x, y):
#         """Add Function"""
#         return x + y
#
#     def subtract(x, y):
#         """Subtract Function"""
#         return x - y
#
#     def multiply(x, y):
#         """Multiply Function"""
#         return x * y
#
#     def divide(x, y):
#         """Divide Function"""
#         try:
#             result = x / y
#         except ZeroDivisionError:
#             logger.exception('Tried to divide by zero')
#         else:
#             return result
#
#     logger.info(
#         'testing if log info is going to print to screen. it should because everything with debug or above is printed since that stream has that level.')
#
#     num_1 = 10
#     num_2 = 0
#
#     add_result = add(num_1, num_2)
#     logger.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))
#
#     sub_result = subtract(num_1, num_2)
#     logger.debug('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))
#
#     mul_result = multiply(num_1, num_2)
#     logger.debug('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))
#
#     div_result = divide(num_1, num_2)
#     logger.debug('Div: {} / {} = {}'.format(num_1, num_2, div_result))
#
#
# def plot():
#     """
#     source:
#         - https://www.youtube.com/watch?v=UO98lJQ3QGI
#         - https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Matplotlib/01-Introduction/finished_code.py
#     """
#     from matplotlib import pyplot as plt
#
#     plt.xkcd()
#
#     ages_x = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
#               36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
#
#     py_dev_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287, 63016, 65998, 70003, 70000,
#                 71496, 75370, 83640, 84666,
#                 84392, 78254, 85000, 87038, 91991, 100000, 94796, 97962, 93302, 99240, 102736, 112285, 100771, 104708,
#                 108423, 101407, 112542, 122870, 120000]
#     plt.plot(ages_x, py_dev_y, label='Python')
#
#     js_dev_y = [16446, 16791, 18942, 21780, 25704, 29000, 34372, 37810, 43515, 46823, 49293, 53437, 56373, 62375, 66674,
#                 68745, 68746, 74583, 79000,
#                 78508, 79996, 80403, 83820, 88833, 91660, 87892, 96243, 90000, 99313, 91660, 102264, 100000, 100000,
#                 91660, 99240, 108000, 105000, 104000]
#     plt.plot(ages_x, js_dev_y, label='JavaScript')
#
#     dev_y = [17784, 16500, 18012, 20628, 25206, 30252, 34368, 38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928,
#              67317, 68748, 73752, 77232,
#              78000, 78508, 79536, 82488, 88935, 90000, 90056, 95000, 90000, 91633, 91660, 98150, 98964, 100000, 98988,
#              100000, 108923, 105000, 103117]
#     plt.plot(ages_x, dev_y, color='#444444', linestyle='--', label='All Devs')
#
#     plt.xlabel('Ages')
#     plt.ylabel('Median Salary (USD)')
#     plt.title('Median Salary (USD) by Age')
#
#     plt.legend()
#
#     plt.tight_layout()
#
#     plt.savefig('plot.png')
#
#     plt.show()
#
#
# def subplot():
#     """https://github.com/CoreyMSchafer/code_snippets/blob/master/Python/Matplotlib/10-Subplots/finished_code.py
#     """
#
#     import pandas as pd
#     from matplotlib import pyplot as plt
#
#     plt.style.use('seaborn')
#
#     data = read_csv('data.csv')
#     ages = data['Age']
#     dev_salaries = data['All_Devs']
#     py_salaries = data['Python']
#     js_salaries = data['JavaScript']
#
#     fig1, ax1 = plt.subplots()
#     fig2, ax2 = plt.subplots()
#
#     ax1.plot(ages, dev_salaries, color='#444444',
#              linestyle='--', label='All Devs')
#
#     ax2.plot(ages, py_salaries, label='Python')
#     ax2.plot(ages, js_salaries, label='JavaScript')
#
#     ax1.legend()
#     ax1.set_title('Median Salary (USD) by Age')
#     ax1.set_ylabel('Median Salary (USD)')
#
#     ax2.legend()
#     ax2.set_xlabel('Ages')
#     ax2.set_ylabel('Median Salary (USD)')
#
#     plt.tight_layout()
#
#     plt.show()
#
#     fig1.savefig('fig1.png')
#     fig2.savefig('fig2.png')
#
#
# def import_utils_test():
#     import uutils
#     import uutils.utils as utils
#     from uutils.utils import logger
#
#     print(uutils)
#     print(utils)
#     print(logger)
#
#     print()
#
#
# def sys_path():
#     """
#
#     python -c "import sys; print(sys.path)”
#
#     python -c "import sys; [print(p) for p in sys.path]"
#     """
#     import sys
#
#     def path():
#         import sys
#         [print(p) for p in sys.path]
#
#     for path in sys.path:
#         print(path)
#
#
# def pycharm_playground():
#     import tqdm
#
#     print('running pycharm playground...')
#
#     b = 0
#     print(b)
#     print('Intermediate print line')
#     print(b)
#     print(b)
#     print('Done!')
#
#
# if __name__ == '__main__':
#     # union_dicts()
#     # get_stdout()
#     # logger()
#     # logger_SO_print_and_write_to_my_stdout()
#     # logging_basic()
#     # logging_to_file()
#     # logging_to_file()
#     # logging_to_file_INFO_LEVEL()
#     # logging_example_from_youtube()
#     # logging_unset_level()
#     # import_utils_test()
#     pycharm_playground()
#     print('\n---> DONE\a\n\n')  ## HIii
#
# # %%
#
# import sys
#
# print(sys.version)
#
# # %%
#
# ## dictionary comprehension looping
#
# d = {'a': 0, 'b': 1}
# lst1 = [f'key:{k}' for k in d]
# lst2 = [f'key:{k}, value:{v}' for k, v in d.items()]
#
# print(lst1)
# print(lst2)
#
# # %%
#
# ## merging two dictionaries
#
# d1 = {'a': 0, 'b': 1}
# d2 = {'c': 2, 'd': 3}
# d3 = {'e': 4, 'f': 5, 'g': 6}
# d = {**d1, **d2, **d3}
#
# print(d)
#
# # %%
#
#
# from collections import OrderedDict
#
# od = OrderedDict([
#     ('first', 1)
# ])
#
# print(od)
# od['first'] = 2
# print(od)
#
# lst = sum([i for i in range(3)])
# print(lst)
# od3 = OrderedDict([(i, i) for i in range(3)])
# print(od3)
# print(3 + float('Inf'))
#
# # %%
#
# # import pathlib
# # from pathlib import Path
# #
# #
# # def make_dirpath_current_datetime_hostname(path=None, comment='', replace_dots=True):
# #     '''
# #     make dir string: runs/CURRENT_DATETIME_HOSTNAME
# #     '''
# #     import socket
# #     import os
# #     from datetime import datetime
# #     # check if root is a PosixPath object
# #     if type(path) != pathlib.PosixPath and path is not None:
# #         path = Path(path)
# #     current_time = datetime.now().strftime('%b%d_%H-%M-%S')
# #     log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
# #     log_dir = Path(log_dir)
# #     print(log_dir._str)
# #     if replace_dots:
# #         log_dir = Path(log_dir._str.replace('.', '_'))
# #     if path is not None:
# #         log_dir = path / log_dir
# #     return log_dir
# #
# #
# # print(type(Path('~')) == pathlib.PosixPath)
# # print()
# #
# # log_dir = make_dirpath_current_datetime_hostname()
# # print(log_dir)
# # log_dir = make_dirpath_current_datetime_hostname('~')
# # print(log_dir)
# # log_dir = make_dirpath_current_datetime_hostname('~', '_jupyter')
# # print(log_dir)
# # log_dir = make_dirpath_current_datetime_hostname('~').expanduser()
# # print(log_dir)
# #
# # string = "geeks for geeks geeks geeks geeks"
# # # Prints the string by replacing geeks by Geeks
# # print(string.replace("geeks", "Geeks"))
# #
# # log_dir = make_dirpath_current_datetime_hostname('~', '_jupyter', True)
# # print(log_dir)
#
# # %%
#
# # adding keys to empty dic
#
# d = {}
# d['a'] = 3
# print(d)
#
# # %%
#
# # unpack list?
#
# (a, b, c) = [1, 2, 3]
# print(a)
#
#
# # %%
#
# ## kwargs
#
# def f(*args, **kwargs):
#     print(args)
#     print(kwargs)
#
#
# f()
# f(1, 2, 3, a=1, b=2, c=3)
#
# # %%
#
# #
# # import json
# #
# # from pathlib import Path
# #
# # p = Path('~/').expanduser()
# # with open(p) as f:
# #     data = json.load(f)
# #     print(data)
# #     print(data['password'])
#
# # %%
#
# import subprocess
#
# from subprocess import Popen, PIPE, STDOUT
#
# cmd = 'ls /etc/fstab /etc/non-existent-file'
# p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
# output = p.stdout.read()
# print(output)
#
# # %%
#
# import sys
#
# print('a')
#
# print(sys.stdout)
#
# # %%
#
# # from pathlib import Path
# #
# #
# # def send_email(subject, message, destination, password_path=None):
# #     """ Send an e-mail from with message to destination email.
# #
# #     NOTE: if you get an error with google gmails you might need to do this:
# #     https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
# #     To use an app password:
# #     https://stackoverflow.com/questions/60975490/how-does-one-send-an-e-mail-from-python-not-using-gmail
# #
# #     Arguments:
# #         message {str} -- message string to send.
# #         destination {str} -- destination email (as string)
# #     """
# #     from socket import gethostname
# #     from email.message import EmailMessage
# #     import smtplib
# #     import json
# #     import sys
# #
# #     server = smtplib.SMTP('smtp.gmail.com', 587)
# #     smtplib.stdout = sys.stdout
# #     server.starttls()
# #     with open(password_path) as f:
# #         config = json.load(f)
# #         server.login('slurm.miranda@gmail.com', config['password'])
# #
# #         # craft message
# #         msg = EmailMessage()
# #
# #         # message = f'{message}\nSend from Hostname: {gethostname()}'
# #         # msg.set_content(message)
# #         msg['Subject'] = subject
# #         msg['From'] = 'slurm.miranda@gmail.com'
# #         msg['To'] = destination
# #         # send msg
# #         server.send_message(msg)
# #
# #
# # ##
# # print("-------> HELLOWWWWWWWW")
# # p = Path('~/automl-meta-learning/automl/experiments/pw_app.config.json').expanduser()
# # send_email(subject='TEST: send_email2', message='MESSAGE', destination='brando.science@gmail.com', password_path=p)
#
# # %%
#
# """
# Demo of the errorbar function, including upper and lower limits
# """
# import numpy as np
# import matplotlib.pyplot as plt
#
# import matplotlib as mpl
#
# mpl.rcParams["errorbar.capsize"] = 3
#
# # https://stackoverflow.com/questions/61415955/why-dont-the-error-limits-in-my-plots-show-in-matplotlib
#
# # example data
# x = np.arange(0.5, 5.5, 0.5)
# y = np.exp(-x)
# xerr = 0.1
# yerr = 0.2
# ls = 'dotted'
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# # standard error bars
# plt.errorbar(x, y, xerr=xerr, yerr=yerr, ls=ls, color='blue')
#
# # including upper limits
# uplims = np.zeros(x.shape)
# uplims[[1, 5, 9]] = True
# plt.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims, ls=ls,
#              color='green')
#
# # including lower limits
# lolims = np.zeros(x.shape)
# lolims[[2, 4, 8]] = True
# plt.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims, ls=ls,
#              color='red')
#
# # including upper and lower limits
# plt.errorbar(x, y + 1.5, marker='o', ms=8, xerr=xerr, yerr=yerr,
#              lolims=lolims, uplims=uplims, ls=ls, color='magenta')
#
# # including xlower and xupper limits
# xerr = 0.2
# yerr = np.zeros(x.shape) + 0.2
# yerr[[3, 6]] = 0.3
# xlolims = lolims
# xuplims = uplims
# lolims = np.zeros(x.shape)
# uplims = np.zeros(x.shape)
# lolims[[6]] = True
# uplims[[3]] = True
# plt.errorbar(x, y + 2.1, marker='o', ms=8, xerr=xerr, yerr=yerr,
#              xlolims=xlolims, xuplims=xuplims, uplims=uplims, lolims=lolims,
#              ls='none', mec='blue', capsize=0, color='cyan')
#
# ax.set_xlim((0, 5.5))
# ax.set_title('Errorbar upper and lower limits')
# plt.show()
#
# # %%
#
# from types import SimpleNamespace
# from pathlib import Path
# from pprint import pprint
#
# args = SimpleNamespace()
# args.data_root = "~/automl-meta-learning/data/miniImagenet"
#
# args.data_root = Path(args.data_root).expanduser()
#
# print(args)
#
# # pprint(dir(args.data_root))
# print(args.data_root.name)
# print('miniImagenet' in args.data_root.name)
#
# # %%
#
# ## sampling N classes for len(meta-set)
# # In sampling without replacement, each sample unit of
# # the population has only one chance to be selected in the sample.
# # because you are NOT replacing what you removed.
#
# import random
#
# N = 5
# len_meta_set = 64
# sample = random.sample(range(0, len_meta_set), N)
#
# print(sample)
#
# for i, n in enumerate(sample):
#     print(f'i={i}\nn={n}\n')
#
#
# # %%
#
# # iterator https://www.programiz.com/python-programming/iterator
#
# class Counter:
#
#     def __init__(self, max=0):
#         self.max = max  # returns up to and including that number
#
#     def __iter__(self):
#         self.n = 0
#         return self
#
#     def __next__(self):
#         if self.n <= self.max:
#             current_count = self.n
#             self.n += 1
#             print(f'current_count = {current_count}')
#             print(f'self.n = {self.n}')
#             print(self.n is current_count)
#             return current_count
#         else:
#             raise StopIteration
#
#
# ## test it
#
# counter = iter(Counter(max=0))
# for count in counter:
#     print(f'count = {count}')
#
# # %%
#
# from tqdm import tqdm
#
# print(tqdm)
#
# lst = range(3)
# print(type(lst))
#
# with tqdm(iter(lst), total=5) as tlist:
#     print(f'tlist = {type(tlist)}')
#     for i in tlist:
#         print(i)
#
# # %%
#
# from tqdm import tqdm
#
#
# class Plus2:
#
#     def __init__(self, max=0):
#         self.max = max  # returns up to and including that number
#
#     def __iter__(self):
#         self.it = 0
#         self.tot = 0
#         return self
#
#     def __next__(self):
#         if self.it <= self.max:
#             self.it += 1
#             self.tot += 2
#             return self.tot
#         else:
#             raise StopIteration
#
#     def __len__(self):
#         return self.max
#
#
# ##
# counter = iter(Plus2(max=int(100000)))
# with tqdm(counter, total=len(counter)) as tqcounter:
#     for idx, pow2 in enumerate(tqcounter):
#         print()
#         print(f'idx = {idx}')
#         print(f'powd2 = {pow2}')
#         pass
#
# # %%
#
# from tqdm import tqdm
#
# for i in tqdm(range(int(9e6))):
#     pass
#
# # %%
#
# from tqdm import tqdm
#
# import time
#
# with tqdm(range(int(5))) as trange:
#     for i in trange:
#         print(f'\ni = {i}')
#         print('done\n')
#         time.sleep(1)
#         pass
#
# # %%
#
# # zip, it aligns elements in one list to elements in the other
#
# l1 = [0, 1, 2]
# l2 = ['a', 'b', 'c']
#
# print(list(zip(l1, l2)))
#
# # %%
#
# from tqdm import tqdm
# import time
#
# lst = range(10000000)
# total = 2
#
# with tqdm(lst, total=total) as tlst:
#     i = 0
#     for _, element in enumerate(tlst):
#         print(f'\n->i = {i}\n')
#         time.sleep(0.2)
#         i += 1
#         if i >= total:
#             break
#
# print('\n--> DONE \a')
#
# # %%
#
# from tqdm import tqdm
# import time
#
# lst = range(10000000)
# total = 2
#
# with tqdm(lst, total=total) as tlst:
#     for idx, element in enumerate(tlst):
#         print(f'\n->idx = {idx}\n')
#         time.sleep(0.2)
#         if idx >= total:
#             break
#
# print('\n--> DONE \a')
#
# # %%
#
# from tqdm import tqdm
# import time
#
# lst = range(10000000)
# total = 2
#
# with tqdm(range(total)) as tcounter:
#     lst = iter(lst)
#     for idx, element in enumerate(tcounter):
#         print(f'\n->idx = {idx}\n')
#         time.sleep(0.2)
#
# print('\n--> DONE \a')
#
# # %%
#
# # Question: Do detached() tensors track their own gradients seperately?
# # Ans: Yes!
# # https://discuss.pytorch.org/t/why-is-the-clone-operation-part-of-the-computation-graph-is-it-even-differentiable/67054/11
#
# import torch
#
# a = torch.tensor([2.0], requires_grad=True)
# b = a.detach()
# b.requires_grad = True
#
# la = (5.0 - a) ** 2
# la.backward()
# print(f'a.grad = {a.grad}')
#
# lb = (6.0 - b) ** 2
# lb.backward()
# print(f'b.grad = {b.grad}')
#
# # %%
#
# import torch
# import torch.nn as nn
#
# from collections import OrderedDict
#
# params = OrderedDict([
#     ('fc0', nn.Linear(in_features=4, out_features=4)),
#     ('ReLU0', nn.ReLU()),
#     ('fc1', nn.Linear(in_features=4, out_features=1))
# ])
# mdl = nn.Sequential(params)
#
# print(params)
# print(mdl._parameters)
# print(params == params)
# print(mdl._parameters == params)
# print(mdl._modules)
#
# print()
# for name, w in mdl.named_parameters():
#     print(name, w.norm(2))
#
# print()
# # mdl._modules['fc0'] = nn.Linear(10,11)
# mdl._modules[0]
#
# for name, w in mdl.named_parameters():
#     print(name, w.norm(2))
#
# # %%
#
# ## Q: are parameters are in computation graph?
# import torch
# import torch.nn as nn
# from torchviz import make_dot
#
# from collections import OrderedDict
#
# fc0 = nn.Linear(in_features=3, out_features=1)
# params = [('fc0', fc0)]
# mdl = nn.Sequential(OrderedDict(params))
#
# x = torch.randn(1, 3)
# y = torch.randn(1)
#
# l = (mdl(x) - y) ** 2
#
# # make_dot(l,{x:'x',y:'y','fc0':fc0})
# print(fc0.weight)
# print(fc0.bias)
# print(fc0.weight.to_tens)
# print()
# # make_dot(l,{x:'x',y:'y','fc0':fc0})
# make_dot(l, {'x': x, 'y': y})
# make_dot(l)
#
# # %%
#
# '''
# expand
# '''
#
# import torch
#
# x = torch.randn([2, 3, 4, 5])
#
# # h_0 of shape (num_layers * num_directions, batch, hidden_size)
# h = torch.randn([1, 4, 8])
#
# x_mean = x.mean()
# print(x_mean.size())
# print(x_mean)
# x = x_mean.expand_as(h)
# print(x.size())
# print(x)
#
# # %%
#
# import torch
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# print(device)
# type(device)
# print(device == 'cpu')
# device.type
#
# # %%
#
# # THIS WORKS
#
# from torch.utils.tensorboard import SummaryWriter
#
# from pathlib import Path
#
# # log_dir (string) – Save directory location.
# # Default is runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.
#
# tb = SummaryWriter()
# tb.add_scalar('loss', 111)
#
# # %%
#
# from torch.utils.tensorboard import SummaryWriter
#
# from pathlib import Path
#
#
# def CURRENT_DATETIME_HOSTNAME(comment=''):
#     # if not log_dir:
#     import socket
#     import os
#     from datetime import datetime
#     current_time = datetime.now().strftime('%b%d_%H-%M-%S')
#     log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
#     return Path(log_dir)
#
#
# # log_dir (string) – Save directory location.
# # Default is runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.
# # tensorboard --logdir=runs
# log_dir = (Path('~/automl-meta-learning/') / CURRENT_DATETIME_HOSTNAME()).expanduser()
# print(log_dir)
# tb = SummaryWriter(log_dir=log_dir)
# tb.add_scalar('loss', 15)
#
# # %%
#
# # download mini-imagenet automatically
#
# # from torchvision.utils import download_and_extract_archive
#
# import torchvision.utils as utils
#
# print(utils)
# # print(download_and_extract_archive)
#
# # %%
#
# # torch concat, https://pytorch.org/docs/stable/torch.html#torch.cat
# # Concatenates the given sequence of seq tensors in the given dimension.
# # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
# import torch
#
# g1 = torch.randn(3, 2)
# g2 = torch.randn(4, 2)
#
# g3 = torch.randn(4, 2, 3)
#
# grads = [g1, g2]
# print(g1.view(-1).size())
# print(g2.view(-1).size())
# print(g3.view(-1).size())
# # print(g3.view(-1))
#
# grads = torch.cat(grads, dim=0)
# print(grads)
# print(grads.size())
# print(grads.mean())
# print(grads.std())
#
# # torch stack, https://pytorch.org/docs/stable/torch.html#torch.stack
# # Concatenates sequence of tensors along a new dimension.
# # All tensors need to be of the same size.
# # torch.stack([g1,g2], dim=0)
#
# # %%
#
# import torch
#
# a = torch.tensor([1, 2, 3.], requires_grad=True)
# a_detached = a.detach()
# print(a_detached.is_leaf)
# a_detached_sum = a.sum()
# print(c.is_leaf)
# d = c.detach()
# print(d.is_leaf)
#
# # %%
#
# import torch
#
# from types import SimpleNamespace
# from pathlib import Path
# from pprint import pprint
#
# x = torch.empty([1, 2, 3])
# print(x.size())
#
# args = SimpleNamespace()
# args.data_root = "~/automl-meta-learning/data/miniImagenet"
#
# # n1313361300001299.jpg
# args.data_root = Path(args.data_root).expanduser()
#
# # %%
#
# import torch
#
# CHW = 3, 12, 12
# x = torch.randn(CHW)
# y = torch.randn(CHW)
#
# new = [x, y]
# new = torch.stack(new)
# print(x.size())
# print(new.size())
#
# # %%
#
# print('a');
# print('b')
#
# # %%
#
# # conver list to tensor
#
# import torch
#
# x = torch.tensor([1, 2, 3.])
# print(x)
#
# # %%
#
# from torchvision.transforms import Compose, Resize, ToTensor
#
# import torchmeta
# from torchmeta.datasets.helpers import miniimagenet
#
# from pathlib import Path
# from types import SimpleNamespace
#
# from tqdm import tqdm
#
# ## get args
# args = SimpleNamespace(episodes=5, n_classes=5, k_shot=5, k_eval=15, meta_batch_size=1, n_workers=4)
# args.data_root = Path("~/automl-meta-learning/data/miniImagenet").expanduser()
#
# ## get meta-batch loader
# train_transform = Compose([Resize(84), ToTensor()])
# dataset = miniimagenet(
#     args.data_root,
#     ways=args.n_classes,
#     shots=args.k_shot,
#     test_shots=args.k_eval,
#     meta_split='train',
#     download=False)
# dataloader = torchmeta.utils.data.BatchMetaDataLoader(
#     dataset,
#     batch_size=args.meta_batch_size,
#     num_workers=args.n_workers)
#
# with tqdm(dataset):
#     print(f'len(dataloader)= {len(dataloader)}')
#     for episode, batch in enumerate(dataloader):
#         print(f'episode = {episode}')
#         train_inputs, train_labels = batch["train"]
#         print(f'train_labels[0] = {train_labels[0]}')
#         print(f'train_inputs.size() = {train_inputs.size()}')
#         pass
#         if episode >= args.episodes:
#             break
#
# # %%
#
# # zip tensors
#
# import torch
#
# x = torch.tensor([1., 2., 3.])
# y = torch.tensor([1, 2, 3])
#
# print(list(zip(x, y)))
#
# xx = torch.randn(2, 3, 84, 84)
# yy = torch.randn(2, 3, 32, 32)
#
# print(len(list(zip(xx, yy))))
#
# # %%
#
# x = 2
# print(x)
#
# # %%
#
# ## sinusioid function
# print('Starting Sinusioid cell')
#
# from torchmeta.toy import Sinusoid
# from torchmeta.utils.data import BatchMetaDataLoader
# from torchmeta.transforms import ClassSplitter
#
# # from tqdm import tqdm
#
# batch_size = 16
# shots = 5
# test_shots = 15
# # dataset = torchmeta.toy.helpers.sinusoid(shots=shots, test_shots=tes_shots)
# metaset_dataset = Sinusoid(num_samples_per_task=shots + test_shots, num_tasks=100, noise_std=None)
# splitter_metset_dataset = ClassSplitter(
#     metaset_dataset,
#     num_train_per_class=shots,
#     num_test_per_class=test_shots,
#     shuffle=True)
# dataloader = BatchMetaDataLoader(splitter_metset_dataset, batch_size=batch_size, num_workers=4)
#
# print(f'batch_size = {batch_size}')
# print(f'len(dataset) = {len(metaset_dataset)}')
# print(f'len(dataloader) = {len(dataloader)}\n')
# for batch_idx, batch in enumerate(dataloader):
#     print(f'batch_idx = {batch_idx}')
#     train_inputs, train_targets = batch['train']
#     test_inputs, test_targets = batch['test']
#     print(f'train_inputs.shape = {train_inputs.shape}')
#     print(f'train_targets.shape = {train_targets.shape}')
#     print(f'test_inputs.shape = {test_inputs.shape}')
#     print(f'test_targets.shape = {test_targets.shape}')
#     if batch_idx >= 1:  # halt after 2 iterations
#         break
#
# print('DONE\a')
#
# # %%
#
# ## notes of torchmeta
#
# from pathlib import Path
# import torchmeta
#
# # meta-set: creates collection of data-sets, D_meta = {D_1, ... Dn}
# print('\n-- Sinusoid(MetaDataset)')
# metaset_sinusoid = torchmeta.toy.Sinusoid(num_samples_per_task=10, num_tasks=1_000_000, noise_std=None)
# print(f'type(metaset_sinusoid) = {type(metaset_sinusoid)}')
# print(f'len(metaset_sinusoid) = {len(metaset_sinusoid)}')
# print(f'metaset_sinusoid = {metaset_sinusoid}')
#
# # this is still a data set but helps implement forming D_i
# # i.e. the N-way, K-shot tasks/datasets we need.
# print('\n-- MiniImagenet(CombinationMetaDataset)')
# data_path = Path('~/data').expanduser()
# metaset_miniimagenet = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True,
#                                                        download=True)
# print(f'type(metaset_miniimagenet) = {type(metaset_miniimagenet)}')
# print(f'len(metaset_miniimagenet) = {len(metaset_miniimagenet)}')
# print(f'metaset_miniimagenet = {metaset_miniimagenet}')
#
# # Splits the data-sets inside the meta-set into support/train & query/test sets
# dataset = metaset_miniimagenet
# dataset = torchmeta.transforms.ClassSplitter(dataset, num_train_per_class=1, num_test_per_class=15, shuffle=True)
# print(dataset)
#
# # %%
#
# import torch
# import torch.nn as nn
# import numpy as np
#
# x = np.random.uniform()
#
# x = torch.rand()
#
# print(x)
#
# l = nn.Linear(1, 1)
#
# y = l(x)
#
# print(y)
#
# # %%
#
# # saving tensors for my data set
# import torch
# import torch.nn as nn
#
# from collections import OrderedDict
#
# from pathlib import Path
#
# # N x's of size D=1 in an interval
# Din, Dout = 3, 2
# num_samples = 5
# lb, ub = -1, 1
# X = (ub - lb) * torch.rand([num_samples, Din]) + lb  # rand gives uniform in [0,1) range
#
# # N y's of size D=1 (from output of NN)
# f = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din, Dout)),
#     ('out', nn.SELU())
# ]))
#
# # fill cnn with Gaussian
# mu1, std1 = 5, 7.5
# f.f1.weight.data.normal_(mu1, std1)
# f.f1.bias.data.normal_(mu1, std1)
#
# # get outputs
# Y = f(X)
# print(Y)
#
# # save tensors and cnn
# # https://stackoverflow.com/questions/1466000/difference-between-modes-a-a-w-w-and-r-in-built-in-open-function
# db = {
#     'X': X,
#     'Y': Y
# }
# path = Path(f'~/data/tmp/SinData_mu1{mu1}_std1{std1}/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
# with open(path / 'db', 'w') as file:  # create file and truncate to length 0, only writing allowed
#     torch.save(db, file)
#
# # %%
#
# # saving data in numpy
#
# import numpy as np
# import pickle
# from pathlib import Path
#
# path = Path('~/data/tmp/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
#
# lb, ub = -1, 1
# num_samples = 5
# x = np.random.uniform(low=lb, high=ub, size=(1, num_samples))
# y = x ** 2 + x + 2
#
# # using save (to npy), savez (to npz)
# np.save(path / 'x', x)
# np.save(path / 'y', y)
# np.savez(path / 'db', x=x, y=y)
# with open(path / 'db.pkl', 'wb') as db_file:
#     pickle.dump(obj={'x': x, 'y': y}, file=db_file)
#
# ## using loading npy, npz files
# x_loaded = np.load(path / 'x.npy')
# y_load = np.load(path / 'y.npy')
# db = np.load(path / 'db.npz')
# with open(path / 'db.pkl', 'rb') as db_file:
#     db_pkl = pickle.load(db_file)
#
# print(x is x_loaded)
# print(x == x_loaded)
# print(x == db['x'])
# print(x == db_pkl['x'])
# print('done')
#
# # %%
#
# import numpy as np
# from pathlib import Path
#
# path = Path('~/data/tmp/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
#
# lb, ub = -1, 1
# num_samples = 5
# x = np.random.uniform(low=lb, high=ub, size=(1, num_samples))
# y = x ** 2 + x + 2
#
# np.save(path / 'x', x)
# np.save(path / 'y', y)
#
# x_loaded = np.load(path / 'x.npy')
# y_load = np.load(path / 'y.npy')
#
# print(x is x_loaded)  # False
# print(x == x_loaded)  # [[ True  True  True  True  True]]
#
# # %%
#
# # saving torch tensors
#
# import torch
# import torch.nn as nn
# import torchvision
#
# from pathlib import Path
# from collections import OrderedDict
#
# path = Path('~/data/tmp/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
#
# tensor_a = torch.rand(2, 3)
# tensor_b = torch.rand(1, 3)
#
# db = {'a': tensor_a, 'b': tensor_b}
#
# torch.save(db, path / 'torch_db')
# loaded = torch.load(path / 'torch_db')
# print(loaded['a'] == tensor_a)
# print(loaded['b'] == tensor_b)
#
# # testing if ToTensor() screws things up
# lb, ub = -1, 1
# N, Din, Dout = 3, 1, 1
# x = torch.distributions.Uniform(low=lb, high=ub).sample((N, Din))
# print(x)
#
# f = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din, Dout)),
#     ('out', nn.SELU())
# ]))
# y = f(x)
#
# transform = torchvision.transforms.transforms.ToTensor()
# y_proc = transform(y)
# print(y_proc)
#
# # %%
#
# # union dictionaries, https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python
#
# d1 = {'a': 1, 'b': 2.5}
# d2 = {'b': 2, 'c': 3, 'd': 4}
# d = {**d1, **d2}
# # duplicates resolved in favour of d2
# print(d)
#
# # %%
#
# # generating uniform variables
#
# import numpy as np
#
# num_samples = 3
# Din = 1
# lb, ub = -1, 1
#
# xn = np.random.uniform(low=lb, high=ub, size=(num_samples, Din))
# print(xn)
#
# import torch
#
# sampler = torch.distributions.Uniform(low=lb, high=ub)
# r = sampler.sample((num_samples, Din))
#
# print(r)
#
# r2 = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))
#
# print(r2)
#
# # process input
# f = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din, Dout)),
#     ('out', nn.SELU())
# ]))
# Y = f(r2)
# print(Y)
#
# # %%
#
# # sampling from normal distribution in torch
#
# import torch
#
# num_samples = 3
# Din = 1
# mu, std = 0, 1
# x = torch.distributions.normal.Normal(loc=mu, scale=std).sample((num_samples, Din))
#
# print(x)
#
# # %%
#
# # creating data and running through a nn and saving it
#
# import torch
# import torch.nn as nn
#
# from pathlib import Path
# from collections import OrderedDict
#
# import numpy as np
#
# import pickle
#
# path = Path('~/data/tmp/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
#
# num_samples = 3
# Din, Dout = 1, 1
# lb, ub = -1, 1
#
# x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))
#
# f = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din, Dout)),
#     ('out', nn.SELU())
# ]))
# y = f(x)
#
# # save data torch to numpy
# x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
# np.savez(path / 'db', x=x_np, y=y_np)
#
# print(x_np)
# # save model
# with open('db_saving_seq', 'wb') as file:
#     pickle.dump({'f': f}, file)
#
# # load model
# with open('db_saving_seq', 'rb') as file:
#     db = pickle.load(file)
#     f2 = db['f']
#
# # test that it outputs the right thing
# y2 = f2(x)
#
# y_eq_y2 = y == y2
# print(y_eq_y2)
#
# db2 = {'f': f, 'x': x, 'y': y}
# torch.save(db2, path / 'db_f_x_y')
#
# print('Done')
#
# db3 = torch.load(path / 'db_f_x_y')
# f3 = db3['f']
# x3 = db3['x']
# y3 = db3['y']
# yy3 = f3(x3)
#
# y_eq_y3 = y == y3
# print(y_eq_y3)
#
# y_eq_yy3 = y == yy3
# print(y_eq_yy3)
#
# # %%
#
# # test for saving everything with torch.save
#
# import torch
# import torch.nn as nn
#
# from pathlib import Path
# from collections import OrderedDict
#
# import numpy as np
#
# import pickle
#
# path = Path('~/data/tmp/').expanduser()
# path.mkdir(parents=True, exist_ok=True)
#
# num_samples = 3
# Din, Dout = 1, 1
# lb, ub = -1, 1
#
# x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))
#
# f = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din, Dout)),
#     ('out', nn.SELU())
# ]))
# y = f(x)
#
# # save data torch to numpy
# x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
# db2 = {'f': f, 'x': x_np, 'y': y_np}
# torch.save(db2, path / 'db_f_x_y')
# # np.savetxt(path / 'output.csv', y_np)  # for csv
#
# db3 = torch.load(path / 'db_f_x_y')
# f3 = db3['f']
# x3 = db3['x']
# y3 = db3['y']
# xx = torch.tensor(x3)
# yy3 = f3(xx)
#
# print(yy3)
#
# # %%
#
# # my saving code for synthetic data, nvm using torch.save for everything
#
# # import torch
# # import torch.nn as nn
# #
# # from pathlib import Path
# # from collections import OrderedDict
# #
# # import numpy as np
# #
# # path = Path('~/data/tmp/').expanduser()
# # path.mkdir(parents=True, exist_ok=True)
# #
# # num_samples = 3
# # Din, Dout = 1, 1
# # lb, ub = -1, 1
# #
# # x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))
# #
# # f = nn.Sequential(OrderedDict([
# #     ('f1', nn.Linear(Din,Dout)),
# #     ('out', nn.SELU())
# # ]))
# # y = f(x)
# #
# # # save data torch to numpy
# # x_np, y_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
# # np.savez(path / 'data', x=x_np, y=y_np)
# #
# # # save model
# # torch.save(f,path / 'f')
#
# # %%
#
# import torch
#
# import torch.nn as nn
#
# from collections import OrderedDict
#
# num_samples = 3
# Din, Dout = 1, 1
# lb, ub = -1, 1
#
# x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples, Din))
#
# hidden_dim = [(Din, 20), (20, 20), (20, 20), (20, 20), (20, Dout)]
# f = nn.Sequential(OrderedDict([
#     ('fc1;l1', nn.Linear(hidden_dim[0][0], hidden_dim[0][1])),
#     ('relu2', nn.ReLU()),
#     ('fc2;l1', nn.Linear(hidden_dim[1][0], hidden_dim[1][1])),
#     ('relu2', nn.ReLU()),
#     ('fc3;l1', nn.Linear(hidden_dim[2][0], hidden_dim[2][1])),
#     ('relu3', nn.ReLU()),
#     ('fc4;l1', nn.Linear(hidden_dim[3][0], hidden_dim[3][1])),
#     ('relu4', nn.ReLU()),
#     ('fc5;final;l2', nn.Linear(hidden_dim[4][0], hidden_dim[4][1]))
# ]))
#
# y = f(x)
#
# print(y)
#
# section_label = [1] * 4 + [2]
# print(section_label)
#
# # %%
#
# # get list of paths to task
# # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
# # https://stackoverflow.com/a/44228436/1601580
#
# from pathlib import Path
# from glob import glob
#
# meta_split = 'train'
# data_path = Path('~/data/LS/debug/fully_connected_NN_mu1_1.0_std1_2.5_mu2_1.0_std2_0.5/')
# data_path = (data_path / meta_split).expanduser()
#
# # with path lib
# tasks_folder = [f for f in data_path.iterdir() if f.is_dir()]
#
# assert ('f_avg' not in tasks_folder)
#
# len_folder = len(tasks_folder)
# print(len_folder)
# print(tasks_folder)
# print()
#
# # with glob
# p = str(data_path) + '/*/'
# print(p)
# tasks_folder = glob(p)
#
# assert ('f_avg' not in tasks_folder)
#
# len_folder = len(tasks_folder)
# print(len_folder)
# print(tasks_folder)
# print()
#
# # with glob and negation
# print(set(glob(str(data_path / "f_avg"))))
# tasks_folder = set(glob(str(data_path / '*'))) - set(glob(str(data_path / "f_avg")))
#
# assert ('f_avg' not in tasks_folder)
#
# len_folder = len(tasks_folder)
# print(len_folder)
# print(tasks_folder)
# print()
#
# # %%
#
# # looping through metasets
#
# from torchmeta.utils.data import BatchMetaDataLoader
# from torchmeta.transforms import ClassSplitter
# from torchmeta.toy import Sinusoid
#
# from tqdm import tqdm
#
# # get data set
# dataset = Sinusoid(num_samples_per_task=25, num_tasks=30)
# shots, test_shots = 5, 15
# # get metaset
# metaset = ClassSplitter(
#     dataset,
#     num_train_per_class=shots,
#     num_test_per_class=test_shots,
#     shuffle=True)
# # get meta-dataloader
# batch_size = 16
# num_workers = 0
# meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
# epochs = 2
#
# print(f'batch_size = {batch_size}')
# print(f'len(metaset) = {len(metaset)}')
# print(f'len(meta_dataloader) = {len(meta_dataloader)}')
# with tqdm(range(epochs)) as tepochs:
#     for epoch in tepochs:
#         for batch_idx, batch in enumerate(meta_dataloader):
#             print(f'\nbatch_idx = {batch_idx}')
#             train_inputs, train_targets = batch['train']
#             test_inputs, test_targets = batch['test']
#             print(f'train_inputs.shape = {train_inputs.shape}')
#             print(f'train_targets.shape = {train_targets.shape}')
#             print(f'test_inputs.shape = {test_inputs.shape}')
#             print(f'test_targets.shape = {test_targets.shape}')
#
# # %%
#
# from tqdm import tqdm
#
# import time
#
# with tqdm(range(5)) as trange:
#     for t in trange:
#         print(t)
#         time.sleep(1)
#
# # %%
#
#
# import torch
# import torch.nn as nn
#
# l1 = torch.tensor([1, 2, 3.]) ** 0.5
# l2 = torch.tensor([0, 0, 0.0])
# mse = nn.MSELoss()
# loss = mse(l1, l2)
# print(loss)
#
# # %%
#
# import numpy as np
#
# x = np.arange(0, 10)
# print(x)
#
# print(x.max())
# print(x.min())
# print(x.mean())
# print(np.median(x))
#
# # %%
#
# x = torch.randn(3)
# print(x)
# print(x.argmax(-1))
#
# # %%
#
# # testing accuracy function
# # https://discuss.pytorch.org/t/calculating-accuracy-of-the-current-minibatch/4308/11
# # https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch
#
# import torch
# import torch.nn as nn
#
# D = 1
# true = torch.tensor([0, 1, 0, 1, 1]).reshape(5, 1)
# print(f'true.size() = {true.size()}')
#
# batch_size = true.size(0)
# print(f'batch_size = {batch_size}')
# x = torch.randn(batch_size, D)
# print(f'x = {x}')
# print(f'x.size() = {x.size()}')
#
# mdl = nn.Linear(D, 1)
# logit = mdl(x)
# _, pred = torch.max(logit.data, 1)
#
# print(f'logit = {logit}')
#
# print(f'pred = {pred}')
# print(f'true = {true}')
#
# acc = (true == pred).sum().item()
# print(f'acc = {acc}')
#
# # %%
#
# # https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be
# # dimension
# # https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/4?u=brando_miranda
# """
# Dimension reduction. It collapses/reduces a specific dimension by selecting an element from that dimension to be
# reduced.
# Consider x is 3D tensor. x.sum(1) converts x into a tensor that is 2D using an element from D1 elements in
# the 1th dimension. Thus:
# x.sum(1) = x[i,k] = op(x[i,:,k]) = op(x[i,0,k],...,x[i,D1,k])
# the key is to realize that we need 3 indices to select a single element. So if we use only 2 (because we are collapsing)
# then we have D1 number of elements possible left that those two indices might indicate. So from only 2 indices we get a
# set that we need to specify how to select. This is where the op we are using is used for and selects from this set.
# In theory if we want to collapse many indices we need to indicate how we are going to allow indexing from a smaller set
# of indices (using the remaining set that we'd usually need).
# """
#
# import torch
#
# x = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 6]
# ])
#
# print(f'x.size() = {x.size()}')
#
# # sum the 0th dimension (rows). So we get a bunch of colums that have the rows added together.
# x0 = x.sum(0)
# print(x0)
#
# # sum the 1th dimension (columns)
# x1 = x.sum(1)
# print(x1)
#
# x_1 = x.sum(-1)
# print(x_1)
#
# x0 = x.max(0)
# print(x0.values)
#
# y = torch.tensor([[
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12]],
#
#     [[13, 14, 15, 16],
#      [17, 18, 19, 20],
#      [21, 22, 23, 24]]])
#
# print(y)
#
# # into the screen [1, 13]
# print(y[:, 0, 0])
# # columns [1, 5, 9]
# print(y[0, :, 0])
# # rows [1, 2, 3, 4]
# print(y[0, 0, :])
#
# # for each remaining index, select the largest value in the "screen" dimension
# y0 = y.max(0)
# print(y0.values)
#
#
# # %%
#
# # understanding making label predictions
# # https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649/3?u=brando_miranda
#
# def calc_accuracy(mdl, X, Y):
#     # reduce/collapse the classification dimension according to max op
#     # resulting in most likely label
#     max_vals, max_indices = mdl(X).max(1)
#     # assumes the first dimension is batch size
#     n = max_indices.size(0)  # index 0 for extracting the # of elements
#     # calulate acc (note .item() to do float division)
#     acc = (max_indices == Y).sum().item() / n
#     return acc
#
#
# import torch
# import torch.nn as nn
#
# # data dimension [batch-size, D]
# D, Dout = 1, 5
# batch_size = 16
# x = torch.randn(batch_size, D)
# y = torch.randint(low=0, high=Dout, size=(batch_size,))
#
# mdl = nn.Linear(D, Dout)
# logits = mdl(x)
# print(f'y.size() = {y.size()}')
# # removes the 1th dimension with a max, which is the classification layer
# # which means it returns the most likely label. Also, note you need to choose .indices since you want to return the
# # position of where the most likely label is (not it's raw logit value)
# pred = logits.max(1).indices
# print(pred)
#
# print('--- preds vs truth ---')
# print(f'predictions = {pred}')
# print(f'y = {y}')
#
# acc = (pred == y).sum().item() / pred.size(0)
# print(acc)
# print(calc_accuracy(mdl, x, y))
#
# # %%
#
# # https://discuss.pytorch.org/t/runtimeerror-element-0-of-variables-does-not-require-grad-and-does-not-have-a-grad-fn/11074/20
#
# import torch
# import torch.nn as nn
#
# x = torch.randn(1)
# mdl = nn.Linear(1, 1)
#
# y = mdl(x)
# print(mdl.weight)
#
# print(y)
#
# # %%
#
# # https://discuss.pytorch.org/t/how-to-get-the-module-names-of-nn-sequential/39682
# # looping through modules but get the one with a specific name
#
# import torch
# import torch.nn as nn
#
# from collections import OrderedDict
#
# params = OrderedDict([
#     ('fc0', nn.Linear(in_features=4, out_features=4)),
#     ('ReLU0', nn.ReLU()),
#     ('fc1L:final', nn.Linear(in_features=4, out_features=1))
# ])
# mdl = nn.Sequential(params)
#
# # throws error
# # mdl['fc0']
#
# for m in mdl.children():
#     print(m)
#
# print()
#
# for m in mdl.modules():
#     print(m)
#
# print()
#
# for name, m in mdl.named_modules():
#     print(name)
#     print(m)
#
# print()
#
# for name, m in mdl.named_children():
#     print(name)
#     print(m)
#
# # %%
#
# # apply mdl to x until the final layer, then return the embeding
#
# import torch
# import torch.nn as nn
#
# from collections import OrderedDict
#
# Din, Dout = 1, 1
# H = 10
#
# modules = OrderedDict([
#     ('fc0', nn.Linear(in_features=Din, out_features=H)),
#     ('ReLU0', nn.ReLU()),
#
#     ('fc1', nn.Linear(in_features=H, out_features=H)),
#     ('ReLU1', nn.ReLU()),
#
#     ('fc2', nn.Linear(in_features=H, out_features=H)),
#     ('ReLU2', nn.ReLU()),
#
#     ('fc3', nn.Linear(in_features=H, out_features=H)),
#     ('ReLU3', nn.ReLU()),
#
#     ('fc4L:final', nn.Linear(in_features=H, out_features=Dout))
# ])
#
# mdl = nn.Sequential(modules)
#
# out = x
# for name, m in self.base_model.named_children():
#     if 'final' in name:
#         # return out
#         break
#     out = m(out)
#
# print(out.size())
#
# # %%
#
# # initializing a constant weight net
# # https://discuss.pytorch.org/t/how-to-add-appropriate-noise-to-a-neural-network-with-constant-weights-so-that-back-propagation-training-works/93411
#
# # import torch
#
# # [layer.reset_parameters() for layer in base_model.children() if hasattr(layer, 'reset_parameters')]
#
# # model = nn.Linear(1, 1)
# # model_copy = copy.deepcopy(model)
#
# # %%
#
# print('start')
#
# # f_avg: PLinReg vs MAML
#
# import numpy as np
# from matplotlib import pyplot as plt
# from pathlib import Path
#
# datas_std = [0.1, 0.125, 0.1875, 0.2]
#
# pl = [2.3078539778125768e-07,
#       1.9997889411762922e-07,
#       2.729681222011256e-07,
#       3.2532371115080884e-07]
# pl_stds = [1.4852212316567463e-08,
#            5.090588920661132e-09,
#            1.1424832554909115e-08,
#            5.058656213138166e-08]
#
# maml = [3.309504692539563e-07,
#         4.1058904888091606e-06,
#         6.8326703386053605e-06,
#         7.4616147721799645e-06]
# maml_stds = [4.039131189060566e-08,
#              3.66839089258494e-08,
#              9.20683484136399e-08,
#              9.789292209743077e-08]
#
# # fig = plt.figure()
# fig, ax = plt.subplots(nrows=1, ncols=1)
#
# ax.set_title('MAML vs Pre-Trained embedding with Linear Regression')
#
# x = datas_std
#
# ax.errorbar(x, pl, yerr=pl_stds, label='PLinReg', marker='o')
# ax.errorbar(x, maml, yerr=maml_stds, label='MAML', marker='o')
# ax.plot()
# ax.legend()
#
# ax.set_xlabel('std (of FNN Data set)')
# ax.set_ylabel('meta-test loss (MSE)')
#
# plt.show()
#
# # path = Path('~/ultimate-utils/plot').expanduser()
# # fig.savefig(path)
#
# print('done \a')
# # %%
#
# # Torch-meta miniImagenet
# # loop through meta-batches of this data set, print the size, make sure it's the size you exepct
#
# import torchmeta
# from torchmeta.utils.data import BatchMetaDataLoader
# from torchmeta.transforms import ClassSplitter
# # from torchmeta.toy import Sinusoid
#
# from tqdm import tqdm
#
# # dataset = Sinusoid(num_samples_per_task=100, num_tasks=20)
# dataset = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_train=True, download=True)
# print(f'type(metaset_miniimagenet) = {type(dataset)}')
# print(f'len(metaset_miniimagenet) = {len(dataset)}')
# shots, test_shots = 5, 15
# # get metaset
# metaset = ClassSplitter(
#     dataset,
#     num_train_per_class=shots,
#     num_test_per_class=test_shots,
#     shuffle=True)
# # get meta-dataloader
# batch_size = 16
# num_workers = 0
# meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
# epochs = 2
#
# print(f'batch_size = {batch_size}')
# print(f'len(metaset) = {len(metaset)}')
# print(f'len(meta_dataloader) = {len(meta_dataloader)}\n')
# with tqdm(range(epochs)) as tepochs:
#     for epoch in tepochs:
#         print(f'\n[epoch={epoch}]')
#         for batch_idx, batch in enumerate(meta_dataloader):
#             print(f'batch_idx = {batch_idx}')
#             train_inputs, train_targets = batch['train']
#             test_inputs, test_targets = batch['test']
#             print(f'train_inputs.shape = {train_inputs.shape}')
#             print(f'train_targets.shape = {train_targets.shape}')
#             print(f'test_inputs.shape = {test_inputs.shape}')
#             print(f'test_targets.shape = {test_targets.shape}')
#             print()
#
# # %%
#
# import torch
#
# x = torch.tensor([1., 2, 3])
# print(x.mean())
#
# print(x * x)
# print(x @ x)
# print(x.matmul(x))
#
# # x.mm(x) weird error
#
# # %%
#
# import torch
#
# x = torch.randn(12, 20)
# y = torch.randn(20, 30)
#
# out = x @ y
# print(out.size())
#
# # %%
# # https://www.youtube.com/watch?v=46RjXawJQgg&t=1493s
#
# from pathlib import Path
#
# from pandas import read_csv
#
# read_csv(Path())
#
# # %%
#
# print('hello-world')
# xx = 2
#
# print(xx)
#
# print(' ')
#
# ##
# print('end!')
#
# # %%
#
# # let's see how big the random values from the normal are
#
# import torch
#
# D = 8
# w = torch.tensor([0.1] * D)
# print(f'w.size() = {w.size()}')
# mu = torch.zeros(w.size())
# std = w * 1.5e-2  # two decimal places and a little more
# noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
#
# print('--- noise ')
# print(noise.size())
# print(noise)
#
# w += noise
# print('--- w')
# print(w.size())
# print(w)
#
# # %%
#
# # editing parameters in pytorch in place without error: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/41
#
# import torch
# import torch.nn as nn
# from collections import OrderedDict
#
# Din, Dout = 8, 1
#
# base_model = nn.Sequential(OrderedDict([
#     ('f1', nn.Linear(Din, Dout)),
#     ('out', nn.SELU())
# ]))
#
# with torch.no_grad():
#     for i, w in enumerate(base_model.parameters()):
#         print(f'--- i = {i}')
#         print(w)
#         w += w + 0.001
#         print(w)
#
# # %%
#
# # pickle vs torch.save
#
# # def log_validation(args, meta_learner, outer_opt, meta_val_set):
# #     """ Log the validation loss, acc. Checkpoint the model if that flag is on. """
# #     if args.save_ckpt:  # pickle vs torch.save https://discuss.pytorch.org/t/advantages-disadvantages-of-using-pickle-module-to-save-models-vs-torch-save/79016
# #         # make dir to logs (and ckpts) if not present. Throw no exceptions if it already exists
# #         path_to_ckpt = args.logger.current_logs_path
# #         path_to_ckpt.mkdir(parents=True, exist_ok=True)  # creates parents if not presents. If it already exists that's ok do nothing and don't throw exceptions.
# #         ckpt_path_plus_path = path_to_ckpt / Path('db')
# #
# #         args.base_model = "check the meta_learner field in the checkpoint not in the args field"  # so that we don't save the child model so many times since it's part of the meta-learner
# #         # note this obj has the last episode/outer_i we ran
# #         torch.save({'args': args, 'meta_learner': meta_learner}, ckpt_path_plus_path)
# #     acc_mean, acc_std, loss_mean, loss_std = meta_eval(args, meta_learner, meta_val_set)
# #     if acc_mean > args.best_acc:
# #         args.best_acc, args.loss_of_best = acc_mean, loss_mean
# #         args.logger.loginfo(
# #             f"***> Stats of Best Acc model: meta-val loss: {args.loss_of_best} +- {loss_std}, meta-val acc: {args.best_acc} +- {acc_std}")
# #     return acc_mean, acc_std, loss_mean, loss_std
#
# # %%
#
# import numpy as np
# from sklearn.linear_model import LinearRegression
#
# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# # y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
# reg = LinearRegression()
# print(reg)
# reg = LinearRegression().fit(X, y)
# print(reg)
# reg.score(X, y)
#
# reg.coef_
#
# reg.intercept_
#
# reg.predict(np.array([[3, 5]]))
#
# # %%
#
# # https://stackoverflow.com/questions/63818676/what-is-the-machine-precision-in-pytorch-and-when-should-one-use-doubles
# # https://discuss.pytorch.org/t/how-does-one-start-using-double-without-unexpected-bugs/95715
# # https://discuss.pytorch.org/t/what-is-the-machine-precision-of-pytorch-with-cpus-or-gpus/9384
#
# import torch
#
# x1 = torch.tensor(1e-6)
# x2 = torch.tensor(1e-7)
# x3 = torch.tensor(1e-8)
# x4 = torch.tensor(1e-9)
#
# eps = torch.tensor(1e-11)
#
# print(x1.dtype)
# print(x1)
# print(x1 + eps)
#
# print(x2)
# print(x2 + eps)
#
# print(x3)
# print(x3 + eps)
#
# print(x4)
# print(x4 + eps)
#
# # %%
#
# # python float is a C double
# # NumPy's standard numpy.float is the same (so C double), also numpy.float64.
# # https://www.doc.ic.ac.uk/~eedwards/compsys/float/
# # https://stackoverflow.com/questions/1049722/what-is-2s-complement
# # https://www.cs.cornell.edu/~tomf/notes/cps104/twoscomp.html#whyworks
# # https://stackoverflow.com/questions/7524838/fixed-point-vs-floating-point-number
# # https://en.wikipedia.org/wiki/Single-precision_floating-point_format
# # https://www.cs.cornell.edu/~tomf/notes/cps104/twoscomp.html#whyworks
#
# import torch
#
# xf = torch.tensor(1e-7)
# xd = torch.tensor(1e-7, dtype=torch.double)
# epsf = torch.tensor(1e-11)
#
# print(xf.dtype)
# print(xf)
# print(xf.item())
# print(type(xf.item()))
#
# #
# print('\n> test when a+eps = a')
# print(xf.dtype)
# print(f'xf = {xf}')
# print(f'xf + 1e-7 = {xf + 1e-7}')
# print(f'xf + 1e-11 = {xf + 1e-11}')
# print(f'xf + 1e-8 = {xf + 1e-8}')
# print(f'xf + 1e-16 = {xf + 1e-16}')
# # after seeing the above it seems that there are errors if things are small
#
# print('\n> test when a+eps = a')
# x = torch.tensor(1e-7, dtype=torch.double)
# print(f'xf = {x}')
# print(f'xf + 1e-7 = {x + 1e-7}')
# print(f'xf + 1e-11 = {x + 1e-11}')
# print(f'xf + 1e-8 = {x + 1e-8}')
# print(f'xf + 1e-16 = {x + 1e-16}')
# # using doubles clearly is better but still has some errors
#
# print('\n> test when a+eps = a')
# x = torch.tensor(1e-4)
# print(f'xf = {x}')
# print(f'xf + 1e-7 = {x + 1e-7}')
# print(f'xf + 1e-11 = {x + 1e-11}')
# print(f'xf + 1e-8 = {x + 1e-8}')
# print(f'xf + 1e-16 = {x + 1e-16}')
#
# # %%
#
# # https://pytorch.org/docs/stable/torchvision/models.html
#
# # %%
#
# import torch
#
# print(torch.zeros(2))
# m = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
# x = m.sample()
# print(x)
#
# # m = torch.distributions.MultivariateNormal(torch.zeros(1, 3), torch.eye(1, 3))
# # mu = m.sample()
# # print(mu)
#
# m = torch.distributions.MultivariateNormal(torch.zeros(1, 5), torch.eye(5))
# y = m.sample()
# print(y)
#
# # %%
#
# from pathlib import Path
# from matplotlib import pyplot as plt
#
# import numpy as np
#
# path = Path('~/data/').expanduser()
#
# # x = np.linspace(0, 2*np.pi, 50)
# x = np.random.uniform(0, 2 * np.pi, 100)
# noise = np.random.normal(0.0, 0.05, 100)
# print(noise)
# y = np.sin(x) + noise
# plt.figure()
# plt.scatter(x, y)
# plt.ylabel('f(x)')
# plt.ylabel('x (raw feature)')
# plt.savefig(path / 'test_fig.pdf')
# plt.savefig(path / 'test_fig.png')
# plt.show()
#
# # %%
#
# from socket import gethostname
# from email.message import EmailMessage
# import smtplib
#
# server = smtplib.SMTP('smtp.gmail.com', 587)
# server.starttls()
# print(server)
#
# # %%
#
# # MTA (Mail Transfer Agent)
# # https://stackoverflow.com/questions/784201/is-there-a-python-mta-mail-transfer-agent
# # https://www.quora.com/How-does-one-send-e-mails-from-Python-using-MTA-Mail-Transfer-Agent-rather-than-an-SMTP-library
# # https://www.reddit.com/r/learnpython/comments/ixlq81/how_does_one_send_emails_from_python_using_mta/
#
# # Q why can't I just send an email directly?
# # Q why do smtp libraries exist
#
# # %%
#
# import smtplib
#
# server = smtplib.SMTP('smtp.intel-research.net', 25)
# server.starttls()
# print(server)
#
#
# # %%
#
# # from socket import gethostname
# # from email.message import EmailMessage
# # import smtplib
# #
# # server = smtplib.SMTP('smtp.gmail.com', 587)
# # server.starttls()
# # # not a real email account nor password, its all ok!
# # server.login('slurm.miranda@gmail.com', 'dummy123!@#$321')
# #
# # # craft message
# # msg = EmailMessage()
# #
# # message = f'{message}\nSend from Hostname: {gethostname()}'
# # msg.set_content(message)
# # msg['Subject'] = subject
# # msg['From'] = 'slurm.miranda@gmail.com'
# # msg['To'] = destination
# # # send msg
# # server.send_message(msg)
#
# # %%
#
# # send email with smtp intel
#
# def send_email(message):
#     from socket import gethostname
#     import smtplib
#     hostname = gethostname()
#     from_address = 'slurm.miranda@gmail.com'
#     from_address = 'miranda9@intel-research.net.'
#     # to_address = [ 'iam-alert@intel-research.net']
#     to_address = ['brando.science@gmail.com']
#     subject = f"Test msg from: {hostname}"
#     ##
#     message = f'Test msg from {hostname}: {message}'
#     full_message = f'From: {from_address}\n' \
#                    f'To: {to_address}\n' \
#                    f'Subject: {subject}\n' \
#                    f'{message}'
#     server = smtplib.SMTP('smtp.intel-research.net')
#     server.sendmail(from_address, to_address, full_message)
#     server.quit()
#     # sys.exit(1)
#
#
# print('start')
# send_email('HelloWorld')
# print('done email test!')
#
#
# # %%
#
# def send_email2(message):
#     from email.mime.multipart import MIMEMultipart
#     from email.mime.text import MIMEText
#     from socket import gethostname
#     import smtplib
#     server = smtplib.SMTP('smtp.intel-research.net')
#     # craft message
#     msg = MIMEMultipart()
#
#     message = f'{message}\nSend from Hostname: {gethostname()}'
#     msg['Subject'] = 'Test email'
#     msg['From'] = 'miranda9@intel-research.net.'
#     msg['To'] = 'brando.science@gmail.com'
#     msg.attach(MIMEText(message, "plain"))
#     # send message
#     server.send_message(msg)
#     # server.sendmail(from_address, to_address, full_message)
#     server.quit()
#
#
# print('start')
# send_email2('HelloWorld')
# print('done email test!')
#
# #%%
#
# from pathlib import Path
#
# message = 'HelloWorld'
# path_to_pdf = Path('~/data/test_fig.pdf').expanduser()
#
# from email.mime.application import MIMEApplication
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from socket import gethostname
# import smtplib
#
# server = smtplib.SMTP('smtp.intel-research.net')
# # craft message
# msg = MIMEMultipart()
#
# message = f'{message}\nSend from Hostname: {gethostname()}'
# msg['Subject'] = 'Test email'
# msg['From'] = 'miranda9@intel-research.net.'
# msg['To'] = 'brando.science@gmail.com'
# msg.attach(MIMEText(message, "plain"))
# # attach pdf
# if path_to_pdf.exists():
#     with open(path_to_pdf, "rb") as f:
#         # attach = email.mime.application.MIMEApplication(f.read(),_subtype="pdf")
#         attach = MIMEApplication(f.read(), _subtype="pdf")
#     attach.add_header('Content-Disposition', 'attachment', filename=str(path_to_pdf))
#     msg.attach(attach)
#
# # send message
# server.send_message(msg)
# # server.sendmail(from_address, to_address, full_message)
# server.quit()
#
# #%%
#
# # Here, we used "w" letter in our argument, which indicates write and will create a file if it does not exist in library
# # Plus sign indicates both read and write.
#
# # with open('data.json', 'w+') as f:
# #     json.dump(self.stats, f)
#
# #%%
#
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter  # https://deeplizard.com/learn/video/psexxmdrufm
#
# path = Path('~/data/logs/').expanduser()
# tb = SummaryWriter(log_dir=path)
# # tb = SummaryWriter(log_dir=args.current_logs_path)
#
# for i in range(3):
#     loss = i + np.random.normal(loc=0, scale=1)
#     tb.add_scalar('loss', loss, i)
#
# # %%
#
# # https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
# # Saving & Loading Model for Inference
# # Save/Load state_dict (Recommended)
# # Save:
# # torch.save(model.state_dict(), PATH)
# #
# # # Load:
# # model = TheModelClass(*args, **kwargs)
# # model.load_state_dict(torch.load(PATH))
# # model.eval()
#
# # %%
#
# # Save:
# # torch.save({
# #             'epoch': epoch,
# #             'model_state_dict': model.state_dict(),
# #             'optimizer_state_dict': optimizer.state_dict(),
# #             'loss': loss,
# #               ...
# #             }, PATH)
# # # Load:
# # model = TheModelClass(*args, **kwargs)
# # optimizer = TheOptimizerClass(*args, **kwargs)
# #
# # checkpoint = torch.load(PATH)
# # model.load_state_dict(checkpoint['model_state_dict'])
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # epoch = checkpoint['epoch']
# # loss = checkpoint['loss']
# #
# # model.eval()
# # # - or -
# # model.train()
#
# # %%
#
# # https://discuss.pytorch.org/t/how-does-load-a-sequential-model-from-a-string/97648
# # https://stackoverflow.com/questions/64109883/how-does-one-load-a-sequential-model-from-a-string-in-pytorch
#
# # %%
#
# torch.save({'f': f,
#             'f_state_dict': f.state_dict(),
#             'f_str': str(f),
#             'f_modules': f._modules,
#             'f_modules_str': str(f._modules)
#             }, path2avg_f)
#
# #%%
#
# from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
#
# path = Path('~/data/tb_test/').expanduser()
# # path = Path('~/logs/logs_Sep29_12-38-08_jobid_-1/tb').expanduser()
# writer = SummaryWriter(path)
#
# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
#
# print('done! \a')
#
# #%%
#
# db = torch.load(str(args.resume_ckpt_path))
# # args.epchs = db['epoch']  # we can start counting from zero
# # args.epoch += 1  # this is needed so that it starts on the next batch since it says the last batch it *did* and range counts with 0 indexing.
# # meta_learner = db['meta_learner']
# args.base_model = db['f']
# # in case loading directly doesn't work
# modules = eval(db['f_modules_str'])
# args.base_model = torch.nn.Sequential(modules)
# f_state_dict = db['f_state_dict']
# args.base_model.load_state_dict(f_state_dict)
#
#
# #%%
#
# # Torch-meta miniImagenet
#
# import torchmeta
# from torchmeta.utils.data import BatchMetaDataLoader
# from torchmeta.transforms import ClassSplitter
#
# from pathlib import Path
#
# from tqdm import tqdm
#
# data_path = Path('~/data/').expanduser()
# meta_split = 'train'
# dataset = torchmeta.datasets.MiniImagenet(data_path, num_classes_per_task=5, meta_split=meta_split, download=True)
# # dataset = torchmeta.datasets.Omniglot(data_path, num_classes_per_task=5, meta_split=meta_split, download=True)
#
# print(f'type(metaset_miniimagenet) = {type(dataset)}')
# print(f'len(metaset_miniimagenet) = {len(dataset)}')
# shots, test_shots = 5, 15
# metaset = ClassSplitter(
#     dataset,
#     num_train_per_class=shots,
#     num_test_per_class=test_shots,
#     shuffle=True)
# batch_size = 16
# num_workers = 0
# meta_dataloader = BatchMetaDataLoader(metaset, batch_size=batch_size, num_workers=num_workers)
# epochs = 2
#
# print(f'batch_size = {batch_size}')
# print(f'len(metaset) = {len(metaset)}')
# print(f'len(meta_dataloader) = {len(meta_dataloader)}\n')
# with tqdm(range(epochs)) as tepochs:
#     for epoch in tepochs:
#         print(f'\n[epoch={epoch}]')
#         for batch_idx, batch in enumerate(meta_dataloader):
#             print(f'batch_idx = {batch_idx}')
#             train_inputs, train_targets = batch['train']
#             test_inputs, test_targets = batch['test']
#             print(f'train_inputs.shape = {train_inputs.shape}')
#             print(f'train_targets.shape = {train_targets.shape}')
#             print(f'test_inputs.shape = {test_inputs.shape}')
#             print(f'test_targets.shape = {test_targets.shape}')
#             print()
#             break
#         break
#
# #%%
#
# from torchmeta.datasets.helpers import omniglot
# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import BatchMetaDataLoader
#
# from pathlib import Path
#
# meta_split = 'train'
# data_path = Path('~/data/').expanduser()
# dataset = omniglot(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
#
# for batch in dataloader:
#     train_inputs, train_targets = batch["train"]
#     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
#
#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
#
# #%%
#
# # replacing a module in in a pytorch model
# # https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/11
#
# import torch
#
# from torchmeta.datasets.helpers import omniglot
# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import BatchMetaDataLoader
#
# from pathlib import Path
#
# import copy
#
# meta_split = 'train'
# data_path = Path('~/data/').expanduser()
# dataset = omniglot(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
#
#
#
# def replace_bn(module, name):
#     """
#     Recursively put desired batch norm in nn.module module.
#
#     set module = net to start code.
#     """
#     # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
#     for attr_str in dir(module):
#         target_attr = getattr(module, attr_str)
#         if type(target_attr) == torch.nn.BatchNorm2d:
#             new_bn = torch.nn.BatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum, target_attr.affine,
#                                           track_running_stats=False)
#             setattr(module, attr_str, new_bn)
#
#     # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
#     for name, immediate_child_module in module.named_children():
#         replace_bn(immediate_child_module, name)
#
# def convert_bn(model):
#     for module in model.modules():
#         if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#             module.__init__(module.num_features, module.eps,
#                             module.momentum, module.affine,
#                             track_running_stats=False)
#
# fc_out_features = 5
#
# # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
# # replace_bn(model, 'model')
# # model.fc = torch.nn.Linear(in_features=512, out_features=fc_out_features, bias=True)
# #
# # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
# # replace_bn(model, 'model')
# # model.fc = torch.nn.Linear(in_features=2048, out_features=fc_out_features, bias=True)
#
# # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=False)
# # replace_bn(model, 'model')
# # model.fc = torch.nn.Linear(in_features=2048, out_features=fc_out_features, bias=True)
#
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=False)
# replace_bn(model, 'model')
# model.fc = torch.nn.Linear(in_features=2048, out_features=fc_out_features, bias=True)
#
# for batch in dataloader:
#     train_inputs, train_targets = batch["train"]
#     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
#     first_meta_batch = train_inputs[0]  # task
#     nk_task = first_meta_batch
#     out = model(nk_task)
#     print(f'resnet out.size(): {out.size()}')
#     break
#
# print('success\a')
#
# # %%
#
# import torch
#
# import torchvision.transforms as transforms
#
# # import torchmeta
# # from torchmeta.datasets.helpers import omniglot
# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import BatchMetaDataLoader
#
# from pathlib import Path
#
# meta_split = 'train'
# data_path = Path('~/data/').expanduser()
#
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# data_augmentation_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(84),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.2),
#     transforms.ToTensor(),
#     normalize])
# dataset = miniimagenet(data_path,
#                        transform=data_augmentation_transforms,
#                        ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
#
# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
#
# print(len(dataloader))
# # for batch_idx, batch in enumerate(dataloader):
# #     print(f'--> batch_idx = {batch_idx}')
# #     train_inputs, train_targets = batch["train"]
# #     print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
# #     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
# #     test_inputs, test_targets = batch["test"]
# #     print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
# #     print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
# #     first_meta_batch = train_inputs[0]  # task
# #     nk_task = first_meta_batch
# #     out = model(nk_task)
# #     print(f'resnet out.size(): {out.size()}')
# #     break
#
# print('success\a')
#
# #%%
#
# import torch
#
# import torchvision.transforms as transforms
#
# # import torchmeta
# # from torchmeta.datasets.helpers import omniglot
# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import BatchMetaDataLoader
#
# from pathlib import Path
#
# meta_split = 'train'
# data_path = Path('~/data/').expanduser()
#
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# data_augmentation_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(84),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.2),
#     transforms.ToTensor(),
#     normalize])
# dataset = miniimagenet(data_path,
#                        transform=data_augmentation_transforms,
#                        ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
# print(f'len augmented = {len(dataloader)}')
#
# dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
# print(f'len normal = {len(dataloader)}')
#
# print('success\a')
#
# #%%
#
# import torch
#
# import torchvision.transforms as transforms
#
# from torchmeta.datasets.helpers import miniimagenet
# from torchmeta.utils.data import BatchMetaDataLoader
#
# from tqdm import tqdm
#
# from pathlib import Path
#
# meta_split = 'train'
# data_path = Path('~/data/').expanduser()
#
# # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # data_augmentation_transforms = transforms.Compose([
# #     transforms.RandomResizedCrop(84),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ColorJitter(
# #         brightness=0.4,
# #         contrast=0.4,
# #         saturation=0.4,
# #         hue=0.2),
# #     transforms.ToTensor(),
# #     normalize])
# # dataset = miniimagenet(data_path,
# #                        transform=data_augmentation_transforms,
# #                        ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# # dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
# # print(f'len augmented = {len(dataloader)}')
#
# dataset = miniimagenet(data_path, ways=5, shots=5, test_shots=15, meta_split=meta_split, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
# print(f'len normal = {len(dataloader)}')
#
# num_batches = 10
# with tqdm(dataloader, total=num_batches) as pbar:
#     for batch_idx, batch in enumerate(pbar):
#         train_inputs, train_targets = batch["train"]
#         print(train_inputs.size())
#         # print(batch_idx)
#         if batch_idx >= num_batches:
#             break
#
# print('success\a')
#
# #%%
#
# from math import comb
#
# total_classes = 64
# n = 5
# number_tasks = comb(total_classes, n)
# print(number_tasks)
#
# #%%
#
# # saving a json file save json file
# # human readable pretty print https://stackoverflow.com/questions/12943819/how-to-prettyprint-a-json-file
#
# import json
#
# data = 'data string'
# with open('data.txt', 'w') as outfile:
#     json.dump(data, outfile)
#
# # json.dump(data, open('data.txt', 'w'))
#
# # with open(current_logs_path / 'experiment_stats.json', 'w+') as f:
# #     json.dump(self.stats, f)
# # data_ars = {key:value for (key,value) in dictonary.items()}
# # x = {key:str(value) for (key,value) in args.__dict__.items()}
#
# with open(args.current_logs_path / 'args.json', 'w+') as argsfile:
#     args_data = {key: str(value) for (key, value) in args.__dict__.items()}
#     json.dump(args_data, argsfile, indent=4)
#
# #%%
#
# # get gpu model as string: https://stackoverflow.com/questions/64526139/how-does-one-get-the-model-of-the-gpu-in-python-and-save-it-as-a-string
#
# #%%
#
#
# image = PILI.open(self.images[idx]).convert('RGB')r (key,value) in args.__dict__.items()}

with open(args.current_logs_path / 'args.json', 'w+') as argsfile:
    args_data = {key: str(value) for (key, value) in args.__dict__.items()}
    json.dump(args_data, argsfile, indent=4)

# %%

# get gpu model as string: https://stackoverflow.com/questions/64526139/how-does-one-get-the-model-of-the-gpu-in-python-and-save-it-as-a-string


# %%

mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
# [0.47214064400000005, 0.45330829125490196, 0.4099612805098039]
std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

mean2 = [0.485, 0.456, 0.406]
std2 = [0.229, 0.224, 0.225]

print(mean)
print(mean2)
print(mean == mean2)

print(std)
print(std2)
print(std == std2)

# %%

# references:
# https://stackoverflow.com/questions/20486700/why-we-always-divide-rgb-values-by-255

import numpy as np

from PIL import Image

import glob

import torchvision.transforms as transforms

from pathlib import Path
import os

transform = transforms.Compose([transforms.ToTensor()])

# get image meta-lstm
split = 'val'
root = Path(f'~/data/miniimagenet_meta_lstm/miniImagenet/{split}').expanduser()

labels = sorted(os.listdir(root))
images = [glob.glob(os.path.join(root, label, '*')) for label in labels]

label_idx = 0
img_idx = 0
img = Image.open(images[img_idx][img_idx])  # .convert('RGB')

# check image as 0-255
a = np.asarray(img)  # 0-255 image range
print(a)
img = Image.fromarray(a)  # from array to img object
print(img)
a2 = np.asarray(a)
print((a == a2).all())

# converts image object or 0-255 image to
img = transform(img)
print(img)

# rfs
# img = np.asarray(self.imgs[item]).astype('uint8')

# meta-lstm
# images = [glob.glob(os.path.join(root, label, '*')) for label in self.labels]
# image = PILI.open(self.images[idx]).convert('RGB')

# %%

from tqdm import tqdm

train_iters = 2
with tqdm(range(train_iters), total=train_iters) as pbar_epochs:
    print(range(train_iters))
    print(list(range(train_iters)))
    for epoch in pbar_epochs:
        print(epoch)

# %%


## sinusioid function
print('Starting Sinusioid cell')

import torchmeta
# from torchmeta.toy import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

# from torchmeta.transforms import ClassSplitter

# from tqdm import tqdm

batch_size = 16
shots = 5
test_shots = 15
dataset = torchmeta.toy.helpers.sinusoid(shots=shots, test_shots=test_shots)
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, num_workers=4)

# print(f'batch_size = {batch_size}')
# print(f'len(dataloader) = {len(dataloader)}\n')
# for batch_idx, batch in enumerate(dataloader):
#     print(f'batch_idx = {batch_idx}')
#     train_inputs, train_targets = batch['train']
#     test_inputs, test_targets = batch['test']
#     print(f'train_inputs.shape = {train_inputs.shape}')
#     print(f'train_targets.shape = {train_targets.shape}')
#     print(f'test_inputs.shape = {test_inputs.shape}')
#     print(f'test_targets.shape = {test_targets.shape}')
#     if batch_idx >= 1:  # halt after 2 iterations
#         break

# two tasks are different
dl = enumerate(dataloader)

_, x1 = next(dl)
x1, _ = x1['train']
print(f'x1 = {x1.sum()}')
_, x2 = next(dl)
x2, _ = x2['train']
print(f'x2 = {x2.sum()}')

assert (x1.sum() != x2.sum())
print('assert pass, tasks have different data')

# same task twice
dl = enumerate(dataloader)

_, x1 = next(dl)
x1, _ = x1['train']
print(f'x1 = {x1.sum()}')
dl = enumerate(dataloader)
_, x2 = next(dl)
x2, _ = x2['train']
print(f'x2 = {x2.sum()}')

assert (x1.sum() == x2.sum())

print('DONE\a')

# %%

# https://github.com/tristandeleu/pytorch-meta/issues/69

from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

batch_size = 16
shots = 5
test_shots = 15

# Seed the dataset with `seed = 0`
dataset = sinusoid(shots=shots, test_shots=test_shots, seed=0)
# `num_workers = 0` to avoid stochasticity of multiple processes
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

batch = next(iter(dataloader))

inputs, _ = batch['train']
print(f'Sum of inputs: {inputs.sum()}')

# %%

# https://github.com/tristandeleu/pytorch-meta/issues/69

from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader


def random_hash():
    return random.randrange(1 << 32)


batch_size = 16
shots = 5
test_shots = 15

# Seed the dataset with `seed = 0`
dataset = sinusoid(shots=shots, test_shots=test_shots, seed=0)
dataset.__hash__ = random_hash
# `num_workers = 0` to avoid stochasticity of multiple processes
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

batch = next(iter(dataloader))

inputs, _ = batch['train']
print(f'Sum of inputs: {inputs.sum()}')

# %%

# https://github.com/tristandeleu/pytorch-meta/issues/69

from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader

batch_size = 16
shots = 5
test_shots = 15

dataset = sinusoid(shots=shots, test_shots=test_shots)
# `num_workers = 0` to avoid stochasticity of multiple processes
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4)

batch = next(iter(dataloader))

inputs, _ = batch['train']
print(f'Sum of inputs: {inputs.sum()}')

# %%

from pathlib import Path

import torch

path = '/home/miranda9/data/dataset_LS_fully_connected_NN_with_BN/meta_set_fully_connected_NN_with_BN_std1_8.0_std2_1.0_noise_std0.1/train/fi_fully_connected_NN_with_BN_norm_f_151.97657775878906'
path = Path(path).expanduser() / 'fi_db.pt'
path = str(path)

# db = torch.load(path)
# torch.jit.load(path)
db = torch.jit.load(str(path))

print(db)

# %%

import torch
import torch.nn as nn

from collections import OrderedDict

fc0 = nn.Linear(in_features=1, out_features=1)
params = [('fc0', fc0)]
mdl = nn.Sequential(OrderedDict(params))

x = torch.tensor([1.0])
y = mdl(x)

print(y)


# %%

# secs per it to days

def sect_per_it_2_days(secs_per_it, total_its):
    days = (secs_per_it * total_its) / (60 * 60 * 24)
    print(days)


print(f'time in days for resnet18_rfs with 1 inner steps')
sect_per_it_2_days(4.76, 100000)

print(f'time in days for resnet18_rfs with 1 inner steps')
sect_per_it_2_days(8.19, 100000)

print(f'time in days for resnet18_rfs with 4 inner steps')
sect_per_it_2_days(16.11, 100000)

print(f'time in days for synthetic with 1 inner steps')
sect_per_it_2_days(46.26, 100000)

print(f'time in days for synthetic with 1 inner steps')
sect_per_it_2_days(3.47, 100000)

print(f'time in days for synthetic with 1 inner steps')
sect_per_it_2_days(2.7, 100000)

print(f'time in days for synthetic with 1 inner steps')
sect_per_it_2_days(5.7, 100000)

print(f'time in days for synthetic with 1 inner steps')
sect_per_it_2_days(46.26, 20000)

print(f'time in days for synthetic with 1 inner steps')
sect_per_it_2_days(2.7, 20_000)

# %%

import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

from pathlib import Path

# get init
path_2_init = Path('~/data/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()
ckpt = torch.load(path_2_init)
mdl = ckpt['f']

#
Din, Dout = 1, 1
mdl = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
mdl2 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))

#
hook1 = SimilarityHook(mdl, "fc1_l1")
hook2 = SimilarityHook(mdl2, "fc1_l1")
mdl.eval()
mdl2.eval()

#
num_samples_per_task = 100
lb, ub = -1, 1
x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
with torch.no_grad():
    mdl(x)
    mdl2(x)
hook1.distance(hook2, size=8)

# %%


import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

from pathlib import Path

# get init
path_2_init = Path('~/data/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()
ckpt = torch.load(path_2_init)
mdl = ckpt['f']

#
Din, Dout = 1, 1
mdl = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
# with torch.no_grad():
#     mdl.fc1_l1.weight.fill_(2.0)
#     mdl.fc1_l1.bias.fill_(2.0)

#
hook1 = SimilarityHook(mdl, "fc1_l1")
hook2 = SimilarityHook(mdl, "fc1_l1")
mdl.eval()

# params for doing "good" CCA
iters = 10
num_samples_per_task = 100
size = 8
# start CCA comparision
lb, ub = -1, 1
with torch.no_grad():
    for _ in range(iters):
        x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
        mdl(x)
hook1.distance(hook2, size=size)

# %%

import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

from pathlib import Path

# get init
# path_2_init = Path('~/data/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()
# ckpt = torch.load(path_2_init)
# mdl = ckpt['f']

#
Din, Dout = 1, 1
mdl1 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU()),
    ('fc2_l2', nn.Linear(Din, Dout)),
]))
mdl2 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU()),
    ('fc2_l2', nn.Linear(Din, Dout)),
]))
with torch.no_grad():
    mu = torch.zeros(Din)
    # std =  1.25e-2
    std = 10
    noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
    # mdl2.fc1_l1.weight.fill_(50.0)
    # mdl2.fc1_l1.bias.fill_(50.0)
    mdl2.fc1_l1.weight += noise
    mdl2.fc1_l1.bias += noise

#
hook1 = SimilarityHook(mdl1, "fc2_l2")
hook2 = SimilarityHook(mdl2, "fc2_l2")
mdl1.eval()
mdl2.eval()

# params for doing "good" CCA
iters = 10
num_samples_per_task = 500
size = 8
# start CCA comparision
lb, ub = -1, 1
with torch.no_grad():
    for _ in range(iters):
        x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
        y1 = mdl1(x)
        y2 = mdl2(x)
        print((y1 - y2).norm(2))
dist = hook1.distance(hook2, size=size)
print(f'dist={dist}')

# %%

a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica", "Vicky")

lst = zip(a, b)
lst = list(lst)
print(lst)

# %%

lst = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
lst = zip(*lst)
lst = list(lst)
print(lst)

import numpy as np

average_per_layer = [np.average(l) for l in lst]
average_total = np.average(average_per_layer)
print(average_per_layer)
print(average_total)

# %%

import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

from pathlib import Path

import copy

# get init
path_2_init = Path('~/data/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()
ckpt = torch.load(path_2_init)
mdl = ckpt['f']
mdl1 = mdl
# mdl2 = copy.deepcopy(mdl1)
mdl2 = copy.deepcopy(mdl)

#
Din, Dout = 1, 1
# mdl1 = nn.Sequential(OrderedDict([
#     ('fc1_l1', nn.Linear(in_features=1, out_features=300, bias=True)),
#     ('bn1_l1', nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#     ('relu1', nn.ReLU()),
#     ('fc2_l1', nn.Linear(in_features=300, out_features=300, bias=True)),
#     ('bn2_l1', nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#     ('relu2', nn.ReLU()),
#     ('fc3_l1', nn.Linear(in_features=300, out_features=300, bias=True)),
#     ('bn3_l1', nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#     ('relu3', nn.ReLU()),
#     ('fc4_final_l2', nn.Linear(in_features=300, out_features=1, bias=True))
# ]))
# mdl2 = nn.Sequential(OrderedDict([
#     ('fc1_l1', nn.Linear(in_features=1, out_features=300, bias=True)),
#     ('bn1_l1', nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#     ('relu1', nn.ReLU()),
#     ('fc2_l1', nn.Linear(in_features=300, out_features=300, bias=True)),
#     ('bn2_l1', nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#     ('relu2', nn.ReLU()),
#     ('fc3_l1', nn.Linear(in_features=300, out_features=300, bias=True)),
#     ('bn3_l1', nn.BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)),
#     ('relu3', nn.ReLU()),
#     ('fc4_final_l2', nn.Linear(in_features=300, out_features=1, bias=True))
# ]))

# with torch.no_grad():
#     mu = torch.zeros(Din)
#     # std =  1.25e-2
#     std = 10
#     noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
#     # mdl2.fc1_l1.weight.fill_(50.0)
#     # mdl2.fc1_l1.bias.fill_(50.0)
#     mdl2.fc1_l1.weight += noise
#     mdl2.fc1_l1.bias += noise

#
# hook1 = SimilarityHook(mdl1, "fc1_l1")
# hook2 = SimilarityHook(mdl2, "fc1_l1")
hook1 = SimilarityHook(mdl1, "fc2_l1")
hook2 = SimilarityHook(mdl2, "fc2_l1")
mdl1.eval()
mdl2.eval()

# params for doing "good" CCA
iters = 10
num_samples_per_task = 500
size = 8
# start CCA comparision
lb, ub = -1, 1
# with torch.no_grad():
#     for _ in range(iters):
#         # x = torch.torch.distributions.Uniform(low=-1, high=1).sample((15, 1))
#         x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
#         y1 = mdl1(x)
#         y2 = mdl2(x)
#         print((y1-y2).norm(2))
for _ in range(iters):
    x = torch.torch.distributions.Uniform(low=-1, high=1).sample((15, 1))
    # x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
    y1 = mdl1(x)
    y2 = mdl2(x)
    print((y1 - y2).norm(2))
dist = hook1.distance(hook2, size=size)
print(f'dist={dist}')

# %%

from sklearn.metrics import explained_variance_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
explained_variance_score(y_true, y_pred)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
ev = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
ev_raw = explained_variance_score(y_true, y_pred, multioutput='raw_values')
ev_weighted = explained_variance_score(y_true, y_pred, multioutput='variance_weighted')

print(ev_raw)
print(ev_weighted)

# %%
# import sklearn.metrics.mean_squared_error as mse, not possible because is a funciton is my guess?
# https://stackoverflow.com/questions/40823418/why-cant-i-import-from-a-module-alias
from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred, squared=False)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
mean_squared_error(y_true, y_pred)

mean_squared_error(y_true, y_pred, squared=False)

mean_squared_error(y_true, y_pred, multioutput='raw_values')

mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])

# %%


import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

from pathlib import Path

import copy

#
Din, Dout = 1, 1
mdl1 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU()),
    ('fc2_l2', nn.Linear(Din, Dout)),
]))
mdl2 = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU()),
    ('fc2_l2', nn.Linear(Din, Dout)),
]))

if torch.cuda.is_available():
    mdl1 = mdl1.cuda()
    mdl2 = mdl2.cuda()

with torch.no_grad():
    mu = torch.zeros(Din)
    # std =  1.25e-2
    std = 10
    noise = torch.distributions.normal.Normal(loc=mu, scale=std).sample()
    # mdl2.fc1_l1.weight.fill_(50.0)
    # mdl2.fc1_l1.bias.fill_(50.0)
    mdl2.fc1_l1.weight += noise
    mdl2.fc1_l1.bias += noise
hook1 = SimilarityHook(mdl1, "fc2_l1")
hook2 = SimilarityHook(mdl2, "fc2_l1")
mdl1.eval()
mdl2.eval()

# params for doing "good" CCA
iters = 10
num_samples_per_task = 500
size = 8
# start CCA comparision
lb, ub = -1, 1

for _ in range(iters):
    x = torch.torch.distributions.Uniform(low=-1, high=1).sample((15, 1))
    if torch.cuda.is_available():
        x = x.cuda()
    # x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
    y1 = mdl1(x)
    y2 = mdl2(x)
    print((y1 - y2).norm(2))
dist = hook1.distance(hook2, size=size)
print(f'dist={dist}')

# %%

# other cca library for layer https://discuss.pytorch.org/t/what-is-a-good-cca-cka-library-for-pytorch-that-works-ideally-with-gpu/104889
# https://github.com/jameschapman19/cca_zoo


# %%

# walrus operator
# https://therenegadecoder.com/code/the-controversy-behind-the-walrus-operator-in-python/


(x := 1)
print(x)

#
from pathlib import Path

path = Path('~/data/coq-hott-dataset-serpi/contrib/HoTTBook.feat').expanduser()
with open(path) as f:
    while line := f.read():
        print(line)

#
# [result for x in values if (result := func(x)) < 10]
#
# if result := do_something():
#     do_more(result)

[y := f(x), y ** 2, y ** 3]

# %%

from lark import Lark

#
grammar = '''start: WORD "," WORD "!"

            %import common.WORD   // imports from terminal library
            %ignore " "           // Disregard spaces in text
         '''
# grates parser
l = Lark(grammar)

print(l.parse("Hello, World!"))

# %%

from lark import Lark
import lark

grammar = """
start: term

term: apply
    | const
    | free
    | var
    | bound
    | abs
apply: "(apply " term " " term ")"
const: "(const " MYSTR  ")"
free: "(free " MYSTR ")"
var: "(var " MYSTR ")"
bound: "(bound " MYSTR ")"
abs: "(abs " MYSTR " " term ")"
MYSTR: LETTER (LETTER | "." | "_" | DIGIT)*

%import common.WORD
%import common.DIGIT
%import common.LETTER
%ignore " "
"""
parser = Lark(grammar)
tree1 = parser.parse(
    "(apply (const HOL.Trueprop) (apply (apply (const HOL.implies) (apply (apply (const HOL.conj) (free A)) (free B))) (apply (apply (const HOL.conj) (free B)) (free A))))")
print(parser.parse(
    "(apply (const HOL.Trueprop) (apply (apply (const HOL.implies) (apply (apply (const HOL.conj) (free A)) (free B))) (apply (apply (const HOL.conj) (free B)) (free A))))"))
print(tree1.pretty())


class IncreaseAllNumbers(lark.Transformer):
    def _call_userfunc(self, tree, children):
        # to do I will need to do something to get the type of variables
        # because the variables' types are not attached yet
        return

    def _call_userfunc_token(self, c):
        print(c)


IncreaseAllNumbers(visit_tokens=True).transform(tree1)

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
print(type(embeds))
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

# %%

my_set = {1, 2, 3}
print(my_set)
print(type(my_set))
print(len(my_set))
print(1 in my_set)

print(my_set.pop())
print(my_set.pop())
print(my_set.pop())

# %%

from collections import defaultdict

lst = [('a', 1), ('b', 2), ('a', 0)]
# collect values for a in one place, collect values for b in another
d = defaultdict(list)  # creates a default dict of value to empty list
for k, v in lst:
    d[k].append(v)

print(d)
print(dict(d))
lst2 = d.items()
print(sorted(lst2))

# %%
import numpy as np

x = np.random.randn(1, 10)
xx = np.array([x, x]).reshape([])

print(xx.shape)

c = np.cov(xx)

# %%

import numpy as np

x = np.random.randn(2, 10)

print(x.shape)

c = np.cov(x)

print(c)
print(c.shape)

# %%

import torch
import torch.nn as nn

from collections import OrderedDict

params = OrderedDict([
    ('fc1', nn.Linear(in_features=4, out_features=4)),
    ('ReLU1', nn.ReLU()),
    ('fc2', nn.Linear(in_features=4, out_features=4)),
    ('ReLU2', nn.ReLU()),
    ('fc3', nn.Linear(in_features=4, out_features=1)),
])
mdl = nn.Sequential(params)

for name, m in mdl.named_children():
    print(f'{name}, {m}')

print()

# for m in mdl.modules():
#     print(m)
#
# print()
#
# for name, m in mdl.named_modules():
#     print(name)
#     print(m)
#
# print()
#
# for name, m in mdl.named_children():
#     print(name)
#     print(m)

# %%

# Meaning of dimension in pytorch operations: https://discuss.pytorch.org/t/whats-different-between-dim-1-and-dim-0/61094/5

# input tensor of dimensions B x C, B = number of batches, C = number of classes.
B = 8
C = 3
inputs = torch.rand(size=(B, C))
soft_dim0 = torch.softmax(inputs, dim=0)
soft_dim1 = torch.softmax(inputs, dim=1)
print('**** INPUTS ****')
print(inputs)
print(inputs.size())
print('**** SOFTMAX DIM=0 ****')
print(soft_dim0)
print(f'soft_dim0[0, :].sum()={soft_dim0[0, :].sum()}')
print(f'soft_dim0[:, 0].sum()={soft_dim0[:, 0].sum()}')
print(soft_dim0.size())
# print('**** SOFTMAX DIM=1 ****')
# print(soft_dim1)


# %%

# cosine similarity

import torch.nn as nn

dim = 1  # apply cosine accross the second dimension/feature dimension
cos = nn.CosineSimilarity(dim=dim)  # eps defaults to 1e-8 for numerical stability

k = 4  # number of examples
d = 8  # dimension
x1 = torch.randn(k, d)
x2 = x1 * 3
print(f'x1 = {x1.size()}')
cos_similarity_tensor = cos(x1, x2)
print(cos_similarity_tensor)
print(cos_similarity_tensor.size())

# %%

import torch.nn as nn


def ned(x1, x2, dim=1, eps=1e-8):
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


def nes(x1, x2, dim=1, eps=1e-8):
    return 1 - ned(x1, x2, dim, eps)


dim = 1  # apply cosine accross the second dimension/feature dimension

k = 4  # number of examples
d = 8  # dimension of feature space
x1 = torch.randn(k, d)
x2 = x1 * 3
print(f'x1 = {x1.size()}')
ned_tensor = ned(x1, x2, dim=dim)
print(ned_tensor)
print(ned_tensor.size())
print(nes(x1, x2, dim=dim))

# %%

import torch

# trying to convert a list of tensors to a torch.tensor

x = torch.randn(3, 1)
xs = [x, x]
# xs = torch.tensor(xs)
xs = torch.as_tensor(xs)

# %%

import torch

# trying to convert a list of tensors to a torch.tensor

x = torch.randn(4)
xs = [x.numpy(), x.numpy()]
# xs = torch.tensor(xs)
xs = torch.as_tensor(xs)

print(xs)
print(xs.size())

# %%

import torch

# trying to convert a list of tensors to a torch.tensor

x = torch.randn(4)
xs = [x.numpy(), x.numpy(), x.numpy()]
xs = [xs, xs]
# xs = torch.tensor(xs)
xs = torch.as_tensor(xs)

print(xs)
print(xs.size())

# %%

# You could use torch.cat or torch.stack to create a tensor from the list.

import torch

x = torch.randn(4)
xs = [x, x]
xs = torch.cat(xs)
print(xs.size())
# xs = torch.stack(xs)
# print(xs.size())


# %%

import torch

# stack vs cat

# cat "extends" a list in the given dimension e.g. adds more rows or columns

x = torch.randn(2, 3)
print(f'{x.size()}')

# add more rows (thus increasing the dimensionality of the column space to 2 -> 6)
xnew_from_cat = torch.cat((x, x, x), 0)
print(f'{xnew_from_cat.size()}')

# add more columns (thus increasing the dimensionality of the row space to 3 -> 9)
xnew_from_cat = torch.cat((x, x, x), 1)
print(f'{xnew_from_cat.size()}')

print()

# stack serves the same role as append in lists. i.e. it doesn't change the original
# vector space but instead adds a new index to the new tensor, so you retain the ability
# get the original tensor you added to the list by indexing in the new dimension
xnew_from_stack = torch.stack((x, x, x, x), 0)
print(f'{xnew_from_stack.size()}')

xnew_from_stack = torch.stack((x, x, x, x), 1)
print(f'{xnew_from_stack.size()}')

xnew_from_stack = torch.stack((x, x, x, x), 2)
print(f'{xnew_from_stack.size()}')

# default appends at the from
xnew_from_stack = torch.stack((x, x, x, x))
print(f'{xnew_from_stack.size()}')

print('I like to think of xnew_from_stack as a \"tensor list\" that you can pop from the front')

print()

lst = []
print(f'{x.size()}')
for i in range(10):
    x += i  # say we do something with x at iteration i
    lst.append(x)
# lstt = torch.stack([x for _ in range(10)])
lstt = torch.stack(lst)
print(lstt.size())

print()

# lst = []
# print(f'{x.size()}')
# for i in range(10):
#     x += i  # say we do something with x at iteration i
#     for j in range(11):
#         x += j
#         lstx
#     lst.append(x)
# # lstt = torch.stack([x for _ in range(10)])
# lstt = torch.stack(lst)
# print(lstt.size())

# %%

import torch


# A class that represents an individual node in a
# Binary Tree
class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val


# A function to do postorder tree traversal
def print_postorder(root):
    # don't do anything if root is Nothing, else traverse according to PostOrder traversal
    # (i.e. left & right until done then print)
    if root:  # if it's None it's False so does nothing, it's true if it's not None
        # First post order print left child (if None this does nothing and then does Post order of the right)
        print_postorder(root.left)

        # Once right has been done for current node do Post order of right tree (if root.right is None do nothing)
        print_postorder(root.right)

        # After everything has been printed in post order way, then you can now print the data of current node
        print(root.val)


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

print("\nPostorder traversal of binary tree is")
print_postorder(root)


# %%

class Node:
    """Node class for general trees"""

    def __init__(self, val):
        self.children = []
        self.val = val  # value of current node

    def forward(self, children_embeddings):
        # just do a sum of children and current value
        return self.val + sum(children_embeddings)


# create top
root = Node(1)
# create left
left = Node(2)
left.children = [Node(4), Node(5)]
# create right
right = Node(3)
# create entire tree
root.children = [left, right]


# A function to do postorder tree traversal
def compute_embedding_bottom_up(root, verbose=False):
    '''
    What we want is to compute all subtrees
    @param root:
    @return:
    '''
    # don't do anything if root is Nothing, else traverse according to PostOrder traversal
    if root:  # if it's None it's False so does nothing, it's true if it's not None
        # Traverse the entire childrens in post order before continuing
        children_embedings = []
        for children in root.children:
            child_embeding = compute_embedding_bottom_up(children, verbose)
            children_embedings.append(child_embeding)
        # After everything has been computed in post order, compute the current embedding
        root_embedding = root.forward(children_embedings)
        print(root_embedding) if verbose else None
        return root_embedding


# should print 4 5 11 3 15
compute_embedding_bottom_up(root, verbose=True)


# %%

class Node:
    """Node class for general trees"""

    def __init__(self, val):
        self.children = []
        self.val = val  # value of current node

    def forward(self, children_embeddings):
        # just do a sum of children and current value
        return self.val + sum(children_embeddings)

    term = {
        "App": [
            {
                "Ind": [
                    "Coq.Relations.Relation_Operators.clos_refl_trans",
                    "0"
                ]
            },
            {
                "Var": [
                    "A"
                ]
            },
            {
                "Var": [
                    "R"
                ]
            }
        ]
    }


def embed():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    torch.manual_seed(1)
    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    print(type(embeds))
    lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)


# %%

import torch

x = torch.randn(5, 1)

print(x.size())

xs = torch.stack([x, x, x])

print(xs)
print(xs.size())
mean_xs = xs.mean(dim=0)

print(mean_xs)

# %%

'''
Need:
- 1 vocabulary of green terms
- 2 vocabulary of black terms (coq/gallina constructs)
- 3 ast trees so we can traverse them (postorder ideally)
- 4 traversal code for generating a embedding using tree_nn
'''
# import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils import data

class TreeNN(torch.nn.Module):
    def __init__(self, vocab, embed_dim, constructors):
        """
            vocab = [idx:word]
        """
        super().__init__()
        # artity 0 are embeddings/vectors
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.vocab_2_idx = {word: idx for idx, word in enumerate(vocab)}  # e.g. {"hello": 0, "world": 1}
        self.embeds = nn.Embedding(len(self.vocab), embed_dim)  # V words in vocab, D size embedding
        # arity k are FNN
        self.constructors = constructors
        self.cons_2_fnn = {}
        for cons in self.constructors:
            fnn = self.get_cons_fnn()
            self.cons_2_fnn[cons] = fnn

    def forward(self, asts):
        """compute embeddings bottom up, so all the children of the ast have to be computed first"""
        # ast = asts[0]
        # embeds = [self.compute_embedding_bottom_up(ast) for ast in asts]
        # return embeds
        ast = asts
        return self.compute_embedding_bottom_up(ast)

    def compute_embedding_bottom_up(self, ast):
        children_embeddings = []
        for child in ast.children:
            if child in self.vocab:
                lookup_tensor = torch.tensor([self.vocab_2_idx[child]], dtype=torch.long)
                child_embed = self.embeds(lookup_tensor)
            else:
                child_embed = self.compute_embedding_bottom_up(child)
            children_embeddings.append(child_embed)
        embed = torch.stack(children_embeddings, dim=0).mean(dim=0)
        cons_fnn = self.cons_2_fnn[ast.val]
        return cons_fnn(embed)

    def get_cons_fnn(self):
        # TODO improve, check if arity is variable or fixed, what NN to choose?
        fnn = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)),
            ('SeLU0', nn.SELU()),
            ('fc1', nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim))
        ]))
        return fnn


class Node:
    """Node class for general trees"""

    def __init__(self, val):
        self.children = []
        self.val = val  # value of current node

    def __repr__(self):
        self.print_post_order()
        return ''

    def print_post_order(self):
        """print all the children first then the current node last"""
        for child in self.children:
            if type(child) is str:
                print(child)
            else:
                child.print_post_order()
        print(self.val)


class JsonToAst:
    def __init__(self):
        self.base_cases = {"Ind", "Var"}

    def generate_ast(self, term):
        '''
        Assumption is that at term is of the form:
            term = {
                cons: [...,term,...]
            }

            base case:
            term = {
                cons: [...,string,...]
            }
        '''
        for cons, args in term.items():
            root = Node(cons)
            if cons in self.base_cases:
                args = args[0]  # TODO ask lasse what to do here
                root.children = [args]
            else:
                for term in args:
                    child = self.generate_ast(term)
                    root.children.append(child)
        return root


####

def test():
    json2ast = JsonToAst()
    term = {
        "App": [
            {
                "Ind": [
                    "Coq.Relations.Relation_Operators.clos_refl_trans",
                    "0"
                ]
            },
            {
                "Var": [
                    "A"
                ]
            },
            {
                "Var": [
                    "R"
                ]
            }
        ]
    }
    ast = json2ast.generate_ast(term)
    print(ast)
    #
    vocab = ["R", "A", "Coq.Relations.Relation_Operators.clos_refl_trans"]
    constructors = ["App", "Ind", "Var"]
    #
    embed_dim = 4
    term_encoder = TreeNN(vocab, embed_dim, constructors)
    term_embedding = term_encoder(ast)
    print(term_embedding)
    print(term_embedding.size())


if __name__ == '__main__':
    test()
    print('done\a')

# %%

import torch

x = torch.randn(4, 3, 2)
xs = torch.cat([x, x, x], dim=0)
print(xs.size())
xs = torch.cat([x, x, x], dim=1)
print(xs.size())
xs = torch.cat([x, x, x], dim=2)
print(xs.size())

# %%
term = {
    "App": [
        {
            "Ind": [
                "Coq.Relations.Relation_Operators.clos_refl_trans",
                "0"
            ]
        },
        {
            "Var": [
                "A"
            ]
        },
        {
            "Var": [
                "R"
            ]
        }
    ]
}

print(term.keys())
keys = list(term.keys())
print(keys[0])

# %%

# python conditional ternery operator

x = 'true' if True else 'false'

# %%

import torch

x = torch.randn([5, 12])
print(x.mean())
print(x.mean().size())

y = torch.tensor(x)

print(y.size())

# %%

# https://discuss.pytorch.org/t/identity-element-for-stack-operator-torch-stack-emtpty-x-x-empty-tensor-exists/111459

import torch

empty = torch.tensor([])
x = torch.randn(3, 5, 7)

print(torch.cat([empty, x], dim=0).size())
print(torch.stack([empty, x], dim=0).size())

# %%

import torch

x = torch.randn(5, 4)
for layer in range(x.size(1)):
    print(f'{layer=}')

# %%

# selecting indices arbitrarily i.e. x[*,indicies,*] were * denotes that the rest of the layers are kept the same

# but for only the last 3 layers [T, L] -> [1]
x = torch.randn(5, 4)
# compute average of first 3 layer

L = x.size(1)
indices = torch.tensor(range(L - 1))
xx = x.index_select(dim=1, index=indices)
print(f'{x=}')
print(f'{xx=}')
print(xx.size())

# %%

import torch


def ned_torch(x1, x2, dim=1, eps=1e-4):
    """
    Normalized eucledian distance in pytorch.

    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753

    :param x1:
    :param x2:
    :param dim:
    :param eps:
    :return:
    """
    ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5


out1 = torch.tensor([[-3.6291e-01],
                     [-1.7674e+00],
                     [-2.1817e+00],
                     [-2.0127e+00],
                     [-1.6210e+00],
                     [-7.1149e-01],
                     [-8.0512e-01],
                     [-3.3430e-01],
                     [-6.6400e-01],
                     [-8.5222e-01],
                     [-1.1699e+00],
                     [-8.9726e-01],
                     [-7.2273e-02],
                     [-4.6621e-01],
                     [-1.7938e+00],
                     [-2.1175e+00],
                     [-1.2470e+00],
                     [-1.5756e-01],
                     [-6.4363e-01],
                     [-6.0576e-01],
                     [-1.6676e+00],
                     [-1.9971e+00],
                     [-5.9432e-01],
                     [-3.4780e-01],
                     [-6.0348e-01],
                     [-1.7820e+00],
                     [-2.2057e-01],
                     [-3.8268e-02],
                     [-1.5633e+00],
                     [-3.5840e-01],
                     [-5.7379e-02],
                     [-2.5210e-01],
                     [-1.9601e+00],
                     [-3.7318e-01],
                     [1.2341e-02],
                     [-2.2946e+00],
                     [-5.3198e-01],
                     [-2.3140e+00],
                     [-1.6823e+00],
                     [-4.7436e-01],
                     [-2.6047e-01],
                     [-2.1642e+00],
                     [-4.7522e-01],
                     [-5.7305e-01],
                     [2.8821e-01],
                     [-2.7846e-01],
                     [-2.5561e-01],
                     [-2.2448e+00],
                     [-1.1109e-02],
                     [-1.6171e+00],
                     [-2.3253e+00],
                     [-1.8158e+00],
                     [-1.5101e+00],
                     [1.1949e-01],
                     [-1.2281e+00],
                     [-4.2565e-01],
                     [-1.0244e+00],
                     [-2.0581e+00],
                     [-1.0552e+00],
                     [2.5954e-01],
                     [2.7600e-01],
                     [-1.2441e+00],
                     [2.5143e-01],
                     [-1.9237e+00],
                     [-2.0799e+00],
                     [-2.0188e+00],
                     [-1.2017e-01],
                     [-2.0858e+00],
                     [-1.4656e+00],
                     [-2.4549e-01],
                     [-2.3728e+00],
                     [-8.0225e-01],
                     [-4.2496e-01],
                     [-8.0095e-01],
                     [4.3450e-01],
                     [3.3060e-01],
                     [-2.1804e+00],
                     [-1.8725e+00],
                     [-1.2165e+00],
                     [-1.9400e+00],
                     [-2.2042e+00],
                     [-1.8880e+00],
                     [-1.2850e+00],
                     [1.2322e-01],
                     [-4.6162e-01],
                     [-8.0890e-01],
                     [-7.8389e-01],
                     [-2.1397e+00],
                     [4.1263e-01],
                     [-2.2107e+00],
                     [2.4144e-01],
                     [-3.8620e-01],
                     [-2.1676e+00],
                     [3.2484e-02],
                     [-1.6298e+00],
                     [-1.6220e+00],
                     [-1.3770e+00],
                     [-2.1185e+00],
                     [-1.1192e+00],
                     [-1.3630e+00],
                     [-4.5632e-01],
                     [-1.8549e+00],
                     [3.4460e-01],
                     [-2.3489e-01],
                     [-2.1207e+00],
                     [-7.0951e-01],
                     [2.8363e-01],
                     [-1.1481e+00],
                     [-5.5500e-01],
                     [-1.9301e+00],
                     [-1.2247e+00],
                     [-5.3754e-01],
                     [-5.6930e-01],
                     [2.5710e-01],
                     [-1.5921e+00],
                     [2.5347e-01],
                     [1.0652e-01],
                     [-1.1256e+00],
                     [-1.4893e+00],
                     [4.2699e-01],
                     [-9.1180e-01],
                     [-9.7470e-01],
                     [-1.1939e+00],
                     [3.5195e-01],
                     [-2.1075e+00],
                     [-1.5541e-01],
                     [-2.3053e+00],
                     [-2.2581e+00],
                     [-1.4817e+00],
                     [-4.7145e-01],
                     [1.5247e-01],
                     [7.7248e-02],
                     [-2.1716e+00],
                     [-4.0977e-01],
                     [-7.6577e-01],
                     [2.2840e-01],
                     [-1.9727e+00],
                     [-1.6670e+00],
                     [-1.7057e+00],
                     [-2.3080e+00],
                     [-4.0681e-01],
                     [1.0423e-03],
                     [-1.5651e+00],
                     [-5.2567e-01],
                     [-1.3016e+00],
                     [-1.6186e+00],
                     [-1.5546e+00],
                     [-1.7983e+00],
                     [1.1193e-01],
                     [-1.0648e+00]])
out2 = torch.tensor([[-0.2625],
                     [0.5472],
                     [0.7860],
                     [0.6886],
                     [0.4628],
                     [-0.0615],
                     [-0.0075],
                     [-0.2790],
                     [-0.0889],
                     [0.0196],
                     [0.2027],
                     [0.0456],
                     [-0.4300],
                     [-0.2029],
                     [0.5624],
                     [0.7491],
                     [0.2472],
                     [-0.3808],
                     [-0.1006],
                     [-0.1225],
                     [0.4897],
                     [0.6796],
                     [-0.1291],
                     [-0.2712],
                     [-0.1238],
                     [0.5556],
                     [-0.3445],
                     [-0.4496],
                     [0.4295],
                     [-0.2651],
                     [-0.4386],
                     [-0.3263],
                     [0.6583],
                     [-0.2565],
                     [-0.4788],
                     [0.8512],
                     [-0.1650],
                     [0.8623],
                     [0.4981],
                     [-0.1982],
                     [-0.3215],
                     [0.7760],
                     [-0.1977],
                     [-0.1413],
                     [-0.6378],
                     [-0.3111],
                     [-0.3243],
                     [0.8224],
                     [-0.4653],
                     [0.4606],
                     [0.8688],
                     [0.5751],
                     [0.3989],
                     [-0.5406],
                     [0.2363],
                     [-0.2263],
                     [0.1189],
                     [0.7148],
                     [0.1367],
                     [-0.6213],
                     [-0.6308],
                     [0.2456],
                     [-0.6166],
                     [0.6373],
                     [0.7274],
                     [0.6922],
                     [-0.4024],
                     [0.7307],
                     [0.3732],
                     [-0.3302],
                     [0.8962],
                     [-0.0092],
                     [-0.2267],
                     [-0.0099],
                     [-0.7222],
                     [-0.6623],
                     [0.7853],
                     [0.6078],
                     [0.2296],
                     [0.6467],
                     [0.7990],
                     [0.6167],
                     [0.2691],
                     [-0.5427],
                     [-0.2056],
                     [-0.0054],
                     [-0.0198],
                     [0.7618],
                     [-0.7096],
                     [0.8028],
                     [-0.6109],
                     [-0.2490],
                     [0.7779],
                     [-0.4904],
                     [0.4679],
                     [0.4634],
                     [0.3221],
                     [0.7496],
                     [0.1735],
                     [0.3141],
                     [-0.2086],
                     [0.5977],
                     [-0.6703],
                     [-0.3363],
                     [0.7509],
                     [-0.0627],
                     [-0.6352],
                     [0.1902],
                     [-0.1517],
                     [0.6410],
                     [0.2344],
                     [-0.1618],
                     [-0.1435],
                     [-0.6199],
                     [0.4461],
                     [-0.6178],
                     [-0.5331],
                     [0.1772],
                     [0.3869],
                     [-0.7178],
                     [0.0540],
                     [0.0902],
                     [0.2166],
                     [-0.6746],
                     [0.7433],
                     [-0.3821],
                     [0.8573],
                     [0.8301],
                     [0.3825],
                     [-0.1999],
                     [-0.5596],
                     [-0.5162],
                     [0.7803],
                     [-0.2355],
                     [-0.0302],
                     [-0.6034],
                     [0.6656],
                     [0.4893],
                     [0.5117],
                     [0.8589],
                     [-0.2372],
                     [-0.4723],
                     [0.4306],
                     [-0.1686],
                     [0.2787],
                     [0.4614],
                     [0.4245],
                     [0.5650],
                     [-0.5362],
                     [0.1421]])

x1 = out1
x2 = out2

print(x1.isnan().any())
print(x2.isnan().any())

dim = 1
eps = 1e-4
diff = (x1 - x2).var(dim=dim)
print(diff.isnan().any())
ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
ned = ned_2 ** 0.5

print(ned)

# conclusion, if you only have 1 number calling .var will result in nan since 1 number doesn't have a variance.

# %%

import torch

x = torch.randn(5, 4)
print(x.isnan().any())

# %%

from argparse import Namespace

# The 2 initial objects
options_foo = Namespace(foo="foo")
options_bar = Namespace(bar="bar")

# The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
print(vars(options_foo))

# the merged object
options_baz = Namespace(**vars(options_foo), **vars(options_bar))

print(options_baz)

# %%

import torch

x = torch.randn(5, 4)

print(x.var(dim=1).size() == torch.Size([5]))


# %%

# pretty printing dictionaries and dics of tensors

def _to_json_dict_with_strings(dictionary):
    """
    Convert dict to dict with leafs only being strings. So it recursively makes keys to strings
    if they are not dictionaries.

    Use case:
        - saving dictionary of tensors (convert the tensors to strins!)
        - saving arguments from script (e.g. argparse) for it to be pretty

    e.g.

    """
    if type(dictionary) != dict:
        return str(dictionary)
    d = {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}
    return d


def to_json(dic):
    import types
    import argparse

    if type(dic) is dict:
        dic = dict(dic)
    else:
        dic = dic.__dict__
    return _to_json_dict_with_strings(dic)


def save_to_json_pretty(dic, path, mode='w', indent=4, sort_keys=True):
    import json

    with open(path, mode) as f:
        json.dump(to_json(dic), f, indent=indent, sort_keys=sort_keys)


def pprint_dic(dic):
    """
    This pretty prints a json

    @param dic:
    @return:

    Note: this is not the same as pprint.
    """
    import json

    # make all keys strings recursively with their naitve str function
    dic = to_json(dic)
    # pretty print
    # pretty_dic = json.dumps(dic, indent=4, sort_keys=True)
    # print(pretty_dic)
    print(json.dumps(dic, indent=4, sort_keys=True))  # only this one works...idk why
    # return pretty_dic


def pprint_namespace(ns):
    """ pretty prints a namespace """
    pprint_dic(ns)


import torch
# import json  # results in non serializabe errors for torch.Tensors
from pprint import pprint

dic = {'x': torch.randn(1, 3), 'rec': {'y': torch.randn(1, 3)}}

pprint_dic(dic)
pprint(dic)

# %%

import torch
import torch.nn as nn
from anatome import SimilarityHook

from collections import OrderedDict

from pathlib import Path

# get init
path_2_init = Path('~/data/logs/logs_Nov17_13-57-11_jobid_416472.iam-pbs/ckpt_file.pt').expanduser()
ckpt = torch.load(path_2_init)
mdl = ckpt['f']

#
Din, Dout = 100, 1
mdl = nn.Sequential(OrderedDict([
    ('fc1_l1', nn.Linear(Din, Dout)),
    ('out', nn.SELU())
]))
# with torch.no_grad():
#     mdl.fc1_l1.weight.fill_(2.0)
#     mdl.fc1_l1.bias.fill_(2.0)

#
hook1 = SimilarityHook(mdl, "fc1_l1")
hook2 = SimilarityHook(mdl, "fc1_l1")
mdl.eval()

# params for doing "good" CCA
iters = 10
num_samples_per_task = 100
# size = 8
# start CCA comparision
lb, ub = -1, 1
with torch.no_grad():
    for _ in range(iters):
        x = torch.torch.distributions.Uniform(low=lb, high=ub).sample((num_samples_per_task, Din))
        mdl(x)
d1 = hook1.distance(hook2)
d2 = hook1.distance(hook2, size=4)
d3 = hook1.distance(hook2, size=None)

print(f'{d1=}')
print(f'{d2=}')
print(f'{d3=}')

# %%

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn

from sklearn.model_selection import train_test_split

# overall split 85:10:5

X = list(range(100))
y = list(range(len(X)))

# first do 85:15 then do 2:1 for val split
# its ok to set it to False since its ok to shuffle but then allow reproducibility with random_state
# shuffle = False  # shufflebool, default=True, Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
random_state = 1  # Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
test_size = 0.15
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
print(len(X_train))
print(len(X_val_test))

# then 2/3 for val, 1/3 for test to get 10:5 split
test_size = 1.0 / 3.0
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size, random_state=random_state)
print(len(X_val))
print(len(X_test))

# %%
# %%

"""
global interpreter lock
The mechanism used by the CPython (the cononical implementation of the Python PL)
interpreter to assure that only one thread executes Python bytecode at a time.

However, some extension modules, either standard or third-party,
are designed so as to release the GIL when doing computationally-intensive
tasks such as compression or hashing. Also, the GIL is always released when doing I/O.

Past efforts to create a “free-threaded” interpreter
(one which locks shared data at a much finer granularity)
have not been successful because performance suffered in the
common single-processor case. It is believed that overcoming this performance
issue would make the implementation much more complicated
and therefore costlier to maintain.

According to this post multiprocessing library is the right library to use (and not asyncio)
https://leimao.github.io/blog/Python-Concurrency-High-Level/

nice basic python mp tutorial: https://docs.python.org/3/library/multiprocessing.html

TODO:
    - spawn vs fork: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.run
"""

# The multiprocessing package offers both local and remote concurrency,
# effectively side-stepping the Global Interpreter Lock by using subprocesses instead of threads.
# Due to this, the multiprocessing module allows the programmer
# to fully leverage multiple processors on a given machine.
# It runs on both Unix and Windows.
# from multiprocessing.context import Process
import os

import time
from multiprocessing import Process
from multiprocessing import Pool


# Ex1: compute a map in parallel

def f(x):
    # time.sleep(1)
    return x * x


def main1():
    with Pool(5) as pool:
        print(pool.map(f, [1, 2, 3, 4, 5]))


# Ex2: example of start and join

def f2(name):
    print('hello', name)


def main2():
    p = Process(target=f2, args=('bob',))
    p.start()
    p.join()


# Ex3: example of halting the line like in go and then continuing after everyone is done

def f3(arg):
    print('--- Inside process ---')
    print(f'args to f3 is {arg}!')
    print('parent process:', os.getppid())
    pid = os.getpid()
    print(f'process started with pid={pid}')
    time.sleep(1)
    print(f'--- process done with pid={pid}')
    print('--- Inside process ---')


def main3():
    """
    Example of how to wait incorrectly (it will not work since it will start a process but not
    start the next until the current one is done)
    :return:
    """
    print(f'main process pid {os.getpid()}')
    num_processes = 4
    processes = [Process(target=f3, args=('arg!',)) for _ in range(num_processes)]
    for p in processes:
        print()
        print(p)
        p.start()
        print(f'starting from the main process (pid={os.getpid()}) process with pid {p.pid}')
        p.join()  # wrong!
    print('main 3 done')


def main4():
    """
    Example of how to wait correctly, it blocks for all processes but calls p.start() on all of them first
    :return:
    """
    print(f'main process pid {os.getpid()}')
    num_processes = 4
    processes = [Process(target=f3, args=('arg!',)) for _ in range(num_processes)]
    for p in processes:
        print()
        print(p)
        p.start()
        print(f'starting from the main process (pid={os.getpid()}) process with pid {p.pid}')
    # wait group! call join on all processes and block until they are all done
    for p in processes:
        p.join()
    print('main 4 done')


# Ex5: wait group implementation (i.e. block until all process declare they are done)

def heavy_compute(args, secs=1):
    time.sleep(secs)


def serial_code_blocking_wrong():
    """
    Example of how to wait incorrectly (it will not work since it will start a process but not
    start the next until the current one is done)
    :return:
    """
    num_processes = 4
    processes = [Process(target=heavy_compute, args=('arg!',)) for _ in range(num_processes)]
    for p in processes:
        p.start()
        p.join()  # wrong!


def parallel_code_blocking_correctly():
    """
    Example of how to wait incorrectly (it will not work since it will start a process but not
    start the next until the current one is done)
    :return:
    """
    num_processes = 4
    processes = [Process(target=heavy_compute, args=('arg!',)) for _ in range(num_processes)]
    for p in processes:
        p.start()
    # wait group! call join on all processes and block until they are all done
    for p in processes:
        p.join()


def main5():
    start = time.time()
    serial_code_blocking_wrong()
    print(f'serial (wrong) execution time = {time.time() - start}')
    start = time.time()
    parallel_code_blocking_correctly()
    print(f'parallel execution time = {time.time() - start}')
    # first should be 4 secs second should 1 second


if __name__ == '__main__':
    start = time.time()
    # main1()
    # main2()
    # main3()
    # main4()
    main5()
    print(f'total execution time = {time.time() - start}')
    print('Done with __main__!\a\n')

# %%

"""
Goal: train in a mp way by computing each example in a seperate process.


tutorial: https://pytorch.org/docs/stable/notes/multiprocessing.html
full example: https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py

Things to figure out:
- fork or spwan for us? see pytorch but see this too https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.run
- shared memory
- do we need num_workers=0, 1 or 2? (one for main thread other for pre-fetching batches)
- run test and check that the 112 process do improve the time for a loop (add progress part for dataloder

docs: https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing

(original python mp, they are compatible: https://docs.python.org/3/library/multiprocessing.html)
"""
# from datetime import time
#
# import torch
# from torch.multiprocessing import Pool
#
#
# def train(cpu_parallel=True):
#     num_cpus = get_num_cpus()  # 112 is the plan for intel's clsuter as an arparse or function
#     model.shared_memory()  # TODO do we need this?
#     # add progressbar for data loader to check if multiprocessing is helping
#     for batch_idx, batch in dataloader:
#         # do this mellow with pool when cpu_parallel=True
#         with Pool(num_cpus) as pool:
#             losses = pool.map(target=model.forward, args=batch)
#             loss = torch.sum(losses)
#             # now do .step as normal
#
#
# if __name__ == '__main__':
#     start = time.time()
#     train()
#     print(f'execution time: {time.time() - start}')

# %%

import torch

print(torch.multiprocessing.get_all_sharing_strategies())
print(torch.multiprocessing.get_sharing_strategy())

torch.multiprocessing.set_sharing_strategy('file_system')

# %%

# getting the id of the process wrt to the pooling: https://stackoverflow.com/questions/10190981/get-a-unique-id-for-worker-in-python-multiprocessing-pool

import multiprocessing


def f(x):
    print(multiprocessing.current_process())
    return x * x


p = multiprocessing.Pool()
print(p.map(f, range(6)))

# %%

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torch.multiprocessing import Pool


# class SimpleDataSet(Dataset):
#
#     def __init__(self, D, num_examples=20):
#         self.data = [torch.randn(D) for _ in range(num_examples)]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]

def main():
    Din, Dout = 3, 1
    model = nn.Linear(Din, Dout)
    criterion = nn.MSELoss()

    def get_loss(data_point):
        x, y = data_point
        y_pred = model(x)
        loss = criterion(y_pred, y)
        return loss

    batch_size = 3
    num_epochs = 10
    num_batches = 5
    num_procs = 5
    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch = [(torch.randn(Din), torch.randn(Dout)) for _ in range(batch_size)]
            with Pool(num_procs) as pool:
                losses = pool.map(get_loss, batch)
                loss = torch.avg(losses)
                loss.backward()


if __name__ == '__main__':
    main()

# %%

# counting number of processors: https://stackoverflow.com/questions/23816546/how-many-processes-should-i-run-in-parallel

# %%

# # List of tuples
# students = [('jack', 34, 'Sydeny', 'Australia'),
#             ('Riti', 30, 'Delhi', 'India'),
#             ('Vikas', 31, 'Mumbai', 'India'),
#             ('Neelu', 32, 'Bangalore', 'India'),
#             ('John', 16, 'New York', 'US'),
#             ('Mike', 17, 'las vegas', 'US')]
# # Create DataFrame object from a list of tuples
# dfObj = pd.DataFrame(students, columns=['Name', 'Age', 'City', 'Country'], index=['a', 'b', 'c', 'd', 'e', 'f'])

# %%

"""
Goal: train in a mp way by computing each example in a seperate process.


tutorial: https://pytorch.org/docs/stable/notes/multiprocessing.html
full example: https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py

Things to figure out:
- fork or spwan for us? see pytorch but see this too https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.run
- shared memory
- do we need num_workers=0, 1 or 2? (one for main thread other for pre-fetching batches)
- run test and check that the 112 process do improve the time for a loop (add progress part for dataloder

docs: https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing

(original python mp, they are compatible: https://docs.python.org/3/library/multiprocessing.html)
"""

# def train(cpu_parallel=True):
#     num_cpus = get_num_cpus()  # 112 is the plan for intel's clsuter as an arparse or function
#     model.shared_memory()  # TODO do we need this?
#     # add progressbar for data loader to check if multiprocessing is helping
#     for batch_idx, batch in dataloader:
#         # do this mellow with pool when cpu_parallel=True
#         with Pool(num_cpus) as pool:
#             losses = pool.map(target=model.forward, args=batch)
#             loss = torch.sum(losses)
#             # now do .step as normal

# https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from torch.multiprocessing import Pool


class SimpleDataSet(Dataset):

    def __init__(self, Din, num_examples=23):
        self.x_dataset = [torch.randn(Din) for _ in range(num_examples)]
        # target function is x*x
        self.y_dataset = [x ** 2 for x in self.x_dataset]

    def __len__(self):
        return len(self.x_dataset)

    def __getitem__(self, idx):
        return self.x_dataset[idx], self.y_dataset[idx]


def get_loss(args):
    x, y, model = args
    y_pred = model(x)
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y)
    return loss


def get_dataloader(D, num_workers, batch_size):
    ds = SimpleDataSet(D)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return dl


def train_fake_data():
    num_workers = 2
    Din, Dout = 3, 1
    model = nn.Linear(Din, Dout).share_memory()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    batch_size = 2
    num_epochs = 10
    # num_batches = 5
    num_procs = 5
    dataloader = get_dataloader(Din, num_workers, batch_size)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(num_epochs):
        for _, batch in enumerate(dataloader):
            batch = [(torch.randn(Din), torch.randn(Dout), model) for _ in batch]
            with Pool(num_procs) as pool:
                optimizer.zero_grad()

                losses = pool.map(get_loss, batch)
                loss = torch.mean(losses)
                loss.backward()

                optimizer.step()
            # scheduler
            scheduler.step()


if __name__ == '__main__':
    # start = time.time()
    # train()
    train_fake_data()
    # print(f'execution time: {time.time() - start}')

# %%

"""
The distributed package included in PyTorch (i.e., torch.distributed) enables researchers and practitioners to
easily parallelize their computations across processes and clusters of machines.

As opposed to the multiprocessing (torch.multiprocessing) package, processes can use different communication backends
and are not restricted to being executed on the same machine.


https://pytorch.org/tutorials/intermediate/dist_tuto.html

"""
"""run.py:"""
# !/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size):
    """
    Distributed function to be implemented later.

    This is the function that is actually ran in each distributed process.
    """
    pass


def init_process_and_run_parallel_fun(rank, size, fn, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library for process to communicate/coordinate with each other/master. It's a backend library.
    """
    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # TODO: I think this is what makes sure that each process can talk to master,
    dist.init_process_group(backend, rank=rank, world_size=size)
    # run parallel function
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # target is the function the (parallel) process will run with args
        p = Process(target=init_process_and_run_parallel_fun, args=(rank, size, run))
        p.start()  # start process
        processes.append(p)

    # wait for all processes to finish by blocking one by one (this code could be problematic see spawn: https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses )
    for p in processes:
        p.join()  # blocks until p is done

# %%

# split string

print("asdf-ghjkl;".split('-'))

# %%

# what happens if we do a .item of a vector

x = torch.randn(1)
print(x.item())

y = torch.randn(1, 1)
print(y.item())

z = torch.randn(1, 4)
print(z.item())

# %%

# attention mechanism from transformers
# inspired from but slightly modified to match equations from paper properly (better vectorization)
# https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a

# -- get data set, each row is an input [N by D] --

import torch

x = [
    [1.0, 0, 1, 0],  # Input 1
    [0, 2, 0, 2],  # Input 2
    [1, 1, 1, 1]  # Input 3
]
x = torch.tensor(x)

print('Usual design matrix where the rows N is the # of examples and D columns the features [N, D]')
print(f'X = [N, D] = {x.size()}\n')

# -- get query, key, value matrices

w_key = [
    [0.0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0]
]
w_query = [
    [1.0, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1]
]
w_value = [
    [0.0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0]
]
w_key = torch.tensor(w_key)
w_query = torch.tensor(w_query)
w_value = torch.tensor(w_value)

print(f'w_key = [D, D_k] = {w_key.size()}')
print(f'w_qry = [D, D_qry] = {w_query.size()}')
print(f'w_val = [D, D_v] = {w_value.size()}\n')

# -- get Q, K, V matrices for each inputs --

keys = x @ w_key
querys = x @ w_query
values = x @ w_value  # [N, D] [D, Dv] = [N, Dv]

# print(keys)
# print(querys)
# print(values)
print(f'keys = K = [N, D_k] = {keys.size()}')
print(f'qry = Q = [N, D_q] = {querys.size()}')
print(f'val = V = [N, D_v] = {values.size()}\n')

# -- calculate attention socres --

# [q1 ; q2; q3 ] @ [k1, k2, k3]
attn_scores = querys @ keys.T

print('Attention scores Q @ K.T')
print(f'attn_scores = [N, N] = {attn_scores.size()}')
print(f'each row i indicates how query values for input i compares to the keys for all others inputs\n')

# -- get real attention --

# have rows sum to 1
attn_scores_softmax = attn_scores.softmax(dim=1)
print(attn_scores_softmax[0, :].sum())
print(attn_scores_softmax[0, :])
print('a[0,0]=<q0, k0>, a[0,1]=<q0,k1> , a[0,2]=<q0,k2>')
# print(attn_scores_softmax)
print(
    'Thus, each row i is a (normalized) weight [0,1] indicating how much each qry input i compares to all others inputs keys')

# For readability, approximate the above as follows
attn_scores_softmax = [
    [0.0, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)

# -- --

# the output of attention from the tutorial:
print((values[:, None] * attn_scores_softmax.T[:, :, None]).sum(dim=0))

# using the equation from the paper [N, N] [N, Dv] = [N, Dv]
sf_qk_v = attn_scores_softmax @ values

print('Here is the attentted "context" vectors!')
print(f'Atten(QK.T) @ V = A*V = [N, Dv] = {sf_qk_v.size()}')
print(sf_qk_v)
print((values[:, None] * attn_scores_softmax.T[:, :, None]).sum(dim=0))
print('Each row i is a context vector weighted with qry i with all keys for 1...Tx by vectors v 1...Tx')
print('i.e. AV[i,:] = sum^Tx_{t=1} a[i,t] v[:,i]')

# %%
#
# from pathlib import Path
# from types import SimpleNamespace
# from torch.utils.tensorboard import SummaryWriter
#
# import pickle
#
# args = SimpleNamespace(log_dir=Path('~/Desktop/').expanduser())
# tb = import torch
#
# class ResNet(torch.nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module
#
#     def forward(self, inputs):
#         return self.module(inputs) + inputsSummaryWriter(log_dir=args.log_dir)  # uncomment for documentation to work
#
# # TypeError: cannot pickle 'tensorflow.python._pywrap_file_io.WritableFile' object
# pickle.dump(tb, open(args.log_dir / 'tb_test' ,'w'))

# %%

import torch


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


#
# mdl = nn.Sequential()

# %%

# layer norm

import torch.nn as nn

input = torch.randn(20, 5, 10, 10)
# With Learnable Parameters
m = nn.LayerNorm(input.size()[1:])
# Without Learnable Parameters
m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
# Normalize over last two dimensions
m = nn.LayerNorm([10, 10])
# Normalize over last dimension of size 10
m = nn.LayerNorm(10)
# Activating the module
output = m(input)

input = torch.randn(20, 256)
# With Learnable Parameters
m = nn.LayerNorm(normalized_shape=256)
# Without Learnable Parameters
# m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
# Normalize over last two dimensions
# m = nn.LayerNorm([10, 10])
# Normalize over last dimension of size 10
# m = nn.LayerNorm(10)
# Activating the module
output = m(input)

print(output.size())

print('-- testing batch size 1 --')

input = torch.randn(1, 256)
# With Learnable Parameters
m = nn.LayerNorm(normalized_shape=256)
# Without Learnable Parameters
# m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
# Normalize over last two dimensions
# m = nn.LayerNorm([10, 10])
# Normalize over last dimension of size 10
# m = nn.LayerNorm(10)
# Activating the module
output = m(input)

print(output.size())

# %%

# f string formatting
# https://miguendes.me/73-examples-to-help-you-master-pythons-f-strings#how-to-add-leading-zeros

# fixed digits after f f-string
print(f'{10.1234:.2f}')

# add 5 leading zeros (note you need the 0 infront of 5
print(f'{42:05}')

num = 42

f"{num:05}"
'00042'

f'{num:+010}'
'+000000042'

f'{num:-010}'
'0000000042'

f"{num:010}"
'0000000042'

num = -42

f'{num:+010}'
'-000000042'

f'{num:010}'
'-000000042'

f'{num:-010}'
'-000000042'

# %%

# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)

# %%

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input.dtype)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target.dtype)
output = loss(input, target)
output.backward()

# %%


print(spt_logits_t.dtype)
print(spt_y_t.dtype)
inner_loss = self.args.criterion(spt_logits_t, spt_y_t)

# %%

# view(-1), view(-1, 1)
# https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view
# the actual value for this dimension will be inferred so that the number of elements in the view matches
# the original number of elements.

import torch

x = torch.randn(1, 5)
x = x.view(-1)
print(x.size())

x = torch.randn(2, 4)
x = x.view(-1, 8)
print(x.size())

x = torch.randn(2, 4)
x = x.view(-1)
print(x.size())

x = torch.randn(2, 4, 3)
x = x.view(-1)
print(x.size())

# %%

import torch

x = torch.randn(torch.Size([5, 1028]))
y = torch.randn(torch.Size([5, 1028]))
# x = (y == x).view(-1)
x = (y == x).reshape(-1)
print(x.size())

# %%

# contiguous vs non-contiguous tensors
# https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
# seems view vs reshape care about
# note that sometimes `view` doesn't work due
# to contiguous/non-contiguous memory so call `reshape(...)`
# instead: https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107 and see https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
# https://stackoverflow.com/questions/48915810/pytorch-contiguous
# https://stackoverflow.com/questions/54095351/in-pytorch-what-makes-a-tensor-have-non-contiguous-memory
# https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch

# %%

# nhttps://pytorch.org/tutorials/beginner/transformer_tutorial.html
# positional encoder pytorch

# transformer docs
# where S is the source sequence length,
# T is the target sequence length, N is the batch size, E is the feature number

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)

# generate_square_subsequent_mask(sz)[SOURCE]
# Generate a square mask for the sequence.
# The masked positions are filled with float(‘-inf’).
# Unmasked positions are filled with float(0.0).

# output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

# Transformer Layers
# nn.Transformer
#
# A transformer model.
#
# nn.TransformerEncoder
#
# TransformerEncoder is a stack of N encoder layers
#
# nn.TransformerDecoder
#
# TransformerDecoder is a stack of N decoder layers
#
# nn.TransformerEncoderLayer
#
# TransformerEncoderLayer is made up of self-attn and feedforward network.
#
# nn.TransformerDecoderLayer
#
# TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

print(out.size())

# %%

# attention

# where S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
# src: (S, N, E)(S,N,E) .
# tgt: (T, N, E)(T,N,E) .
# src_mask: (S, S)(S,S) .
# tgt_mask: (T, T)(T,T)

import torch
import torch.nn as nn

batch_size = 4
S = 12
T = 17
d_model = 8
nhead = 1
transformer_model = nn.Transformer(d_model=d_model, nhead=nhead, num_decoder_layers=6, num_encoder_layers=6)
src = torch.rand((S, batch_size, d_model))
tgt = torch.rand((T, batch_size, d_model))
out = transformer_model(src, tgt)

print(out.size())

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
qry = src
key = src
value = src
out = mha(qry, key, value)
print(len(out))
# Shapes for outputs:
# attn_output: (L, N, E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
# attn_output_weights: (N, L, S) where N is the batch size,
# L is the target sequence length, S is the source sequence length.
print(out[0].size())
print(out[1].size())

# %%

# https://stackoverflow.com/questions/52981833/sklearn-python-log-loss-for-logistic-regression-evaluation-raised-an-error/66569833#66569833

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])

clf.predict_proba(X[:2, :])

clf.score(X, y)

y_probs = cls.predict_proba(X)
qry_loss_t = metrics.log_loss(y, y_probs)

# %%

# refs:
# https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch
# https://discuss.pytorch.org/t/how-to-calculate-accuracy-in-pytorch/80476/5
# https://discuss.pytorch.org/t/how-does-one-get-the-predicted-classification-label-from-a-pytorch-model/91649

# how to get the class prediction

batch_size = 4
n_classes = 2
y_logits = torch.randn(batch_size, n_classes)  # usually the scores
print('scores (logits) for each class for each example in batch (how likely a class is unnormalized)')
print(y_logits)
print('the max over entire tensor (not usually what we want)')
print(y_logits.max())
print('the max over the n_classes dim. For each example in batch returns: '
      '1) the highest score for each class (most likely class)\n, and '
      '2) the idx (=class) with that highest score')
print(y_logits.max(1))

print('-- calculate accuracy --')

# computing accuracy in pytorch
"""
random.choice(a, size=None, replace=True, p=None)
Generates a random sample from a given 1-D array

for pytorch random choice https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
"""

import torch
import torch.nn as nn

in_features = 1
n_classes = 10
batch_size = n_classes

mdl = nn.Linear(in_features=in_features, out_features=n_classes)

x = torch.randn(batch_size, in_features)
y_logits = mdl(x)  # scores/logits for each example in batch [B, n_classes]
# get for each example in batch the label/idx most likely according to score
# y_max_idx[b] = y_pred[b] = argmax_{idx \in [n_classes]} y_logit[idx]
y_max_scores, y_max_idx = y_logits.max(dim=1)
y_pred = y_max_idx  # predictions are really the inx \in [n_classes] with the highest scores
y = torch.randint(high=n_classes, size=(batch_size,))
# accuracy for 1 batch
assert (y.size(0) == batch_size)
acc = (y == y_pred).sum() / y.size(0)
acc = acc.item()

print(y)
print(y_pred)
print(acc)

# %%

# topk accuracy

# torch.topk = Returns the k largest elements of the given input tensor along a given dimension.

import torch

batch_size = 2
n_classes = 3
y_logits = torch.randn(batch_size, n_classes)
print('- all values in tensor x')
print(y_logits)
print('\n- for each example in batch get top 2 most likely values & classes/idx (since dim=1 is the dim for classes)'
      '\n1) first are the actual top 2 scores & 2) then the indicies/classes corresponding those largest scores')
print(y_logits.topk(k=2, dim=1))

# %%

from copy import deepcopy

from pathlib import Path

path_cluster = '/home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb'
path_cluster_intel = '/homes/miranda9/data/logs/logs_Dec04_15-49-00_jobid_446010.iam-pbs/tb'
path_vision = '/home/miranda9/data/logs/logs_Dec04_18-39-14_jobid_1528/tb'
dirs = path_cluster.split('/')

for dir_name in deepcopy(dirs):
    if dir_name == 'data':
        break
    else:
        dirs.pop(0)
dirs = ['~'] + dirs
dirs = '/'.join(dirs)
dir = Path(dirs).expanduser()

path_cluster.replace('/home/miranda9/', '~')

print(dir)

# %%

# floats f-string

var = 1.0
print(f'{var}')

# f adds many ugly 0's
var = 1
print(f'{var:f}')

var = 0.0001
print(f'{var}')

# ok it truncates, no!

var = 1.0
print(f'{var:.2f}')

var = 0.0001
print(f'{var:.2f}')

# %%

import bisect
from collections import OrderedDict

p = 0
x = bisect.bisect_left([10, 20], p)
print(x)
p = 10
x = bisect.bisect_left([10, 20], p)
print(x)
p = 11
x = bisect.bisect_left([10, 20], p)
print(x)
p = 21
x = bisect.bisect_left([10, 20], p)
print(x)

#
# p = 10
# x = bisect.bisect_left(OrderedDict({10: 'a', 11: 'b'}), p)
# print()

# %%

# for indexing into an interval to get the index the value corresponds to

import bisect

flatten_lst_files = ['f1', 'f2', 'f3']
cummulative_end_index = [4, 5 + 6, 5 + 7 + 1]
print(cummulative_end_index)
files = {'f1': list(range(5)), 'f2': list(range(7)), 'f3': list(range(2))}


def get_lower_cummulative(file_idx):
    if file_idx == 0:
        return file_idx
    else:
        return cummulative_end_index[file_idx - 1] + 1


def get_node_idx(idx):
    # gets the index for the value we want
    file_idx = bisect.bisect_left(cummulative_end_index, idx)
    # now get the actual value
    file = flatten_lst_files[file_idx]
    print(file)
    lower_cummulative_val = get_lower_cummulative(file_idx)
    node_idx = idx - lower_cummulative_val
    # print(node_idx)
    node = files[file][node_idx]
    # print(node)
    return node


for idx in range(5 + 7 + 2):
    node = get_node_idx(idx)
    print(node)
    print()

# %%

# computing cummulative sums counts frequencies

import pandas as pd

# importing numpy module
import numpy as np

# making list of values
values = [3, 4, 7, 2, 0]

# making series from list
series = pd.Series(values)

# calling method
cumsum = list(series.cumsum())
cumsum = np.array(series.cumsum())

# display
print(cumsum)

# %%

# splitting list of files into 3 train, val, test

import numpy as np


def split_two(lst, ratio=[0.5, 0.5]):
    assert (np.sum(ratio) == 1.0)  # makes sure the splits make sense
    train_ratio = ratio[0]
    # note this function needs only the "middle" index to split, the remaining is the rest of the split
    indices_for_splittin = [int(len(lst) * train_ratio)]
    train, test = np.split(lst, indices_for_splittin)
    return train, test


def split_three(lst, ratio=[0.8, 0.1, 0.1]):
    import numpy as np

    train_r, val_r, test_r = ratio
    assert (np.sum(ratio) == 1.0)  # makes sure the splits make sense
    # note we only need to give the first 2 indices to split, the last one it returns the rest of the list or empty
    indicies_for_splitting = [int(len(lst) * train_r), int(len(lst) * (train_r + val_r))]
    train, val, test = np.split(lst, indicies_for_splitting)
    return train, val, test


files = list(range(10))
train, test = split_two(files)
print(train, test)
train, val, test = split_three(files)
print(train, val, test)

# %%

from typing import List, NewType

# https://stackoverflow.com/questions/33045222/how-do-you-alias-a-type-in-python

Vector = List[float]  # alias shortens
URL = NewType("URL", str)  # new type

# this is better since URL is a string but any string is NOT usually a URL
print(URL is str)

# %%

# convert list of ints to tensor

import torch

y_batch = [944104324, 146561759, 938461041, 1035383419]
y_batch = torch.tensor(y_batch)
print(y_batch)
print(type(y_batch))
print(y_batch.dtype)

# %%

# counter

from collections import Counter

vocab = Counter()
lst = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
for elem in lst:
    vocab.update([elem])

print(vocab)

vocab.update(lst)
print(vocab)

print(Counter(['a', 'a', 'b']))

# Counter({'R': 1, 'A': 1, 'Coq.Relations.Relation_Operators.clos_refl_trans': 1, '0': 1})
# vocab.update(['adsf'])
# vocab
# Counter({'R': 1, 'A': 1, 'Coq.Relations.Relation_Operators.clos_refl_trans': 1, '0': 1, 'adsf': 1})
# Counter(a=0)
# Counter({'a': 0})
# vocab.update({'qwert':0}

# %%

from argparse import Namespace

opts = Namespace(rank=-1, world_size=0, batch_size=4, split='train', num_workers=0)
opts.path2dataprep = Path('~/data/lasse_datasets_coq/dag_data_prep.pt').expanduser()
opts.path2vocabs = Path('~/data/lasse_datasets_coq/dag_counters.pt').expanduser()
opts.path2hash2idx = Path('~/data/lasse_datasets_coq/dag_hash2index.pt').expanduser()

counters = torch.load(opts.path2vocabs)
vocab = counters['leafs_counter']
constructors = counters['cons_counter']
db_hash2idx = torch.load(opts.path2hash2idx)
hash2idx = db_hash2idx['hash2idx']
num_tactic_hashes = len(hash2idx.keys())

# %%

# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import torchtext
from collections import Counter

# text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
# label_pipeline = lambda x: int(x) - 1

counter = Counter()

counter.update(['a', 'a', 'a', 'bob', 'bob', 'cat', 'dog'])
print(counter)

vocab = torchtext.vocab.Vocab(counter)
vocab2 = torchtext.vocab.Vocab(counter, min_freq=2)

print(vocab)
# print('a' in vocab)
print(vocab['a'])
print(vocab['bob'])
print(vocab['cat'])
print(vocab['dog'])
print(vocab['asdf'])

print()
print(vocab2['a'])
print(vocab2['bob'])
print(vocab2['cat'])
print(vocab2['dog'])
print(vocab['asdf'])

print()
print(len(vocab))

# text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
#
# from torch.utils.data import DataLoader
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def collate_batch(batch):
#     label_list, text_list, offsets = [], [], [0]
#     for (_label, _text) in batch:
#          label_list.append(label_pipeline(_label))
#          processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#          text_list.append(processed_text)
#          offsets.append(processed_text.size(0))
#     label_list = torch.tensor(label_list, dtype=torch.int64)
#     offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
#     text_list = torch.cat(text_list)
#     return label_list.to(device), text_list.to(device), offsets.to(device)
#
# train_iter = AG_NEWS(split='train')
# dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

# %%

import torch

x = torch.randn([1, 4])
y = torch.randn([1, 4])

xy = torch.stack([x, y])
print(xy.size())

xy = torch.cat([x, y])
print(xy.size())

# %%

"""

python -m memory_profiler file.py

"""

# %%

# list of letters
letters = ['a', 'b', 'd', 'e', 'i', 'j', 'o']


# function that filters vowels
def filter_vowels(letter):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if letter in vowels:
        return True
    else:
        return False


print(filter)
filtered_vowels = filter(filter_vowels, letters)

print('The filtered vowels are:')
for vowel in filtered_vowels:
    print(vowel)

# %%
# The filter() method constructs an iterator from elements of an iterable for which a function returns true.

# filter things that are not None, i.e. we want to keep things that are not None.

list2filter = ['a', 'b', None]
print(list2filter)

filteredlist = filter(lambda x: x is not None, list2filter)
print(list(filteredlist))

# this is much better: https://stackoverflow.com/questions/61925671/use-only-some-items-in-a-list-comprehension

# %%

import capnp

# import dag_api_capnp
capnp.remove_import_hook()
try:
    # first try to see if we are running a benchmark & the capnp schema is in the share conda folder
    dag_api_capnp = str(Path(os.environ['CONDA_PREFIX'] + '/share/dag_api.capnp').expanduser())
    dag_api_capnp = capnp.load(dag_api_capnp)
except:
    # else run the one in the main project folder
    dag_api_capnp = str(Path('~/coq-tactician-graph/src/dag_api.capnp').expanduser())
    dag_api_capnp = capnp.load(dag_api_capnp)

# %%

import capnp

capnp.remove_import_hook()
example_msg_capnp = Path("~/ultimate-utils/example_msg.capnp").expanduser()
example_msg_capnp = capnp.load(str(example_msg_capnp))

# Building
addresses = example_msg_capnp.AddressBook.newMessage()
people = addresses.init('people', 1)

# %%

# python index slicing

lst = [1, 2, 3, 4]
print(lst[:0])
print(lst[:1])
print(lst[:2])

# its non inclusive

# %%

import dgl.data

dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)

# %%

import dgl
import numpy as np
import torch

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)

u = np.concatenate([src, dst])
v = np.concatenate([dst, src])
# Construct a DGLGraph
dgl.DGLGraph((u, v))

# %%

import dgl
import numpy as np
import torch

import networkx as nx

import matplotlib.pyplot as plt

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
print(f'{g=}')
print(f'{g.edges()=}')

# Since the actual graph is undirected, we convert it for visualization purpose.
nx_G = g.to_networkx().to_undirected()
print(f'{nx_G=}')

# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True)

plt.show()

# %%

# https://stackoverflow.com/questions/28533111/plotting-networkx-graph-with-node-labels-defaulting-to-node-name

import dgl
import numpy as np
import torch

import networkx as nx

import matplotlib.pyplot as plt

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
print(f'{g=}')
print(f'{g.edges()=}')

# Since the actual graph is undirected, we convert it for visualization purpose.
g = g.to_networkx().to_undirected()
print(f'{g=}')

labels = {0: "app", 1: "cons", 2: "with", 3: "app3", 4: "app4", 5: "app5"}

# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos, labels=labels, with_labels=True)

plt.show()

# %%

from graphviz import Digraph

g = Digraph('G', filename='hello2.gv')
print(f'{g=}')

g.edge('Hello', 'World')

g.view()

# %%

import dgl
import numpy as np
import torch

import networkx as nx

import matplotlib.pyplot as plt

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
print(f'{g=}')
print(f'{g.edges()=}')

# Since the actual graph is undirected, we convert it for visualization purpose.
g = g.to_networkx().to_undirected()
g = nx.nx_agraph.to_agraph(g)
g.layout()
# g.draw()
g.draw("file.png")
print(f'{g=}')

# plt.show()

# from IPython.display import Image, display
#
# def view_pydot(pdot):
#     plt = Image(pdot.create_png())
#     display(plt)
#
# view_pydot(g)

# %%

# https://stackoverflow.com/questions/28533111/plotting-networkx-graph-with-node-labels-defaulting-to-node-name

import dgl
import numpy as np
import torch

import networkx as nx

import matplotlib.pyplot as plt

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
print(f'{g=}')
print(f'{g.edges()=}')

# Since the actual graph is undirected, we convert it for visualization purpose.
g = g.to_networkx().to_undirected()
print(f'{g=}')

# relabel
int2label = {0: "app", 1: "cons", 2: "with", 3: "app3", 4: "app4", 5: "app5"}
g = nx.relabel_nodes(g, int2label)

# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos, with_labels=True)

plt.show()

#%%

# https://stackoverflow.com/questions/28533111/plotting-networkx-graph-with-node-labels-defaulting-to-node-name

import dgl
import numpy as np
import torch

import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
print(f'{g=}')
print(f'{g.edges()=}')

# Since the actual graph is undirected, we convert it for visualization purpose.
g = g.to_networkx().to_undirected()
print(f'{g=}')

# relabel
int2label = {0: "app", 1: "cons", 2: "with", 3: "app3", 4: "app4", 5: "app5"}
g = nx.relabel_nodes(g, int2label)

# https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
g = nx.nx_agraph.to_agraph(g)
print(f'{g=}')
print(f'{g.string()=}')

# draw
g.layout()
g.draw("file.png")

# https://stackoverflow.com/questions/20597088/display-a-png-image-from-python-on-mint-15-linux
img = mpimg.imread('file.png')
plt.imshow(img)
plt.show()

# remove file https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder
Path('./file.png').expanduser().unlink()
# import os
# os.remove('./file.png')

# %%

# networkx to dgl: https://docs.dgl.ai/en/0.6.x/generated/dgl.from_networkx.html

import dgl
import networkx as nx
import numpy as np
import torch

nx_g = nx.DiGraph()
# Add 3 nodes and two features for them
nx_g.add_nodes_from([0, 1, 2], feat1=np.zeros((3, 1)), feat2=np.ones((3, 1)))
# Add 2 edges (1, 2) and (2, 1) with two features, one being edge IDs
nx_g.add_edge(1, 2, weight=np.ones((1, 1)), eid=np.array([1]))
nx_g.add_edge(2, 1, weight=np.ones((1, 1)), eid=np.array([0]))

g = dgl.from_networkx(nx_g)

# ... https://docs.dgl.ai/en/0.6.x/generated/dgl.from_networkx.html


# %%

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_node('a')
G.add_node('b', attr1='cons')
print(f'{G=}')

pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

# adding reated edges and nodes: https://stackoverflow.com/questions/28488559/networkx-duplicate-edges/51611005

#%%

import pylab
import networkx as nx
g=nx.Graph()
g.add_node('Golf',size='small')
g.add_node('Hummer',size='huge')
g.add_edge('Golf','Hummer')
labels = nx.get_node_attributes(g, 'size')
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos, labels=labels, with_labels=True)
# nx.draw(g, labels=labels)
pylab.show()


#%%

import dgl
import numpy as np
import torch

import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path

g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
print(f'{g=}')
print(f'{g.edges()=}')

# Since the actual graph is undirected, we convert it for visualization purpose.
g = g.to_networkx().to_undirected()
print(f'{g=}')

# relabel
int2label = {0: "app", 1: "cons", 2: "with", 3: "app3", 4: "app4", 5: "app"}
g = nx.relabel_nodes(g, int2label)

# https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
g = nx.nx_agraph.to_agraph(g)
print(f'{g=}')
print(f'{g.string()=}')

# draw
g.layout()
g.draw("file.png")

# https://stackoverflow.com/questions/20597088/display-a-png-image-from-python-on-mint-15-linux
img = mpimg.imread('file.png')
plt.imshow(img)
plt.show()

# remove file https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder
Path('./file.png').expanduser().unlink()
# import os
# os.remove('./file.png')

#%%

import pylab
import networkx as nx
g=nx.Graph()
g.add_node('Golf',size='small')
g.add_node('Hummer',size='huge')
g.add_edge('Golf','Hummer')
labels = nx.get_node_attributes(g, 'size')
pos = nx.kamada_kawai_layout(g)
nx.draw(g, pos, labels=labels, with_labels=True)
pylab.show()

#%%

import pygraphviz as pgv

g = pgv.AGraph()
g.add_node('Golf',label='small')
g.add_node('Hummer',label='huge')
g.add_edge('Golf','Hummer')
# labels = nx.get_node_attributes(g, 'size')
# pos = nx.kamada_kawai_layout(g)
# nx.draw(g, pos, labels=labels, with_labels=True)
# pylab.show()
g.layout()
g.draw('file.png')

img = mpimg.imread('file.png')
plt.imshow(img)
plt.show()

# Path('./file.png').expanduser().unlink()
#%%

import pygraphviz as pgv
G=pgv.AGraph()
ndlist = [1,2,3]
for node in ndlist:
    # label = "Label #" + str(node)
    label = "app"
    G.add_node(node, label=label)
G.layout()
G.draw('example.png', format='png')

img = mpimg.imread('example.png')
plt.imshow(img)
plt.show()

# Path('./file.png').expanduser().unlink()

#%%

# load a graph from a dot for networkx: https://stackoverflow.com/questions/42172548/read-dot-graph-in-networkx-from-a-string-and-not-file
# G = nx_agraph.from_agraph(pygraphviz.AGraph(dotFormat))

#%%

import dgl.data

dataset = dgl.data.CoraGraphDataset()
print('-- my print statments --')
print('Number of categories: {dataset.num_classes} \n')

g = dataset[0]
print(f'{g=}')

# print('Node features')
# print(g.ndata)
# print('Edge features')
# print(g.edata)

h_node_features = g.ndata['feat']
print(h_node_features.size())

#%%

# import dgl
import networkx as nx
import numpy as np
import torch

nx_g = nx.DiGraph()
# Add 3 nodes and two features for them
nx_g.add_nodes_from([0, 1, 2], feat1=np.zeros((3, 1)), feat2=np.ones((3, 1)))
print(f'{nx_g=}')
# Add 2 edges (1, 2) and (2, 1) with two features, one being edge IDs
nx_g.add_edge(1, 2, weight=np.ones((1, 1)), eid=np.array([1]))
nx_g.add_edge(2, 1, weight=np.ones((1, 1)), eid=np.array([0]))
print(f'{nx_g=}')

#%%

import random

foo = ['a', 'b', 'c', 'd', 'e']
print(random.choice(foo))

#%%

pf_body1 = ['Proof.',
            'unfold lookup_incl;', 'simpl;', 'intros.',
            'match_destr;', 'unfold equiv in *;', 'subst.',
            '- apply lookup_in in H1.',
            'apply in_dom in H1.',
            'intuition.',
            '- auto.',
            'Qed.']
pf_body1 = pf_body1[1:-1]
print(pf_body1)

#%%

pf_body1 = [ 'Proof.',
  'unfold lookup_incl;', 'simpl;', 'intros.',
  'match_destr;', 'unfold equiv in *;', 'subst.',
  '- apply lookup_in in H1.',
    'apply in_dom in H1.',
    'intuition.',
  '- auto.',
'Qed.']

def mask_lemma_in_pf_body(pf_body:str, lemma:str, mask_token:str='<Predict>') -> str:
    return [tactic_cmd.replace(lemma, mask_token) for tactic_cmd in pf_body]

print(mask_lemma(pf_body1, 'in_dom'))

#%%

x = [1,2]
xx = ['a','b']
print(list(zip(x,xx)))

#%%

thms = "lookup_incl_cons_l_nin (l1 l2:list (A*B)) x y : \
  lookup_incl l1 l2 -> \
  ~ In x (domain l1) -> \
  lookup_incl l1 ((x,y)::l2)."
pf_bodies = [['Proof.',
              'unfold lookup_incl;', 'simpl;', 'intros.',
              'match_destr;', 'unfold equiv in *;', 'subst.',
              '- apply lookup_in in H1.',
              'apply in_dom in H1.',
              'intuition.',
              '- auto.',
              'Qed.']]
pf_bodies[0] = pf_body[0][1:-1]

#%%

from lark import Lark
json_parser = Lark(r"""
    value: dict
         | list
         | ESCAPED_STRING
         | SIGNED_NUMBER
         | "true" | "false" | "null"

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : ESCAPED_STRING ":" value

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='value')
text = '{}'
ast = json_parser.parse(text)
print(ast.pretty())

text = '{"key": ["item0", "item1", 3.14]}'
ast = json_parser.parse(text)
print(ast.pretty())

#%%

from lark import Lark
json_parser = Lark(r"""
    value: dict dict "f"
         | list
         | ESCAPED_STRING
         | SIGNED_NUMBER
         | "true" | "false" | "null"

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : ESCAPED_STRING ":" value

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='value')

text = '{} {} f'
ast = json_parser.parse(text)
print(ast)
print(ast.pretty())

# text = '{"key": ["item0", "item1", 3.14, "true"]}'
# ast = json_parser.parse(text)
# print(ast)
# print(ast.pretty())

#%%

from lark import Lark
json_parser = Lark(r"""
    pair: pair "," pair // 1
         | string // 2
    string : "a" // 3
        | "b" // 4

    %import common.WS
    %ignore WS

    """, start='pair', keep_all_tokens=True)

text = 'a'
ast = json_parser.parse(text)
print(ast)
print(ast.pretty())
# rule seq
rule_seq = ['pair', 'string', "1"]
rule_seq2 = ['pair->string', 'string->1']

text = "a, b"
ast = json_parser.parse(text)
print(ast)
print(ast.pretty())
rule_seq2 = ['pair -> pair "," pair', 'pair->string', 'pair->string', 'string->a', 'string->b']
rule_seq3 = [1, 2, 2, 3, 4]
rule_seq3 = [1, 2, 2, 3, 3, 4, 5]

#%%

from lark import Lark, Tree, Token
json_parser = Lark(r"""
    pair: pair "," pair // 1
         | string // 2
    string : "a" // 3
        | "b" // 4

    %import common.WS
    %ignore WS
    """, start='pair', keep_all_tokens=True)

text = 'a'
ast = json_parser.parse(text)
print(ast)
print(ast.pretty())
# rule seq
rule_seq = ['pair', 'string', "1"]
rule_seq2 = ['pair->string', 'string->1']

text = "a, b"
ast = json_parser.parse(text)
print(ast)
print(ast.pretty())
rule_seq2 = ['pair->pair "," pair', 'pair->string', 'pair->string', 'string->a', 'string->b']
rule_seq2 = [rule.split('->') for rule in rule_seq2]
rule_seq3 = [1, 2, 2, 3, 4]

non_terminals = ['pair', 'string']
terminals = [",", "a", "b"]

def is_terminal(sym):
    # true if matches hardcoded symbols in grammar or a regex, note this only works if the nt has been checked first.
    return sym in terminals  # or matches regex

def is_non_terminal(sym):
    return sym in non_terminals

def build_lark_tree(rule_seq:list[tuple]) -> Tree:
    print(rule_seq)
    nt, next_syms = rule_seq[0]
    if len(rule_seq) == 1:
        return Tree(nt, [Token('literal', next_syms)])
    else:
        rule_seq = rule_seq[1:]
        next_syms = next_syms.split(" ")
        asts = []
        nt_idx = 0
        for next_sym in next_syms:
            if is_non_terminal(next_sym):
                ast = Tree(next_sym, build_lark_tree(rule_seq[nt_idx:]))
                nt_idx += 1
            elif is_terminal(next_sym):
                ast = Token('literal', next_sym)
            else:
                raise ValueError(f'Invalid: {next_sym} didnt match anything')
            asts.append(ast)
        return Tree(nt, asts)

print('---- generating ast from Rule Seq')
build_lark_tree(rule_seq2)

#%%

from collections import defaultdict

from lark import Lark, Tree, Token
from lark.grammar import Rule, NonTerminal, Symbol, Terminal
from lark.reconstruct import Reconstructor


def build(rules: list[Rule], rule_seq: list[int], build_term) -> Tree:
    def build_rule(rule: Rule) -> Tree:
        l = []
        for i, e in enumerate(rule.expansion):
            if e.is_term:
                l.append(build_term(e))
            else:
                l.append(e)
                targets[e].append((l, i))
        return Tree(rule.origin.name, l)

    out: list = [NonTerminal("start")]
    targets = defaultdict(list)
    targets[out[0]].append((out, 0))
    for i in rule_seq:
        r = rules[i]
        assert r.alias is None, "Can't have aliases"
        assert r.options.keep_all_tokens, "need to have keep_all_tokens"
        assert not (r.options.expand1 or r.origin.name.startswith("_")), "Can't have a rule that expands it's children"
        ts = targets[r.origin]
        l, i = ts.pop(0)
        assert l[i] == r.origin, l
        l[i] = build_rule(r)
    return out[0]


grammar = r"""
start: "a" // rule 0
     | "a" start // rule 1
"""

parser = Lark(grammar, keep_all_tokens=True)

print(parser.rules)

rule_seq1 = [1, 0]
ast = build(parser.rules, rule_seq1, lambda t: Token(t.name, "a"))
print(ast)
text = Reconstructor(parser).reconstruct(ast, None)  # has string "aa"
print(repr(text))


#%%

from collections import deque

# Initializing a queue
q = deque()

# Adding elements to a queue
q.append('a')
q.append('b')
q.append('c')

print("Initial queue")
print(q)

# Removing elements from a queue
print("\nElements dequeued from the queue")
print(q.popleft())
print(q.popleft())
print(q.popleft())

print("\nQueue after removing elements")
print(q)

#%%

# https://github.com/MegaIng/lark_ast_generator/blob/master/ast_generator.py#L114

#%%

from typing import Union

from collections import deque

from lark import Lark, Tree, Token
from lark.grammar import Rule, NonTerminal, Symbol, Terminal
from lark.reconstruct import Reconstructor

grammar = r"""
    pair: pair "," pair // 1
         | string // 2
    string : "a" // 3
        | "b" // 4

    %import common.WS
    %ignore WS
    """
parser = Lark(grammar, start='pair', keep_all_tokens=True)
print(parser.rules)
print(parser.rules[0])

#%%

# I want a queue that removes the

#%%

from __future__ import annotations

from collections import defaultdict
from random import choice
from typing import Optional, Callable

from lark import Lark, Token, Tree, Transformer
from lark.grammar import Terminal, NonTerminal, Rule
from lark.lexer import TerminalDef
from lark.visitors import Interpreter


class ASTGenerator:
    def __init__(self, parser: Lark, term_builder=None):
        self.parser = parser
        self.term_builder = term_builder
        self.term_by_name = {t.name: t for t in self.parser.terminals}
        self.rule_by_symbol = defaultdict(list)
        for r in self.parser.rules:
            self.rule_by_symbol[r.origin].append(r)

    def _term_builder(self, term: Terminal):
        term_def: TerminalDef = self.term_by_name[term.name]
        if term_def.pattern.type == "str":
            return Token(term.name, term_def.pattern.value)
        elif self.term_builder:
            return self.term_builder(term_def)
        else:
            raise ValueError("Can't build Token for Terminal %r" % term.name)

    def _rule_builder(self, rule: Rule, hole: Hole):
        children = []
        for sym in rule.expansion:
            if sym.is_term:
                if not sym.filter_out or rule.options.keep_all_tokens:
                    children.append(self._term_builder(sym))
            else:
                children.append(sym)
        tree = Tree(rule.alias or rule.origin.name, children)
        if not rule.alias and (tree.data.startswith("_") or (rule.options.expand1 and len(children) == 1)):
            hole.expand(tree)
        else:
            hole.fill(tree)

    def start_build(self, start=None):
        # We could just copy the code
        start = self.parser.parser._verify_start(start)
        return HoleTree(NonTerminal(start))

    def build_absolute_index(self, hole_tree: HoleTree, rules: list[int]):
        for i in rules:
            r = self.parser.rules[i]
            hole = hole_tree.get_for_symbol(r.origin)
            self._rule_builder(r, hole)

    def build_relative_index(self, hole_tree: HoleTree, rules: list[int]):
        meaning = []
        for i in rules:
            hole = hole_tree.bfs_first_hole
            options = self.rule_by_symbol[hole.symbol]
            rule = options[i]
            meaning.append((i, hole.path, rule))
            self._rule_builder(rule, hole)
        return meaning

    def build_picker(self, hole_tree: HoleTree, picker: Callable[[list[Rule], Hole], Rule], n: int = None):
        track = []
        i = 0
        while hole_tree.any_holes and (n is None or i < n):
            hole = hole_tree.bfs_first_hole
            options = self.rule_by_symbol[hole.symbol]
            rule = picker(options, hole)
            track.append(options.index(rule))
            self._rule_builder(rule, hole)
            i += 1
        return track


class InlineTree(Tree):
    pass


class Hole:
    def __init__(self, target: Optional[list], index: int, hole_tree: HoleTree, path: tuple[int, ...]):
        self.target = target
        if target is None:
            self.symbol = index
            self.index = 0
        else:
            self.symbol = target[index]
            self.index = index
        assert isinstance(self.symbol, NonTerminal), self.symbol
        self.hole_tree = hole_tree
        self.path = path

    def _get_holes(self, values, target, offset):
        for i, v in enumerate(values):
            if isinstance(v, NonTerminal):
                yield Hole(target, offset + i, self.hole_tree, (*self.path, i))

    def expand(self, tree: Tree):
        assert self.target is not None, "start rule can't be inlined"
        self.target[self.index] = InlineTree(tree.data, tree.children, tree.meta)
        self.hole_tree.filled(self, self._get_holes(tree.children, tree.children, 0))

    def fill(self, tree: Tree):
        if self.target is None:
            self.hole_tree.set_start(tree)
        else:
            self.target[self.index] = tree
        self.hole_tree.filled(self, self._get_holes(tree.children, tree.children, 0))


def flatten_inline_tree(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, InlineTree):
            for sub_x in flatten_inline_tree(x.children):
                yield sub_x
        else:
            yield x


class _InlineExpands(Interpreter):
    def __default__(self, tree):
        new_tree = Tree(tree.data, list(flatten_inline_tree(tree.children)), tree.meta)
        new_tree.children = self.visit_children(new_tree)
        return new_tree


class HoleTree:
    def __init__(self, start_symbol):
        self._tree = None
        self.holes_by_path = {}
        self.holes_by_symbol = defaultdict(list)
        self.holes_by_path[()] = Hole(None, start_symbol, self, ())
        self.holes_by_symbol[start_symbol].append(self.holes_by_path[()])

    def set_start(self, tree):
        assert self._tree is None
        self._tree = tree

    def filled(self, old_hole, new_holes):
        self.holes_by_symbol[old_hole.symbol].remove(old_hole)
        assert self.holes_by_path.pop(old_hole.path) is old_hole
        for nh in new_holes:
            self.holes_by_symbol[nh.symbol].append(nh)
            assert nh.path not in self.holes_by_path
            self.holes_by_path[nh.path] = nh

    def tree(self, raw:bool = False):
        return _InlineExpands().visit(self._tree) if not raw else self._tree

    @property
    def bfs_first_hole(self):
        return self.holes_by_path[min(self.holes_by_path, key=lambda t: (len(t), t))]

    @property
    def any_holes(self):
        return bool(self.holes_by_path)

    def get_for_symbol(self, symbol):
        return self.holes_by_symbol[symbol][0]


def random_picker(options, hole):
    return choice(options)


def depth(min_depth=3, max_depth=5, base=random_picker):
    def picker(options: list[Rule], hole):
        current = len(hole.path)
        if current < min_depth:
            new_options = [o for o in options
                           if any(not s.is_term for s in o.expansion)]
            if new_options:
                options = new_options
        if current + 1 > max_depth:
            new_options = [o for o in options
                           if all(s.is_term for s in o.expansion)]
            if new_options:
                options = new_options
        return base(options, hole)

    return picker
#%%

from collections import defaultdict
from operator import neg
from typing import Iterable

from lark import Lark, Tree, Token
from lark.grammar import Symbol, NonTerminal, Terminal
from lark.reconstruct import Reconstructor, is_iter_empty
from lark.tree_matcher import is_discarded_terminal, TreeMatcher
from lark.visitors import Transformer_InPlace, Interpreter


class RulesGenerator(Interpreter):
    def __init__(self, parser):
        super(RulesGenerator, self).__init__()
        self.parser = parser
        self.rules_by_name = defaultdict(list)
        self.aliases = defaultdict(list)
        for i, r in enumerate(self.parser.rules):
            self.rules_by_name[r.origin.name].append((r, i))
            if r.alias is not None:
                self.rules_by_name[r.alias].append((r, i))
                self.aliases[r.alias].append(r.origin.name)
        for n, rs in self.rules_by_name.items():
            self.rules_by_name[n] = sorted(rs, key=lambda t: -len(t[0].expansion))
        self.tree_matcher = TreeMatcher(parser)
        self.current_path = []
        self.values = {}

    def _check_name(self, data, target):
        if data == target:
            return True
        elif data in self.aliases:
            return target in self.aliases[data]
        else:
            return False

    def _check_expansion(self, orig_expansion, expansion):
        return len(orig_expansion) == len(expansion) and all(o == e for o, e in zip(orig_expansion, expansion))

    def get_rule(self, tree):
        candidates = self.rules_by_name[tree.data]
        matches = [(r, i) for (r, i) in candidates
                   if self._check_expansion(tree.meta.orig_expansion, r.expansion)]
        if not matches:
            # Sometimes, tree_matcher returns weird self rules Tree('expansion', [Tree('expansion', [...])])
            if len(tree.meta.orig_expansion) == 1 and self._check_name(tree.meta.orig_expansion[0].name, tree.data):
                return None
            assert matches, ("No rule left that was applied", tree, candidates)
        assert len(matches) == 1, ("Can't decide which rule was applied", candidates, matches)
        return matches[0][1]

    def __default__(self, tree):
        if not getattr(tree.meta, 'match_tree', False):
            # print("|"*len(self.current_path), "old", tree)
            tree = self.tree_matcher.match_tree(tree, tree.data)
        # print("|"*len(self.current_path), tree)
        r = self.get_rule(tree)
        for i, c in enumerate(tree.children):
            if isinstance(c, Tree):
                self.current_path.append(i)
                tree.children[i] = self.visit(c)
                self.current_path.pop()
        # print("|"*len(self.current_path),"final", tree)
        if r is not None:
            self.values[tuple(self.current_path)] = r
        return tree

    def get_rules(self, tree) -> Iterable[int]:
        self.current_path = []
        self.values = {}
        self.visit(tree)
        return [i for k, i in sorted(self.values.items(), key=lambda t: tuple(map(neg, t[0])))]

#%%

import pprint as pp
from lark import Lark

grammar = r"""
    ?pair: pair "," pair // 0
         | string // 1
    string : "a" -> aaa // 2
        | "b" -> bbb // 3

    %import common.WS
    %ignore WS
    """
start = 'pair'
parser = Lark(grammar, start=start, keep_all_tokens=True)
# parser = Lark(grammar, start=start)
text = "a, b"
ast = parser.parse(text)
print(ast.pretty())
pp.pprint(parser.rules)

#%%

from lark import Lark

from lark.visitors import Transformer_InPlace, Interpreter

class Inte1(Interpreter):

    def pair(self, tree):
        print('pair')
        self.visit_children(tree)
    
    def string(self, tree):
        print('string')

grammar = r"""
    pair: pair "," pair // 0
         | string // 1
    string : "a" // 2
        | "b" // 3

    %import common.WS
    %ignore WS
    """
start = 'pair'
parser = Lark(grammar, start=start, keep_all_tokens=True)
text = "a, b"
ast = parser.parse(text)
print(ast)
Inte1().visit(ast)

#%%

x = [1, 2, 5]

print(sorted(x))

x = [1.1, 1.5, 0.1, 1.0]
print(list(map(round, x)))
print(sorted(x, key=round))

from lark.tree_matcher import is_discarded_terminal, TreeMatcher
#%%
from lark import Lark
json_parser = Lark(r"""
    value: dict
         | list
         | ESCAPED_STRING
         | SIGNED_NUMBER
         | "true" | "false" | "null"

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : ESCAPED_STRING ":" value

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='value', keep_all_tokens=True)
text = '{}'
ast = json_parser.parse(text)
print(ast.pretty())

text = '{"key": ["item0", "item1", 3.14]}'
ast = json_parser.parse(text)
print(ast.pretty())

#%%

ctx = {'x': 1}
names = ['y', 'y']
all_names = set(list(ctx.keys()) + names)
print(all_names)

#%%
import re

bfs_regex = re.compile(r'x\d+')

assert not bfs_regex.match('x')
print(bfs_regex.match('x'))
print(bfs_regex.search('x'))
assert bfs_regex.match('x0')
print(bfs_regex.match('x0'))
print(bfs_regex.search('x0'))

#%%

print("_".join(['x0']))

print('123'.isnumeric())
print('asdfadsf'.isnumeric())
print('x123'.isnumeric())

#%%

# this checks that both tensors are actually the same
import torch
import torch.nn as nn
import torch.optim as optim

embeds = nn.Embedding(1, 1)
lin = nn.Linear(3, 1)
embeds.weight = torch.nn.Parameter(lin.weight)
sgd = optim.SGD(lin.parameters(), 10)
print(lin.weight)
print(embeds.weight)
out = 10*(2 - lin(torch.randn(1, 3)))*2
out.backward()
sgd.step()
print(lin.weight)
print(embeds.weight)

# this succeded because the weights are the same value after the backward step

#%%

from collections import Counter
from torchtext.vocab import Vocab
import torch.nn as nn

counter_vocab = Counter({'a': 1, 'b': 2, '0': 5})

v = Vocab(counter_vocab)
table = nn.Embedding(len(v), 4)

lookup_tensor = torch.tensor([1, 2, 0], dtype=torch.long)
embed = table(lookup_tensor)

print(embed.size())
print(embed.t().size())

att = embed.t() @ embed
# att = embed.t() * embed

print(att)

from torch.nn.functional import softmax

#%%
import torch

B, T, D = 4, 12, 512
x = torch.randn(B, T, D)
encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
out = encoder(x)
print(out.sum())

encoder.batch_first = False
out = encoder(x)
print(out.sum())

encoder.batch_first = True
out = encoder(x)
print(out.sum())

#%%
import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)
print(out.size())

#%%

# right shift

# [B, Ty] -> [B, Ty] (right shifted, replace initial vectors with random noise)
import torch
from torch import Tensor

y: Tensor = torch.arange(0, 12)
y = y.view(3, 4)
print(y.size())
print(y)
yy = y.roll(shifts=1, dims=1)
yy[:, 0] = 0  # SEEMS DANGEROUS!
print(yy)

# scary, perhaps it's better to only index the first T-1 and then initialize the first as zero...?
#%%

from torchtext.vocab import vocab
from collections import Counter, OrderedDict
counter = Counter(["a", "a", "b", "b", "b"])
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
v1 = vocab(ordered_dict)
print(v1['a']) #prints 1
# print(v1['out of vocab']) #raise RuntimeError since default index is not set
tokens = ['e', 'd', 'c', 'b', 'a']
v2 = vocab(OrderedDict([(token, 1) for token in tokens]))
#adding <unk> token and default index
unk_token = '<unk>'
default_index = -1
if unk_token not in v2:
    v2.insert_token(unk_token, 0)
v2.set_default_index(default_index)
print(v2['<unk>']) #prints 0
print(v2['out of vocab']) #prints -1
#make default index same as index of unk_token
v2.set_default_index(v2[unk_token])
v2['out of vocab'] is v2[unk_token] #prints True


#%%

import torch

# x = torch.randn(2, 3)
# sz = x.size( )
sz = 4
mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# print(x)
print(mask)

#%%

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

#%%

import uutils.torch

# inheritance in python
# Q: do my subclasses inherit the specific values form parent?


class Parent:
    def __init__(self, field):
        self.field = field
        self.child = Child(field)

class Child(Parent):
    def __init__(self, field):
        super().__init__(field)
        self.y = y
        # print(f'{self.field}')

    def forward(self, x):
        print(f'{x=}')
        print(f'{self.field}')

parent = Parent(field=2)

# %%

import torch
from torch.nn.utils.rnn import pad_sequence

# note the sequences start with 2 for SOS and end with 3 for EOS
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']
PAD_IDX = special_symbols.index('<pad>')
src_batch = [torch.tensor([2, 7911, 3]), torch.tensor([2,   8269,  5,  18,  3])]
print(f'batch_size={len(src_batch)}')

src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
print(src_batch.size())
print(src_batch)

#%%

import torch

sz = 4
mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# print(x)
print(mask)

#%%

# https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value

import torch
t = torch.Tensor([0, 2, 3, 2, 1])
print(t.size())
# print(t == 2)
print((t == 2).nonzero().size())
print((t == 2).nonzero())
print((t == 2).nonzero()[0])
print((t == 2).nonzero()[0][0])
print((t == 2).nonzero()[0][0].item())

print((t == 99).nonzero())
print((t == 99).nonzero())

# t = torch.Tensor([1, 0, 2, 3, 2, 2, 1])
# print(t == 222)
# print((t == 222).nonzero(as_tuple=True)[0])
# print((t == 222).nonzero(as_tuple=True))

# print( ((t == 2).nonzero(as_tuple=True)[0]) )
# print( ((t == 2).nonzero(as_tuple=True)[0]).size() )
# print( (t == 2).nonzero() )
# print( (t == 2).nonzero().size() )


#%%

# from lark import Lark
import lark as l

json_parser = l.Lark(r"""
    value: dict dict "f"
         | list
         | ESCAPED_STRING
         | SIGNED_NUMBER
         | "true" | "false" | "null"

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : ESCAPED_STRING ":" value

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='value')

text = '{} {} f'
ast = json_parser.parse(text)
print(ast)
print(ast.pretty())

# %%

import torch

B, T, D = 2, 3, 4

x = torch.randn(B, T, D)

print(x)
print()
print(torch.transpose(x, 1, 2))

# %%

import torch

x = torch.zeros(4, 3)
print(x)
x[1:, :] = torch.ones(1, 3)
print(x)

#%%

import time
import progressbar

with progressbar.ProgressBar(max_value=10) as bar:
    for i in range(10):
        time.sleep(0.1)
        time.sleep(1)
        bar.update(i)

#%%

# import time
# import progressbar
# 
# bar = progressbar.ProgressBar(max_value=10)
# for i in range(10):
#     time.sleep(0.1)
#     print(f'{i=}')
#     bar.update(i)

#%%

from tqdm import tqdm
import time

with tqdm(total=10) as bar:
    for i in range(10):
        # time.sleep(0.1)
        time.sleep(1)
        print(f'{i=}')
        bar.update(i)


#%%

from tqdm import tqdm
import time

for i in tqdm(range(10)):
    # time.sleep(0.1)
    time.sleep(5)
    print(f'{i=}')

#%%

# progress bar 2 with it per second: https://github.com/WoLpH/python-progressbar/issues/250

import time
import progressbar

with progressbar.ProgressBar(max_value=10, unit='it') as bar:
    for i in range(10):
        time.sleep(0.1)
        time.sleep(1)
        bar.update(i)

#%%

# conda install -y pytorch-geometric -c rusty1s -c conda-forge

import torch
from torch_geometric.data import Data

# [2, number_edges], edge = (node_idx1, node_idx2), e.g. e = (0,1) = (n0, n1) (note this is reflected on the type torch long)

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

# features to each node [num_nodes, D]

x = torch.tensor([[0.0], [-1.0], [1.0]])

data = Data(x=x, edge_index=edge_index)
print(data)

# https://discuss.pytorch.org/t/pytorch-geometric/44994
# https://stackoverflow.com/questions/61274847/how-to-visualize-a-torch-geometric-graph-in-python

import networkx as nx
from torch_geometric.utils.convert import to_networkx

g = to_networkx(data)
nx.draw(g)

#%%

#%%

# import time
# import progressbar
# 
# with progressbar.ProgressBar(max_value=10) as bar:
#     for i in range(10):
#         time.sleep(0.1)
#         time.sleep(1)
#         bar.update(i)

import time
import progressbar

bar = progressbar.ProgressBar(max_value=5)
for i in range(5):
    time.sleep(1)
    bar.update(i)

"""
 80% (4 of 5) |####################      | Elapsed Time: 0:00:04 ETA:   0:00:01
"""

#%%

import time
import progressbar

widgets = [
    progressbar.Percentage(),
    progressbar.Bar(),
    ' ', progressbar.SimpleProgress(),
    ' ', progressbar.ETA(),
    ' ', progressbar.AdaptiveTransferSpeed(unit='it'),
]

bar = progressbar.ProgressBar(widgets=widgets)
for i in bar(range(100)):
    time.sleep(0.2)
    bar.update(i)
"""
19%|##########                                           | 19 of 100 ETA:   0:00:17   4.9 it/s
"""

#%%

"""
from default
 99% (9998 of 10000) |########## | Elapsed Time: 1 day, 16:35:09 ETA:   0:00:26
"""

import time
import progressbar

widgets = [
    progressbar.Percentage(),
    ' ', progressbar.SimpleProgress(format=f'({progressbar.SimpleProgress.DEFAULT_FORMAT})'),
    ' ', progressbar.Bar(),
    ' ', progressbar.Timer(), ' |',
    ' ', progressbar.ETA(), ' |',
    ' ', progressbar.AdaptiveTransferSpeed(unit='it'),
]

bar = progressbar.ProgressBar(widgets=widgets)
for i in bar(range(100)):
    time.sleep(0.1)
    bar.update(i)

#%%

import uutils

def test_good_progressbar():
    import time
    bar = uutils.get_good_progressbar()
    for i in bar(range(100)):
        time.sleep(0.1)
        bar.update(i)

    print('---- start context manager test ---')
    max_value = 10
    with uutils.get_good_progressbar(max_value=max_value) as bar:
        for i in range(max_value):
            time.sleep(1)
            bar.update(i)

test_good_progressbar()

#%%

import time
import progressbar

# bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
bar = uutils.get_good_progressbar(max_value=progressbar.UnknownLength)
for i in range(20):
    time.sleep(0.1)
    bar.update(i)

#%%

import torch
import transformers
from transformers.optimization import Adafactor, AdafactorSchedule

model = torch.nn.Linear(1, 1)
optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)