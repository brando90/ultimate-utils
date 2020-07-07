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
    meta_str=''
    ## REPORT TIMES
    start_time = start
    seconds = (time.time() - start_time)
    minutes = seconds/ 60
    hours = minutes/ 60
    if verbose:
        print(f"--- {seconds} {'seconds '+meta_str} ---")
        print(f"--- {minutes} {'minutes '+meta_str} ---")
        print(f"--- {hours} {'hours '+meta_str} ---")
        print('\a')
    ##
    msg = f'time passed: hours:{hours}, minutes={minutes}, seconds={seconds}'
    return msg, seconds, minutes, hours

def params_in_comp_graph():
    import torch
    import torch.nn as nn
    from torchviz import make_dot
    fc0 = nn.Linear(in_features=3,out_features=1)
    params = [('fc0', fc0)]
    mdl = nn.Sequential(OrderedDict(params))

    x = torch.randn(1,3)
    #x.requires_grad = True  # uncomment to put in computation graph
    y = torch.randn(1)

    l = ( mdl(x) - y )**2

    #make_dot(l, params=dict(mdl.named_parameters()))
    params = dict(mdl.named_parameters())
    #params = {**params, 'x':x}
    make_dot(l,params=params).render('data/debug/test_img_l',format='png')

def check_if_tensor_is_detached():
    a = torch.tensor([2.0], requires_grad=True)
    b = a.detach()
    b.requires_grad = True
    print(a == b)
    print(a is b)
    print(a)
    print(b)

    la = (5.0 - a)**2
    la.backward()
    print(f'a.grad = {a.grad}')

    lb = (6.0 - b)**2
    lb.backward()
    print(f'b.grad = {b.grad}')

def deep_copy_issue():
    params = OrderedDict( [('fc1',nn.Linear(in_features=3,out_features=1))] )
    mdl0 = nn.Sequential(params)
    mdl1 = copy.deepcopy(mdl0)
    print(id(mdl0))
    print(mdl0)
    print(id(mdl1))
    print(mdl1)
    # my update
    mdl1.fc1.weight = nn.Parameter( mdl1.fc1.weight + 1 )
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
    #url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename = 'miniImagenet.tgz'
    root = '~/tmp/' # dir to place downloaded file in
    download_file_from_google_drive(file_id, root, filename)

def extract():
    from torchvision.datasets.utils import extract_archive
    from_path = os.path.expanduser('~/Downloads/miniImagenet.tgz')
    extract_archive(from_path)

def download_and_extract_miniImagenet(root):
    import os
    from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

    ## download miniImagenet
    #url = 'https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    file_id = '1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR'
    filename = 'miniImagenet.tgz'
    download_file_from_google_drive(file_id, root, filename)
    fpath = os.path.join(root, filename) # this is what download_file_from_google_drive does
    ## extract downloaded dataset
    from_path = os.path.expanduser(fpath)
    extract_archive(from_path)
    ## remove the zip file
    os.remove(from_path)
    
def torch_concat():
    import torch
    
    g1 = torch.randn(3,3)
    g2 = torch.randn(3,3)

def inner_loop1():
    n_inner_iter = 5
    inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

    qry_losses = []
    qry_accs = []
    meta_opt.zero_grad()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, copy_initial_weights=False
        ) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(n_inner_iter):
                spt_logits = fnet(x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                diffopt.step(spt_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            qry_logits = fnet(x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_losses.append(qry_loss.detach())
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            qry_loss.backward()

    meta_opt.step()
    qry_losses = sum(qry_losses) / task_num
    qry_accs = 100. * sum(qry_accs) / task_num
    i = epoch + float(batch_idx) / n_train_iter
    iter_time = time.time() - start_time

def inner_loop2():
    n_inner_iter = 5
    inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

    qry_losses = []
    qry_accs = []
    meta_opt.zero_grad()
    meta_loss = 0
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, copy_initial_weights=False
        ) as (fnet, diffopt):
            # Optimize the likelihood of the support set by taking
            # gradient steps w.r.t. the model's parameters.
            # This adapts the model's meta-parameters to the task.
            # higher is able to automatically keep copies of
            # your network's parameters as they are being updated.
            for _ in range(n_inner_iter):
                spt_logits = fnet(x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                diffopt.step(spt_loss)

            # The final set of adapted parameters will induce some
            # final loss and accuracy on the query dataset.
            # These will be used to update the model's meta-parameters.
            qry_logits = fnet(x_qry[i])
            qry_loss = F.cross_entropy(qry_logits, y_qry[i])
            qry_losses.append(qry_loss.detach())
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry[i]).sum().item() / querysz
            qry_accs.append(qry_acc)

            # Update the model's meta-parameters to optimize the query
            # losses across all of the tasks sampled in this batch.
            # This unrolls through the gradient steps.
            #qry_loss.backward()
            meta_loss += qry_loss

    qry_losses = sum(qry_losses) / task_num
    qry_losses.backward()
    meta_opt.step()
    qry_accs = 100. * sum(qry_accs) / task_num
    i = epoch + float(batch_idx) / n_train_iter
    iter_time = time.time() - start_time

def error_unexpected_way_to_by_pass_safety():
    # https://stackoverflow.com/questions/62415251/why-am-i-able-to-change-the-value-of-a-tensor-without-the-computation-graph-know

    import torch 
    a = torch.tensor([1,2,3.], requires_grad=True)
    # are detached tensor's leafs? yes they are
    a_detached = a.detach()
    #a.fill_(2) # illegal, warns you that a tensor which requires grads is used in an inplace op (so it won't be recorded in computation graph so it wont take the right derivative of the forward path as this op won't be in it)
    a_detached.fill_(2) # weird that this one is allowed, seems to allow me to bypass the error check from the previous comment...?!
    print(f'a = {a}')
    print(f'a_detached = {a_detached}')
    a.sum().backward()

def detach_playground():
    import torch

    a = torch.tensor([1,2,3.], requires_grad=True)
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
    #a.fill_(2) # illegal, warns you that a tensor which requires grads is used in an inplace op (so it won't be recorded in computation graph so it wont take the right derivative of the forward path as this op won't be in it)
    a_detached.fill_(2) # weird that this one is allowed, seems to allow me to bypass the error check from the previous comment...?!
    print(f'a = {a}')
    print(f'a_detached = {a_detached}')
    ## conclusion: detach basically creates a totally new tensor which cuts gradient computations to the original but shares the same memory with original
    out = a.sigmoid()
    out_detached = out.detach()
    out_detached.zero_()
    out.sum().backward()

def clone_playground():
    import torch

    a = torch.tensor([1,2,3.], requires_grad=True)
    a_clone = a.clone()
    print(f'a_clone.is_leaf = {a_clone.is_leaf}')
    print(f'a is a_clone = {a is a_clone}')
    print(f'a == a_clone = {a == a_clone}')
    print(f'a = {a}')
    print(f'a_clone = {a_clone}')
    #a_clone.fill_(2)
    a_clone.mul_(2)
    print(f'a = {a}')
    print(f'a_clone = {a_clone}')
    a_clone.sum().backward()
    print(f'a.grad = {a.grad}')

def clone_vs_deepcopy():
    import copy
    import torch

    x = torch.tensor([1,2,3.])
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

    x = torch.tensor([1,2,3.], requires_grad=True)
    y = x + 1
    print(f'x.is_leaf = {x.is_leaf}')
    print(f'y.is_leaf = {y.is_leaf}')
    x += 1 # not allowed because x is a leaf, since changing the value of a leaf with an inplace forgets it's value then backward wouldn't work IMO (though its not the official response)
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
    loops = 5 # how many iterations in the inner loop we want to do

    x = torch.tensor(np.random.random((N,1)), dtype=torch.float64) # features for inner training loop
    y = x * actual_multiplier # target for inner training loop
    model = nn.Linear(1, 1, bias=False).double() # simplest possible model - multiple input x by weight w without bias
    meta_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)


    def run_inner_loop_once(model, verbose, copy_initial_weights):
        lr_tensor = torch.tensor([0.3], requires_grad=True)
        momentum_tensor = torch.tensor([0.5], requires_grad=True)
        opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
        with higher.innerloop_ctx(model, opt, copy_initial_weights=copy_initial_weights, override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
            for j in range(loops):
                if verbose:
                    print('Starting inner loop step j=={0}'.format(j))
                    print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                    print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
                out = fmodel(x)
                if verbose:
                    print('    Notice how `out` is `x` multiplied by the latest version of weight: {0:.4} * {1:.4} == {2:.4}'.format(x[0,0].item(), list(fmodel.parameters())[0].item(), out[0].item()))
                loss = ((out - y)**2).mean()
                diffopt.step(loss)

            if verbose:
                # after all inner training let's see all steps' parameter tensors
                print()
                print("Let's print all intermediate parameters versions after inner loop is done:")
                for j in range(loops+1):
                    print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                print()

            # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
            weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
            meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier)**2
            print('  Final meta-loss: {0}'.format(meta_loss.item()))
            meta_loss.backward() # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
            if verbose:
                print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad, momentum_tensor.grad))
                print('  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
            return meta_loss.item()

    print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
    meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
    meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
    final_meta_gradient = list(model.parameters())[0].grad.item()
    # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
    # a bunch of hand-wavy statements and difficult to read code.
    # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
    meta_opt.step()
    meta_opt.zero_grad()
    meta_step = - meta_lr * final_meta_gradient # how much meta_opt actually shifted inital weight value
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
    actual_multiplier = 3.5 # the parameters we want the model to learn
    meta_lr = 0.00001
    loops = 5 # how many iterations in the inner loop we want to do

    x = torch.randn(N,1) # features for inner training loop
    y = x * actual_multiplier # target for inner training loop
    model = nn.Linear(1, 1, bias=False)# model(x) = w*x, simplest possible model - multiple input x by weight w without bias. goal is to w~~actualy_multiplier
    outer_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)

    def run_inner_loop_once(model, verbose, copy_initial_weights):
        lr_tensor = torch.tensor([0.3], requires_grad=True)
        momentum_tensor = torch.tensor([0.5], requires_grad=True)
        inner_opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=copy_initial_weights, override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
            for j in range(loops):
                if verbose:
                    print('Starting inner loop step j=={0}'.format(j))
                    print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                    print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
                out = fmodel(x)
                if verbose:
                    print(f'    Notice how `out` is `x` multiplied by the latest version of weight: {x[0,0].item()} * {list(fmodel.parameters())[0].item()} == {out[0].item()}')
                loss = ((out - y)**2).mean()
                diffopt.step(loss)

            if verbose:
                # after all inner training let's see all steps' parameter tensors
                print()
                print("Let's print all intermediate parameters versions after inner loop is done:")
                for j in range(loops+1):
                    print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                print()

            # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
            weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
            meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier)**2
            print('  Final meta-loss: {0}'.format(meta_loss.item()))
            meta_loss.backward() # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
            if verbose:
                print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad, momentum_tensor.grad))
                print('  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
            return meta_loss.item()

    print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
    meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
    meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
    print("\nLet's see if we got any gradient for initial model parameters: {0}\n".format(list(model.parameters())[0].grad))

    print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
    final_meta_gradient = list(model.parameters())[0].grad.item()
    # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
    # a bunch of hand-wavy statements and difficult to read code.
    # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
    outer_opt.step()
    outer_opt.zero_grad()
    meta_step = - meta_lr * final_meta_gradient # how much meta_opt actually shifted inital weight value
    meta_loss_val3 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)

    meta_loss_gradient_approximation = (meta_loss_val3 - meta_loss_val2) / meta_step

    print()
    print('Side-by-side meta_loss_gradient_approximation and gradient computed by `higher` lib: {0:.4} VS {1:.4}'.format(meta_loss_gradient_approximation, final_meta_gradient))

def tqdm_torchmeta():
    from torchvision.transforms import Compose, Resize, ToTensor

    import torchmeta
    from torchmeta.datasets.helpers import miniimagenet

    from pathlib import Path
    from types import SimpleNamespace

    from tqdm import tqdm

    ## get args
    args = SimpleNamespace(episodes=5,n_classes=5,k_shot=5,k_eval=15,meta_batch_size=1,n_workers=4)
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

if __name__ == "__main__":
    start = time.time()
    print('pytorch playground!')
    # params_in_comp_graph()
    #check_if_tensor_is_detached()
    #deep_copy_issue()
    #download_mini_imagenet()
    #extract()
    #download_and_extract_miniImagenet(root='~/tmp')
    #download_and_extract_miniImagenet(root='~/automl-meta-learning/data')
    #torch_concat()
    #detach_vs_cloe()
    #error_unexpected_way_to_by_pass_safety()
    #clone_playground()
    #inplace_playground()
    #clone_vs_deepcopy()
    #copy_initial_weights_playground()
    tqdm_torchmeta()
    print('--> DONE')
    time_passed_msg, _, _, _ = report_times(start)
    print(f'--> {time_passed_msg}')