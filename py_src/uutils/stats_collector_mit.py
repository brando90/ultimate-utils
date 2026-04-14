# https://github.com/brando90/overparam_experiments

from math import inf

import numpy as np

try:
    from maps import NamedDict  # don't remove this
except ImportError:
    class NamedDict(dict):
        """Fallback attribute-accessible dict used by legacy callers."""

        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__


def _to_float(value):
    return value.item() if hasattr(value, 'item') else float(value)


def _is_nan(value) -> bool:
    return not np.isfinite(_to_float(value))


def evalaute_running_mdl_data_set(loss, error, net, dataloader, device, iterations=inf):
    """Evaluate average batch loss/error over at most ``iterations`` batches."""
    import torch

    running_loss, running_error = 0.0, 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            if batch_index >= iterations:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            running_loss += _to_float(loss(outputs, targets))
            running_error += _to_float(error(outputs, targets))
            num_batches += 1
    if num_batches == 0:
        raise ValueError('dataloader produced no batches to evaluate')
    return running_loss / num_batches, running_error / num_batches


def evalaute_mdl_on_full_data_set(loss, error, net, dataloader, device, iterations=inf):
    """Evaluate exact dataset loss/error over at most ``iterations`` batches."""
    import torch

    dataset_size = len(dataloader.dataset)
    avg_loss, avg_error = 0.0, 0.0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            batch_size = targets.size()[0]
            if batch_index >= iterations:
                n_total = batch_size * batch_index
                if n_total == 0:
                    raise ValueError('dataloader produced no batches to evaluate')
                avg_loss = (dataset_size / batch_size) * avg_loss
                avg_error = (dataset_size / batch_size) * avg_error
                return avg_loss / n_total, avg_error / n_total
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            avg_loss += (batch_size / dataset_size) * _to_float(loss(outputs, targets))
            avg_error += (batch_size / dataset_size) * _to_float(error(outputs, targets))
    return avg_loss, avg_error


def get_function_evaluation_from_name(name):
    if callable(name):
        return name
    if name == 'evalaute_running_mdl_data_set':
        return evalaute_running_mdl_data_set
    if name == 'evalaute_mdl_on_full_data_set':
        return evalaute_mdl_on_full_data_set
    return None

class StatsCollector:
    '''
    Class that has all the stats collected during training.
    '''
    def __init__(self,net,trials=1,epochs=0,save_every_epoch=False,evalaute_mdl_data_set='evalaute_running_mdl_data_set'):
        self.save_every_epoch = save_every_epoch
        ''' loss & errors lists'''
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_errors, self.val_errors, self.test_errors = [], [], []
        self.train_accs, self.val_accs, self.test_accs = [], [], []
        ''' stats related to parameters'''
        nb_param_groups = len( list(net.parameters()) )
        self.grads = [ [] for i in range(nb_param_groups) ]
        self.w_norms = [ [] for i in range(nb_param_groups) ]
        self.perturbations_norms = [ [] for i in range(nb_param_groups) ]
        ''' reference net errors '''
        self.ref_train_losses, self.ref_val_losses, self.ref_test_losses = -1, -1, -1
        self.ref_train_errors, self.ref_val_errors, self.ref_test_errors = -1, -1, -1
        self.ref_train_accs, self.ref_val_accs, self.ref_test_accs = -1, -1, -1
        ''' '''
        self.rs = []
        ''' '''
        D=(trials,epochs)
        self.all_train_losses, self.all_val_losses, self.all_test_losses = np.zeros(D), np.zeros(D), np.zeros(D)
        self.all_train_errors, self.all_val_errors, self.all_test_errors = np.zeros(D), np.zeros(D), np.zeros(D)
        self.all_train_accs, self.all_val_accs, self.all_test_accs = np.zeros(D), np.zeros(D), np.zeros(D)
        ''' '''
        self.random_dirs = []
        ''' '''
        if self.save_every_epoch:
            self.weights = np.zeros(epochs,)
        ''' '''
        evalaute_mdl_data_set = get_function_evaluation_from_name(evalaute_mdl_data_set)
        if evalaute_mdl_data_set is None:
            raise ValueError(f'Data set function evaluator evalaute_mdl_data_set={evalaute_mdl_data_set} is not defined.')
        else:
            self.evalaute_mdl_data_set = evalaute_mdl_data_set

    def collect_mdl_params_stats(self,mdl):
        '''
            log parameter stats
            Note: for each time this function is called, it appends the stats once. If it goes through each list and
            append each time it means it extends the list each time it's called. If this function its called each time
            at the end of every epoch it means that each list index will correspond to some value at some epoch depending
            at that index.
        '''
        for index, W in enumerate(mdl.parameters()):
            self.w_norms[index].append( W.data.norm(2) )
            if W.grad is not None:
                grad_norm = W.grad.data.norm(2)
                self.grads[index].append(grad_norm)
                if _is_nan(grad_norm):
                    raise ValueError(f'NaN detected in gradient norm for parameter group {index}')

    def append_losses_errors_accs(self,train_loss, train_error, test_loss, test_error):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_errors.append(train_error)
        self.test_errors.append(test_error)
        self.train_accs.append(1.0-train_error)
        self.test_accs.append(1.0-test_error)

    def add_perturbation_norms_from_perturbations(self,net,perturbations):
        for index, W in enumerate(net.parameters()):
            self.perturbations_norms[index].append( perturbations[index].norm(2) )

    def record_errors_loss_reference_net(self, criterion, error_criterion, net, trainloader, testloader, device):
        train_loss, train_error = self.evalaute_mdl_data_set(criterion, error_criterion, net, trainloader, device)
        test_loss, test_error = self.evalaute_mdl_data_set(criterion, error_criterion, net, testloader, device)
        self.ref_train_losses, self.ref_test_losses = train_loss, test_loss
        self.ref_train_errors, self.ref_test_errors = train_error, test_error
        self.ref_train_accs, self.ref_test_accs = 1.0 - train_error, 1.0 - test_error

    def append_all_losses_errors_accs(self,dir_index,epoch,errors_losses):
        (Er_train_loss,Er_train_error,Er_test_loss,Er_test_error) = errors_losses
        self.all_train_losses[dir_index,epoch] = Er_train_loss
        self.all_test_losses[dir_index,epoch] = Er_test_loss
        self.all_train_errors[dir_index,epoch] = Er_train_error
        self.all_test_errors[dir_index,epoch] = Er_test_error
        self.all_train_accs[dir_index,epoch] = 1.0 - Er_train_error
        self.all_test_accs[dir_index,epoch] = 1.0 - Er_test_error

    def get_stats_dict(self):
        ## TODO: loop through fields?
        stats = NamedDict(
            train_losses=self.train_losses,val_losses=self.val_losses,test_losses=self.test_losses,
            train_errors=self.train_errors,val_errors=self.val_errors,test_errors=self.test_errors,
            train_accs=self.train_accs,val_accs=self.val_accs,test_accs=self.test_accs,
            grads=self.grads,
            w_norms=self.w_norms,
            perturbations_norms=self.perturbations_norms,
            ref_train_losses=self.ref_train_losses,ref_val_losses=self.ref_val_losses,ref_test_losses=self.ref_test_losses,
            ref_train_errors=self.ref_train_errors, ref_val_errors=self.ref_val_errors,ref_test_errors=self.ref_test_errors,
            ref_train_accs=self.ref_train_accs,ref_val_accs=self.ref_val_accs,ref_test_accs=self.ref_test_accs,
            all_train_losses=self.all_train_losses,all_val_losses=self.all_val_losses,all_test_losses=self.all_test_losses,
            all_train_errors=self.all_train_errors,all_val_errors=self.all_val_errors,all_test_errors=self.all_test_errors,
            all_train_accs=self.all_train_accs,all_val_accs=self.all_val_accs,all_test_accs=self.all_test_accs,
            rs=self.rs,random_dirs=self.random_dirs
        )
        return stats
