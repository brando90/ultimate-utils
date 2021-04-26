# https://github.com/brando90/overparam_experiments

import numpy as np

from maps import NamedDict # don't remove this

#from new_training_algorithms import get_function_evaluation_from_name
#from new_training_algorithms import evalaute_running_mdl_data_set
#import nn_models as nn_mdls

#import utils

#from pdb import set_trace as st

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
        self.all_train_losses, self.all_val_losses, self.all_test_losses = np.zeros(D), [],  np.zeros(D)
        self.all_train_errors, self.all_val_errors, self.all_test_errors =  np.zeros(D), [],  np.zeros(D)
        self.all_train_accs, self.all_val_accs, self.all_test_accs = np.zeros(D), [],  np.zeros(D)
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
                self.grads[index].append( W.grad.data.norm(2) )
                if utils.is_NaN(W.grad.data.norm(2)):
                    raise ValueError(f'Nan Detected error happened at: i={i} loss_val={loss_val}, loss={loss}')

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

    def record_errors_loss_reference_net(self,criterion,error_criterion,net,trainloader,testloader,enable_cuda):
        train_loss, train_error = self.evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda)
        test_loss, test_error = self.evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda)
        self.ref_train_losses, self.ref_test_losses = train_loss, train_error
        self.ref_train_accs, self.ref_test_accs = test_loss, test_error

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