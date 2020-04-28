import time
import numpy as np
import torch

from torch.autograd import Variable

import data_utils
import utils

from math import inf
import os
from maps import NamedDict

#from good_minima_discriminator import divide_params_by
import nn_models as nn_mdls

from pdb import set_trace as st

def get_norm(net,p=2):
    w_norms = 0
    for index, W in enumerate(net.parameters()):
        w_norms += W.norm(p)
    return w_norms

def divide_params_by(W,net):
    '''
        W: make sure W is non-trainable if you wish to divide by a constant.
    '''
    params = net.named_parameters()
    dict_params = dict(params)
    for name, param in dict_params.items():
        if name in dict_params:
            new_param = param/W
            dict_params[name] = new_param
    net.load_state_dict(dict_params)
    return net

def dont_train(net):
    '''
    set training parameters to false.
    :param net:
    :return:
    '''
    for param in net.parameters():
        param.requires_grad = False
    return net

def initialize_to_zero(net):
    '''
    sets weights of net to zero.
    '''
    for param in net.parameters():
        #st()
        param.zero_()

def evalaute_running_mdl_data_set(loss,error,net,dataloader,device,iterations=inf):
    '''
    Evaluate the approx (batch) error of the model under some loss and error with a specific data set.
    The batch error is an approximation of the train error (empirical error), so it computes average batch size error
    over all the batches of a specific size. Specifically it computes:
    avg_L = 1/N_B sum^{N_B}_{i=1} (1/B sum_{j \in B_i} l_j )
    which is the average batch loss over N_B = ceiling(# data points/ B )
    '''
    running_loss,running_error = 0,0
    with torch.no_grad():
        for i,(inputs,targets) in enumerate(dataloader):
            if i >= iterations:
                break
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            running_loss += loss(outputs,targets).item()
            running_error += error(outputs,targets).item()
    return running_loss/(i+1),running_error/(i+1)

def evalaute_mdl_on_full_data_set(loss,error,net,dataloader,device,iterations=inf):
    '''
    Evaluate the error of the model under some loss and error with a specific data set, but use the full data set.
    Note: this method is exact.
    '''
    N = len(dataloader.dataset)
    avg_loss,avg_error = 0,0
    with torch.no_grad():
        for i,(inputs,targets) in enumerate(dataloader):
            batch_size = targets.size()[0]
            if i >= iterations:
                n_total = batch_size*i
                avg_loss = (N/batch_size)*avg_loss
                avg_error = (N/batch_size)*avg_loss
                return avg_loss/n_total,avg_error/n_total
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            avg_loss += (batch_size/N)*loss(outputs,targets).item()
            avg_error += (batch_size/N)*error(outputs,targets).item()
    return avg_loss,avg_error

def collect_hist(net,dataloader,device):
    '''
    Collect histogram of activations of last layer.
    '''
    N = len(dataloader.dataset)
    ''' '''
    hist = np.zeros( (N,10) ) ## TODO fix hack, don't hardcode # of classes
    j = 0
    with torch.no_grad():
        for i,(inputs,targets) in enumerate(dataloader):
            batch_size = targets.size()[0]
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            hist[j:j+batch_size,:] = outputs.cpu().numpy()
            j += batch_size
    return hist

def get_function_evaluation_from_name(name):
    if name == 'evalaute_running_mdl_data_set':
        evalaute_mdl_data_set = evalaute_running_mdl_data_set
    elif name == 'evalaute_mdl_on_full_data_set':
        evalaute_mdl_data_set = evalaute_mdl_on_full_data_set
    else:
        return None
    return evalaute_mdl_data_set

class Trainer:

    def __init__(self,trainloader,testloader, optimizer, scheduler, criterion,error_criterion, stats_collector, device,
                 expt_path='',net_file_name='',all_nets_folder='',save_every_epoch=False, evalaute_mdl_data_set='evalaute_running_mdl_data_set',
                 reg_param=0.0,p=2):
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.error_criterion = error_criterion
        self.stats_collector = stats_collector
        self.device = device
        self.reg_param = reg_param
        self.p = p
        ''' '''
        self.stats_collector.save_every_epoch = save_every_epoch
        ''' save all models during training '''
        self.save_every_epoch = save_every_epoch
        self.expt_path = expt_path
        self.net_file_name = net_file_name
        ## if we need to save all nets at every epochs
        if self.save_every_epoch:
            ## and the paths and files are actually passed by user (note '' == sort of None, or user didn't set them)
            if self.expt_path != '' and self.net_file_name != '':
                self.all_nets_path = os.path.join(expt_path, all_nets_folder) #expt_path/all_nets_folder
                utils.make_and_check_dir(self.all_nets_path)
        ''' '''
        self.evalaute_mdl_data_set = get_function_evaluation_from_name(evalaute_mdl_data_set)
        if evalaute_mdl_data_set is None:
            raise ValueError(f'Data set function evaluator evalaute_mdl_data_set={evalaute_mdl_data_set} is not defined.')

    def train_and_track_stats(self,net, nb_epochs,iterations=inf,target_train_loss=inf,precision=0.10**-7):
        '''
        train net with nb_epochs and 1 epoch only # iterations = iterations
        '''
        ''' Add stats before training '''
        train_loss_epoch, train_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.trainloader, self.device, iterations)
        test_loss_epoch, test_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.testloader, self.device, iterations)
        self.stats_collector.collect_mdl_params_stats(net)
        self.stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
        print(f'[-1, -1], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
        ''' perhaps save net @ epoch '''
        self.perhaps_save(net,epoch=0)
        ''' Start training '''
        print('about to start training')
        for epoch in range(nb_epochs):  # loop over the dataset multiple times
            self.scheduler.step()
            net.train()
            running_train_loss,running_train_error = 0.0, 0.0
            for i,(inputs,targets) in enumerate(self.trainloader):
                ''' zero the parameter gradients '''
                self.optimizer.zero_grad()
                ''' train step = forward + backward + optimize '''
                inputs,targets = inputs.to(self.device),targets.to(self.device)
                outputs = net(inputs)
                #st()
                #loss = self.criterion(outputs, targets)
                loss = self.criterion(outputs,targets) + self.reg_param*get_norm(net,p=self.p)**2
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
                running_train_error += self.error_criterion(outputs,targets)
                ''' print error first iteration'''
                #if i == 0 and epoch == 0: # print on the first iteration
                #    print(data_train[0].data)
            ''' End of Epoch: evaluate nets on data '''
            net.eval()
            if self.evalaute_mdl_data_set.__name__ == 'evalaute_running_mdl_data_set':
                train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
            else:
                train_loss_epoch, train_error_epoch = self.evalaute_mdl_data_set(self.criterion, self.error_criterion, net, self.trainloader, self.device,iterations)
            test_loss_epoch, test_error_epoch = self.evalaute_mdl_data_set(self.criterion,self.error_criterion,net,self.testloader,self.device,iterations)
            ''' collect results at the end of epoch'''
            self.stats_collector.collect_mdl_params_stats(net)
            self.stats_collector.append_losses_errors_accs(train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch)
            print(f'[{epoch}, {i+1}], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
            ''' perhaps save net @ epoch '''
            self.perhaps_save(net,epoch=epoch)
            ''' check target loss '''
            if abs(train_loss_epoch - target_train_loss) < precision:
                return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch
        return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch

    def perhaps_save(self,net,epoch):
        ''' save net model '''
        if self.save_every_epoch:
            epoch_net_file_name = f'{self.net_file_name}_epoch_{epoch}'
            net_path_to_filename = os.path.join(self.all_nets_path,epoch_net_file_name)
            torch.save(net, net_path_to_filename)