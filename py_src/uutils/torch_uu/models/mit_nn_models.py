import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import unittest

import copy

from pdb import set_trace as st

class Flatten(nn.Module):
   def forward(self, x):
       N, C, H, W = x.size() # read in N, C, H, W
       return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def reset_parameters(net):
    params = net.named_parameters()
    # for name,param in params:
    #     param.reset_parameters()
    for name, param_train in params:
        layer_name, param_name = name.split('.')
        layer = getattr(net, layer_name)
        layer.reset_parameters()


def count_nb_params(net):
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    return count

class Net(nn.Module):
    # Cifar/MNIST example
    def __init__(self):
        super(Net, self).__init__()
        # in_channels = # channels from previous layer
        # out_channels = # of filters (since thats the # filters for next layer)
        # kernel_size = tuple (H,W) in pytorch
        self.conv1 = nn.Conv2d(3, 6, 5) #(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##

class MMNISTNet(nn.Module):
    def __init__(self):
        super(MMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        #return F.log_softmax(x, dim=1)

##

class AllConvNet(nn.Module):

    def __init__(self, dropout=False, nc=3, num_classes=10,only_1st_layer_bias=False,do_bn=False):
        super().__init__()
        bias = True
        self.dropout = do_bn
        self.conv1 = nn.Conv2d(nc, 96, 3, padding=1,bias=True)
        if only_1st_layer_bias:
            bias = False
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1,bias=bias)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2,bias=bias)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1,bias=bias)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1,bias=bias)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2,bias=bias)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1,bias=bias)
        self.conv8 = nn.Conv2d(192, 192, 1,bias=bias)
        self.conv9 = nn.Conv2d(192, num_classes, 1,bias=bias)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))
        class_out = F.relu(self.conv9(conv8_out))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        return pool_out

class AllConvNetStefOe(nn.Module):
    ## https://github.com/StefOe/all-conv-pytorch/blob/master/allconv.py

    def __init__(self, dropout=False, nc=3, num_classes=10,only_1st_layer_bias=False):
        super().__init__()
        bias = True
        self.dropout = dropout
        self.conv1 = nn.Conv2d(nc, 96, 3, padding=1,bias=bias)
        if only_1st_layer_bias:
            bias = False
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1,bias=bias)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2,bias=bias)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1,bias=bias)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1,bias=bias)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2,bias=bias)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1,bias=bias)
        self.conv8 = nn.Conv2d(192, 192, 1,bias=bias)
        self.conv9 = nn.Conv2d(192, num_classes, 1,bias=bias)

    def forward(self, x):
        if self.dropout:
            x = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        if self.dropout:
            conv3_out = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        if self.dropout:
            conv6_out = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.conv9(conv8_out))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        return pool_out

class GBoixNet(nn.Module):
    def __init__(self,CHW, Fs, Ks, FCs,do_bn=False,only_1st_layer_bias=False):
        super(GBoixNet, self).__init__()
        C,H,W = CHW
        self.do_bn = do_bn
        self.nb_conv_layers = len(Fs)
        ''' Initialize Conv layers '''
        layer = 0
        self.convs = []
        self.bns_convs = []
        out = Variable(torch.FloatTensor(1, C,H,W))
        in_channels = C
        for i in range(self.nb_conv_layers):
            F,K = Fs[i], Ks[i]
            bias = self._bias_flag(only_1st_layer_bias,layer)
            ##
            conv = nn.Conv2d(in_channels,F,K,bias=bias) #(in_channels, out_channels, kernel_size)
            setattr(self,f'conv{i}',conv)
            self.convs.append(conv)
            ##
            if self.do_bn:
                bn = nn.BatchNorm2d(F)
                setattr(self,f'bn2D_conv{i}',bn)
                self.bns_convs.append(bn)
            ##
            in_channels = F
            out = conv(out)
            layer+=1
        ''' Initialize FC layers'''
        self.nb_fcs_layers = len(FCs)
        ##
        self.fcs = []
        self.bns_fcs = []
        CHW = out.numel()
        in_features = CHW
        for i in range(self.nb_fcs_layers-1):
            out_features = FCs[i]
            bias = self._bias_flag(only_1st_layer_bias, i)
            ##
            fc = nn.Linear(in_features, out_features, bias=bias)
            setattr(self,f'fc{i}', fc)
            self.fcs.append(fc)
            ##
            if self.do_bn:
                print('BN_FC')
                bn_fc = nn.BatchNorm1d(out_features)
                setattr(self, f'bn1D_fc{i}', bn_fc)
                self.bns_fcs.append(bn_fc)
            ##
            in_features = out_features
            layer+=1
        ##
        i = self.nb_fcs_layers-1
        out_features = FCs[i]
        bias = self._bias_flag(only_1st_layer_bias, layer)
        fc = nn.Linear(in_features, out_features, bias=bias)
        layer+=1
        ##
        setattr(self,f'fc{i}', fc)
        self.fcs.append(fc)
        self.nb_layers = layer

    def forward(self, x):
        ''' conv layers '''
        for i in range(self.nb_conv_layers):
            conv = self.convs[i]
            ##
            z = conv(x)
            if self.do_bn:
                bn = self.bns_convs[i]
                z = bn(z)
            x = F.relu(z)
        _, C, H, W = x.size()
        ''' FC layers '''
        x = x.view(-1, C * H * W)
        for i in range(self.nb_fcs_layers-1):
            fc = self.fcs[i]
            z = fc(x)
            if self.do_bn:
                bn_fc = self.bns_fcs[i]
                z = bn_fc(z)
            x = F.relu(z)
        # last layer doesn't have a relu
        fc = self.fcs[self.nb_fcs_layers-1]
        x = fc(x)
        return x

    def _bias_flag(self,only_1st_layer_bias,i):
        '''
        We want to return always True if only_1st_layer_bias==False (since it means every layer should have a bias)
        and if only_1st_layer_bias=True then we want to return True only if i==0 (first layer)
        '''
        if not only_1st_layer_bias: # only_1st_layer_bias == False
            return True
        else: # only_1st_layer_bias == True
            return i == 0 ## True only if its the first layer

##

class BoixNet(nn.Module):
    ## The network has 2 convolutional layers followed by 3 fully connected.
    ## Use ReLUs, and no batch normalization or regularizers.
    ## Trained with cross-entropy
    ## https://discuss.pytorch.org/t/when-creating-new-neural-net-from-scratch-how-does-one-statically-define-what-the-size-of-the-a-flatten-layer-is-not-at-runtime/14235
    def __init__(self,C,H,W, nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3,do_bn=False):
        super(BoixNet, self).__init__()
        self.do_bn = do_bn
        ''' Initialize conv layers'''
        self.conv1 = nn.Conv2d(3,nb_filters1, kernel_size1) #(in_channels, out_channels, kernel_size)
        if self.do_bn: self.bn_conv1 = nn.BatchNorm2d(nb_filters1)
        self.conv2 = nn.Conv2d(nb_filters1,nb_filters2, kernel_size2)
        if self.do_bn: self.bn_conv2 = nn.BatchNorm2d(nb_filters2)
        CHW = ((H-kernel_size1+1)-kernel_size2+1) * ((W-kernel_size1+1)-kernel_size2+1) * nb_filters2
        ''' '''
        self.fc1 = nn.Linear(CHW, nb_units_fc1)
        if self.do_bn: self.fc1_bn = nn.BatchNorm1d(nb_units_fc1)
        self.fc2 = nn.Linear(nb_units_fc1,nb_units_fc2)
        if self.do_bn: self.fc2_bn = nn.BatchNorm1d(nb_units_fc2)
        self.fc3 = nn.Linear(nb_units_fc2,nb_units_fc3)
        if self.do_bn: self.fc3_bn = nn.BatchNorm1d(nb_units_fc3) #layer right before output no BN

    def forward(self, x):
        ''' conv layers'''
        pre_act1 = self.bn_conv1(self.conv1(x)) if self.do_bn else self.conv1(x)
        a_conv1 = F.relu(pre_act1)
        ##
        pre_act2 = self.bn_conv2(self.conv2(a_conv1)) if self.do_bn else self.conv2(a_conv1)
        a_conv2 = F.relu(pre_act2)
        ''' FC layers '''
        _,C,H,W = a_conv2.size()
        a_flat_conv2 = a_conv2.view(-1,C*H*W)
        ##
        pre_act_fc1 = self.fc1_bn(self.fc1(a_flat_conv2)) if self.do_bn else self.fc1(a_flat_conv2)
        a_fc1 = F.relu(pre_act_fc1)
        pre_act_fc2 = self.fc2_bn(self.fc2(a_fc1)) if self.do_bn else self.fc2(a_fc1)
        a_fc2 = F.relu(pre_act_fc2)
        pre_act_fc3 = self.fc3_bn(self.fc3(a_fc2)) if self.do_bn else self.fc3(a_fc2)
        a_fc3 = pre_act_fc3
        return a_fc3

class BoixNet2(nn.Module):
    ## The network has 2 convolutional layers followed by 3 fully connected.
    ## Use ReLUs, and no batch normalization or regularizers.
    ## Trained with cross-entropy
    ## https://discuss.pytorch.org/t/when-creating-new-neural-net-from-scratch-how-does-one-statically-define-what-the-size-of-the-a-flatten-layer-is-not-at-runtime/14235
    def __init__(self,C,H,W, nb_filters1,nb_filters2, kernel_size1,kernel_size2, nb_units_fc1,nb_units_fc2,nb_units_fc3,do_bn=False):
        super(BoixNet, self).__init__()
        self.do_bn = do_bn
        ''' Initialize conv layers'''
        self.conv1 = nn.Conv2d(3,nb_filters1, kernel_size1) #(in_channels, out_channels, kernel_size)
        if self.do_bn: self.bn_conv1 = nn.BatchNorm2d(nb_filters1)
        self.conv2 = nn.Conv2d(nb_filters1,nb_filters2, kernel_size2)
        if self.do_bn: self.bn_conv2 = nn.BatchNorm2d(nb_filters2)
        CHW = ((H-kernel_size1+1)-kernel_size2+1) * ((W-kernel_size1+1)-kernel_size2+1) * nb_filters2
        ''' '''
        self.fc1 = nn.Linear(CHW, nb_units_fc1)
        if self.do_bn: self.fc1_bn = nn.BatchNorm1d(nb_units_fc1)
        self.fc2 = nn.Linear(nb_units_fc1,nb_units_fc2)
        if self.do_bn: self.fc2_bn = nn.BatchNorm1d(nb_units_fc2)
        self.fc3 = nn.Linear(nb_units_fc2,nb_units_fc3)
        #if self.do_bn: self.fc3_bn = nn.BatchNorm1d(nb_units_fc3) #layer right before output no BN

    def forward(self, x):
        ''' conv layers'''
        pre_act1 = self.bn_conv1(self.conv1(x)) if self.do_bn else self.conv1(x)
        a_conv1 = F.relu(pre_act1)
        ##
        pre_act2 = self.bn_conv2(self.conv2(a_conv1)) if self.do_bn else self.conv2(a_conv1)
        a_conv2 = F.relu(pre_act2)
        ''' FC layers '''
        _,C,H,W = a_conv2.size()
        a_flat_conv2 = a_conv2.view(-1,C*H*W)
        ##
        pre_act_fc1 = self.fc1_bn(self.fc1(a_flat_conv2)) if self.do_bn else self.fc1(a_flat_conv2)
        a_fc1 = F.relu(pre_act_fc1)

        pre_act_fc2 = self.fc2_bn(self.fc2(a_fc1)) if self.do_bn else self.fc2(a_fc1)
        a_fc2 = F.relu(pre_act_fc2)

        #pre_act_fc3 = self.fc3_bn(self.fc3(a_fc2)) if self.do_bn else self.fc3(a_fc2) #layer right before output no BN
        pre_act_fc3 = self.fc3(a_fc2)
        a_fc3 = pre_act_fc3
        return a_fc3
##

class LiaoNet(nn.Module):
    ## 5 conv net FC
    ## TODO: use ModuleList https://discuss.pytorch.org/t/is-setattr-something-we-need-when-creating-custom-nn-with-changing-layers/14555
    def __init__(self,C,H,W, Fs, Ks, FC,do_bn=False):
        super(LiaoNet, self).__init__()
        self.do_bn = do_bn
        self.nb_conv_layers = len(Fs)
        ''' Initialize Conv layers '''
        self.convs = []
        self.bns = []
        out = Variable(torch.FloatTensor(1, C,H,W))
        in_channels = C
        for i in range(self.nb_conv_layers):
            F,K = Fs[i], Ks[i]
            conv = nn.Conv2d(in_channels,F,K) #(in_channels, out_channels, kernel_size)
            if self.do_bn:
                bn = nn.BatchNorm2d(F)
                setattr(self,f'bn{i}',bn)
                self.bns.append(bn)
            ##
            setattr(self,f'conv{i}',conv)
            self.convs.append(conv)
            ##
            in_channels = F
            out = conv(out)
        ''' Initialize FC layers'''
        CHW = out.numel()
        self.fc = nn.Linear(CHW,FC)

    def forward(self, x):
        ''' conv layers '''
        for i in range(self.nb_conv_layers):
            conv = self.convs[i]
            ##
            z = conv(x)
            if self.do_bn:
                bn = self.bns[i]
                z = bn(z)
            x = F.relu(z)
        _,C,H,W = x.size()
        x = x.view(-1,C*H*W)
        ''' FC layers '''
        x = self.fc(x)
        #if self.do_bn:
        #    x = self.bn_fc(x)
        return x

####

def get_AndyNet():
    mdl = nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(96),
        nn.Conv2d(96, 96, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(96),
        nn.Conv2d(96, 96, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(96),
        nn.Conv2d(96, 192, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(192),
        nn.Conv2d(192, 192, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(192),
        nn.Conv2d(192, 192, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(192),
        Flatten(),
        nn.Linear(3072, 10)
    )
    return mdl

####

class PertNet(nn.Module):

    def __init__(self,net):
        super(PertNet, self).__init__()
        self.original_net = net
        self.net_perturbed = copy.deepcopy(net)
        self.pert = self.init_pert( copy.deepcopy(net) )

    def init_pert(self):
        # TODO
        return pert

    def add_pert(self):
        '''
        computes NN(W+pert):
            net_perturbed.W = net_original.W + net_pert.W
        :return:
        '''
        params_original = self.original_net.named_parameters()
        params_pert = self.pert.named_parameters()
        params_pert_net = self.net_perturbed.named_parameters()
        ''' '''
        dict_pert = dict(params_pert) # <- net with perturbations!
        dict_pert_net = dict(params_pert_net) # <- net we are updating
        for param_name, W in params_original:
            if param_name in params_pert_net:
                ## W_pert <- W + pert
                pert = dict_pert[param_name]
                dict_pert_net[param_name] = W + pert
        interpolated_net.load_state_dict(dict_params_interpolated)
        return interpolated_net

class TestStringMethods(unittest.TestCase):

    def test_GBoixNet(self):
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()