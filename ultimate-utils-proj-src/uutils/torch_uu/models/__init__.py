from collections import OrderedDict
from typing import Optional

import torch
from torch import nn


def get_linear_model(in_features: int, out_features: int) -> nn.Module:
    return nn.Linear(in_features=in_features, out_features=out_features)


def get_named_one_layer_linear_model(Din: int, Dout: int, bias: bool = True) -> nn.Module:
    """
    Returns a linear Fully Connected Neural Network (FCNN) with random weights and only layer named `fc0`.

    Note: biases are also initialized randomly if true, via: init.uniform_(self.bias, -bound, bound).
    """
    from collections import OrderedDict
    params = OrderedDict([('fc0', nn.Linear(in_features=Din, out_features=Dout, bias=bias))])
    mdl = nn.Sequential(params)
    return mdl


def get_named_one_layer_random_linear_model(Din: int, Dout: int, bias: bool = True) -> nn.Module:
    """
    Returns a random linear Fully Connected Neural Network (FCNN) with random weights and only layer named `fc0`.
    """
    return get_named_one_layer_linear_model(Din, Dout, bias)


def get_named_identity_one_layer_linear_model(D: int, debug: bool = False) -> nn.Module:
    """
    Returns a model with identity matrix as weights and no biases.

    Useful for testing anatome's cca so that it returns the pure data matrix.
    """
    from collections import OrderedDict
    params = OrderedDict([('fc0', nn.Linear(in_features=D, out_features=D, bias=False))])
    mdl: nn.Module = nn.Sequential(params)
    mdl.fc0.weight = nn.Parameter(torch.diag(torch.ones(D)))
    if debug:
        print(f'You should see the identity matrix:\n{mdl.fc0.weight=}')
    return mdl


def hardcoded_3_layer_model(in_features: int, out_features: int) -> nn.Module:
    """
    Returns a nn sequential model with 3 layers (2 hidden and 1 output layers).
    ReLU activation. Final layer are the raw scores (so final layer is a linear layer).

    """
    from collections import OrderedDict
    hidden_features = in_features
    modules = OrderedDict([
        ('fc0', nn.Linear(in_features=in_features, out_features=hidden_features)),
        ('ReLU0', nn.ReLU()),
        ('fc1', nn.Linear(in_features=hidden_features, out_features=hidden_features)),
        ('ReLU2', nn.ReLU()),
        ('fc2', nn.Linear(in_features=hidden_features, out_features=out_features))
    ])
    mdl = nn.Sequential(modules)
    return mdl


def get_simple_model(in_features: int, hidden_features: int, out_features: int, num_layer: int = 2):
    """
    Note: num_layers is defined as hidden + outer layer (even if the last has an identity function). So each set of
    activations is a "layer".
    Note: usually the last layer is the scores even for classification so no need to add softmax usually if you use CE
    pytorch loss.

    :param in_features:
    :param hidden_features:
    :param out_features:
    :param num_layer:
    :return:
    """
    assert num_layer >= 1
    if num_layer == 1:
        modules = OrderedDict([
            ('fc1', nn.Linear(in_features=hidden_features, out_features=out_features))
        ])
    elif num_layer == 2:
        # for clarity
        modules = OrderedDict([
            ('fc0', nn.Linear(in_features=in_features, out_features=hidden_features)),
            ('ReLU0', nn.ReLU()),
            ('fc1', nn.Linear(in_features=hidden_features, out_features=out_features))
        ])
    else:
        modules = OrderedDict([])
        for l in range(num_layer - 1):
            if l < num_layer:
                # append one fc + relu
                modules.append((f'fc{l}', nn.Linear(in_features=in_features, out_features=hidden_features)))
                modules.append((f'ReLU{l}', nn.ReLU()))
            else:
                modules.append((f'fc{l}', nn.Linear(in_features=in_features, out_features=hidden_features)))
    mdl = nn.Sequential(modules)
    return mdl


def get_single_conv_model(in_channels: int, num_out_filters: int, kernel_size: int = 3, padding: int = 1) -> nn.Module:
    """
    Gives a conv layer with in_channels as input.
    Default gives a filter size such that the spatial dimensions (H, W) do not change i.e. the output tensor after
    passing data in of this convolution is [B, new_filter_size, H, W] (so same H and W).

    Note:
        - H' = (H + 2*padding - kernel_size.H + 1) if padding = 1, kernel_size.H = 3 we have:
            H + 2 - 3 + 1 = H, which retains the size.
    :param in_channels: e.g. 3 for the the input being the raw image.
    :param num_out_filters:
    :param kernel_size:
    :param padding:
    :return:
    """
    conv_layer: nn.Module = nn.Conv2d(in_channels=in_channels,
                                      out_channels=num_out_filters,
                                      kernel_size=kernel_size,
                                      padding=padding)
    return conv_layer


def get_5cnn_model(image_size: int = 84,
                   bn_eps: float = 1e-3,
                   bn_momentum: float = 0.95,
                   n_classes: int = 5,
                   filter_size: int = 32,
                   levels: Optional = None,
                   spp: bool = False) -> nn.Module:
    """
    Gets a 5CNN that does not change the spatial dimension [H,W] as it processes the image.
    :return:
    """
    from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner
    mdl: nn.Module = get_default_learner(image_size, bn_eps, bn_momentum, n_classes, filter_size, levels, spp)
    return mdl


# -- misc

def _set_track_running_stats_to_false(module: nn.Module, name: str):
    """
    refs:
        - https://discuss.pytorch.org/t/batchnorm1d-with-batchsize-1/52136/8
        - https://stackoverflow.com/questions/64920715/how-to-use-have-batch-norm-not-forget-batch-statistics-it-just-used-in-pytorch
    """
    assert False, 'Untested'
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        print(target_attr)
        if type(target_attr) == torch.nn.BatchNorm1d:
            target_attr.track_running_stats = False
            # target_attr.running_mean = input.mean()
            # target_attr.running_var = input.var()
            # target_attr.num_batches_tracked = torch.tensor(0, dtype=torch.long)

    # "recurse" iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        _path_bn_layer_for_functional_eval(immediate_child_module, name)


def _replace_bn(module: nn.Module, name: str):
    """
    Recursively put desired batch norm in nn.module module.
    Note, this will replace

    set module = net to start code.
    """
    assert False, 'Untested'
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # - I don't think this is right. You need to retain the old values but change track_running_stats to False...
            new_bn = torch.nn.BatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum,
                                          target_attr.affine,
                                          track_running_stats=False)
            new_bn.load_state_dict(target_attr.state_dict())
            setattr(module, attr_str, new_bn)

    # "recurse" iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        _replace_bn(immediate_child_module, name)


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def _reset_all_linear_layer_weights(model: nn.Module) -> nn.Module:
    """
    Resets all weights recursively for linear layers.

    ref:
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)
            # torch.nn.init.xavier_uniform(m.weight.data)

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(init_weights)


def reset_all_weights_with_specific_layer_type(model: nn.Module, modules_type2reset) -> nn.Module:
    """
    Resets all weights recursively for linear layers.

    ref:
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def init_weights(m):
        if type(m) == modules_type2reset:
            # if type(m) == torch.nn.BatchNorm2d:
            #     m.weight.fill_(1.0)
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(init_weights)


# -- tests

def reset_params_test():
    """
    test works especially becuase the reset norm for both pretrained and random is very close.

    lp_norm(resnet18)=tensor(517.5472, grad_fn=<AddBackward0>)
    lp_norm(resnet18_random)=tensor(668.0970, grad_fn=<AddBackward0>)
    lp_norm(resnet18)=tensor(517.5472, grad_fn=<AddBackward0>)
    lp_norm(resnet18_random)=tensor(668.0970, grad_fn=<AddBackward0>)
    lp_norm(resnet18)=tensor(476.0279, grad_fn=<AddBackward0>)
    lp_norm(resnet18_random)=tensor(475.9575, grad_fn=<AddBackward0>)
    """
    import torchvision.models as models
    from uutils.torch_uu import lp_norm

    resnet18 = models.resnet18(pretrained=True)
    resnet18_random = models.resnet18(pretrained=False)

    print(f'{lp_norm(resnet18)=}')
    print(f'{lp_norm(resnet18_random)=}')
    print(f'{lp_norm(resnet18)=}')
    print(f'{lp_norm(resnet18_random)=}')
    reset_all_weights(resnet18)
    reset_all_weights(resnet18_random)
    print(f'{lp_norm(resnet18)=}')
    print(f'{lp_norm(resnet18_random)=}')


if __name__ == '__main__':
    reset_params_test()
    print('Done! \a\n')
