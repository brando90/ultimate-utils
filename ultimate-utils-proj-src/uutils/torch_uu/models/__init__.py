from collections import OrderedDict

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
