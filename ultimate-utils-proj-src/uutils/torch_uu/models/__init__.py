from collections import OrderedDict

from torch import nn

def get_linear_model(in_features, out_features):
    return nn.Linear(in_features=in_features, out_features=out_features)

def hardcoded_3_layer_model(in_features: int, out_features: int) -> nn.Module:
    """
    Returns a nn sequential model with 3 layers (2 hidden and 1 output layers).
    ReLU activation. Final layer are the raw scores (so final layer is a linear layer).

    """
    from collections import OrderedDict
    hidden_features = in_features
    params = OrderedDict([
        ('fc0', nn.Linear(in_features=in_features, out_features=hidden_features)),
        ('ReLU0', nn.ReLU()),
        ('fc1', nn.Linear(in_features=hidden_features, out_features=hidden_features)),
        ('ReLU2', nn.ReLU()),
        ('fc2', nn.Linear(in_features=hidden_features, out_features=out_features))
    ])
    mdl = nn.Sequential(params)
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
        params = OrderedDict([
            ('fc1', nn.Linear(in_features=hidden_features, out_features=out_features))
        ])
    elif num_layer == 2:
        # for clarity
        params = OrderedDict([
            ('fc0', nn.Linear(in_features=in_features, out_features=hidden_features)),
            ('ReLU0', nn.ReLU()),
            ('fc1', nn.Linear(in_features=hidden_features, out_features=out_features))
        ])
    else:
        params = OrderedDict([])
        for l in range(num_layer - 1):
            if l < num_layer:
                # append one fc + relu
                params.append((f'fc{l}', nn.Linear(in_features=in_features, out_features=hidden_features)))
                params.append((f'ReLU{l}', nn.ReLU()))
            else:
                params.append((f'fc{l}', nn.Linear(in_features=in_features, out_features=hidden_features)))
    mdl = nn.Sequential(params)
    return mdl
