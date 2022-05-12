from argparse import Namespace
from typing import Optional

import task2vec
from models import get_model
from task2vec import ProbeNetwork


def get_probe_network(args: Namespace,

                      model_option: Optional[str] = None,
                      **model_hps
                      ) -> task2vec.ProbeNetwork:
    """
    Note: a model is not the same as a probe network. Make sure you respect the probe network interface.

    :param args:
    :param model_option:
    :return:
    """
    model_option: str = args.model_option if model_option is None else model_option
    if model_option == 'None':
        probe_network: ProbeNetwork = get_model('resnet18', pretrained=True, num_classes=5)
    elif model_option == 'resnet18_pretrained_imagenet':
        probe_network: ProbeNetwork = get_model('resnet18', pretrained=True, num_classes=args.n_cls)
    elif model_option == 'resnet18_random':
        probe_network: ProbeNetwork = get_model('resnet18', pretrained=False, num_classes=args.n_cls)
    elif model_option == 'resnet34_pretrained_imagenet':
        probe_network: ProbeNetwork = get_model('resnet34', pretrained=True, num_classes=args.n_cls)
    elif model_option == 'resnet34_random':
        probe_network: ProbeNetwork = get_model('resnet34', pretrained=False, num_classes=args.n_cls)
    elif model_option == '5cnn_random':
        # probe_network: nn.Module = get_default_learner()
        # probe_network: ProbeNetwork = get_5CNN_random_probe_network()
        raise NotImplementedError
    else:
        raise ValueError(f'')

    assert isinstance(probe_network, task2vec.ProbeNetwork), f'Make sure your model is of type ProbeNework & respects' \
                                                             f'its API. Got type: {type(probe_network)}'
    return probe_network
