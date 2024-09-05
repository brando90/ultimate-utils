from argparse import Namespace
from torch.nn import Module
from typing import Optional

from uutils.torch_uu.metrics.diversity.task2vec_based_metrics.task2vec import ProbeNetwork

def get_probe_network(args: Namespace,

                      model_option: Optional[str] = None,
                      **model_hps
                      ) -> ProbeNetwork:
    """
    Note: a model is not the same as a probe network. Make sure you respect the probe network interface.

    :param args:
    :param model_option:
    :return:
    """
    from uutils.torch_uu.metrics.diversity.task2vec_based_metrics.models import get_model

    # print(f'{get_model=}')
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
        from uutils.torch_uu.models.learner_from_opt_as_few_shot_paper import get_default_learner_and_hps_dict
        probe_network, _ = get_default_learner_and_hps_dict()  # 5cnn
        # probe_network: Module = get_default_learner()
        # probe_network: ProbeNetwork = get_5CNN_random_probe_network()
        # not implemented because it needs the probe network API I think...
        raise NotImplementedError
    elif model_option == '3FNN_5_gaussian':
        from models import get_model, gaussian_net
        probe_network: ProbeNetwork = gaussian_net(num_classes=args.n_cls)
    else:
        raise ValueError(f'')

    assert isinstance(probe_network, ProbeNetwork), f'Make sure your model is of type ProbeNework & respects' \
                                                             f'its API. Got type: {type(probe_network)}'
    # -
    from uutils.torch_uu.distributed import move_model_to_dist_device_or_serial_device
    probe_network = move_model_to_dist_device_or_serial_device(args.rank, args, probe_network)
    return probe_network


# tests

def get_model_test():
    print(f'{get_model=}')
    probe_network: ProbeNetwork = get_model('resnet18', pretrained=True, num_classes=5)
    print(probe_network)

    args = Namespace(n_cls=5)
    args.model_option = 'resnet18_pretrained_imagenet'
    probe_network: ProbeNetwork = get_probe_network(args)
    print(probe_network)


if __name__ == '__main__':
    get_model_test()
