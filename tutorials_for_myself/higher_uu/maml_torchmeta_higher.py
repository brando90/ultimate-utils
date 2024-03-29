"""
For correctness see details here:
- SO: https://stackoverflow.com/questions/70961541/what-is-the-official-implementation-of-first-order-maml-using-the-higher-pytorch/74270560#74270560
- gitissue: https://github.com/facebookresearch/higher/issues/102
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging

from collections import OrderedDict

import higher  # tested with higher v0.2

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

logger = logging.getLogger(__name__)


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def train(args):
    logger.warning('This script is an example to showcase the data-loading '
                   'features of Torchmeta in conjunction with using higher to '
                   'make models "unrollable" and optimizers differentiable, '
                   'and as such has been  very lightly tested.')

    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       meta_train=True,
                       download=args.download,
                       )
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()
    inner_optimiser = torch.optim.SGD(model.parameters(), lr=args.step_size)
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    # understanding ETA: https://github.com/tqdm/tqdm/issues/40, 00:05<00:45 means 5 seconds have elapsed and a further (estimated) 45 remain. < is used as an ASCII arrow really rather than a less than sign.
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)

            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):
                track_higher_grads = True
                # track_higher_grads = False  # never set to False it seems.
                with higher.innerloop_ctx(model, inner_optimiser, track_higher_grads=track_higher_grads, copy_initial_weights=False) as (fmodel, diffopt):
                    train_logit = fmodel(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)

                    # diffopt.step(inner_loss)
                    # FO with track_higher_grads = True
                    # diffopt.step(inner_loss, grad_callback=lambda grads: [g.detach() for g in grads])

                    test_logit = fmodel(test_input)
                    outer_loss += F.cross_entropy(test_logit, test_target)

                    # inspired by https://github.com/facebookresearch/higher/blob/15a247ac06cac0d22601322677daff0dcfff062e/examples/maml-omniglot.py#L165
                    # outer_loss = F.cross_entropy(test_logit, test_target)
                    # outer_loss.backward()

                    with torch.no_grad():
                        accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)

            outer_loss.backward()
            # print(list(model.parameters()))
            # print(f"{meta_optimizer.param_groups[0]['params'] is list(model.parameters())}")
            # print(f"{meta_optimizer.param_groups[0]['params'][0].grad is not None=}")
            # print(f"{meta_optimizer.param_groups[0]['params'][0].grad=}")
            print(f"{meta_optimizer.param_groups[0]['params'][0].grad.norm()}")
            assert meta_optimizer.param_groups[0]['params'][0].grad is not None
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches:
                break

    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_omniglot_'
                                                    '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)


if __name__ == '__main__':
    seed = 0

    import random
    import numpy as np
    import torch
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    # - make pytorch determinsitc
    # makes all ops determinsitic no matter what. Note this throws an errors if you code has an op that doesn't have determinsitic implementation
    torch.manual_seed(seed)
    # if always_use_deterministic_algorithms:
    torch.use_deterministic_algorithms(True)
    # makes convs deterministic
    torch.backends.cudnn.deterministic = True
    # doesn't allow benchmarking to select fastest algorithms for specific ops
    torch.backends.cudnn.benchmark = False
    # - make python determinsitic
    np.random.seed(seed)
    random.seed(seed)

    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str, default=Path('~/data/torchmeta_data').expanduser(),
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--step-size', type=float, default=0.4,
                        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_false',
                        help='Do not Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
                                         and torch.cuda.is_available() else 'cpu')

    print(f'{args.device=}')
    train(args)
