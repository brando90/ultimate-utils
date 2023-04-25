"""


Build a pytorch python class called LinearModelFromFixedNNFeatures that takes in a nn.Module called backbone and builds a linear model from it.
So it should have a self.w_lin = nn.Linear(Din, Dout) in the __init__ class where Din is the size of the output features of the features extracted
from the backbone and importantly, the parameters of the input backbone are not part of the parameters of our linear class.
Therefore, we assert that the backbone are not included as a parameter in our linear model.
Dout is an argument to __init__.
One case we want to handle is the backbone being a clip model from huggingface, so we need to be able to extract the features embedding from
clip and then get the size of the features embedding for Din.
The forward function should take in a batch of data and return the output of the linear model by extracting an emedding layer
using the features extractor for the backbone (e.g. huggingface's clip) make sure that is cloned and detached so gradients
do not pass to the backbone and the produces a prediction doing
y = self.w(phi_x) where phi_x is the features extracted from the backbone.
We also assert that backbone's parameters are not part of our linear model's parameters before returning y.
We also print the number of parameters for self, self.w_lin.
Write good concise clear comments to the code and produce the comment in one shot.


linear model from a nn.Module that takes in a nn.module but doesn't include nn.module input as a parameter in our linear model.

"""

import torch
import torch.nn as nn

import numpy as np
import random


class LinearModelFromFixedNNFeatures(nn.Module):
    def __init__(self, backbone: nn.Module, Dout: int):
        super().__init__()
        # - Store the backbone model without registering its as a parameter so no grads for them
        # Register the backbone model as a buffer, https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch#:~:text=Registering%20these%20%22arguments%22%20as%20the,updating%20them%20using%20SGD%20mechanism.
        self.register_buffer("backbone", None)
        self.backbone = backbone

        # -- Extract the feature dimension from the backbone model
        # e.g. For the CLIP model from Huggingface, the feature dimension is 768
        self.D_feature_embeddings = get_feature_size_of_model_embedding_feature_extractor(backbone)
        print(f'{self.D_feature_embeddings=}')

        # -- Initialize the linear layer for the linear model
        Din = self.D_feature_embeddings
        self.w_lin = nn.Linear(Din, Dout)

        # - Optional sanity checks:
        # print params of everyone (backbone, w_lin, self) seperately, making sure self only has w_lin
        self._print_num_params_self_and_w_lin()
        self._assert_backbone_params_grads_are_not_set()

    def forward(self, x: torch.Tensor, assert_no_grads_for_backbone: bool = True) -> torch.Tensor:
        # Extract features from the backbone model
        with torch.no_grad():
            # .clone() creates a copy of the extracted features tensor. This new tensor has the same data as the original tensor but is stored in a different memory location.
            # .detach() removes the new tensor from the computation graph. This means that gradient information will not be backpropagated through the new tensor to its origin.
            features = get_feature_size_of_model_embedding_feature_extractor(self, x).clone().detach()

        # Apply the linear layer to the features
        y = self.w_lin(features)

        # Assert that backbone parameters are not part of our linear model's parameters
        if assert_no_grads_for_backbone:
            self._assert_backbone_params_grads_are_not_set()
        return y

    def fast_adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Won't implement (for now), but for completion we could have the self.w_lin be adapted for prediction e.g. using
        sgd, adam or lbfgs.
        """
        pass

    def _assert_backbone_params_not_in_class(self):
        """
        note when using register bufffer: Registering these "arguments" as the model's buffer allows pytorch to track them and save them like regular parameters, but prevents pytorch from updating them using SGD mechanism. ref: https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch#:~:text=Registering%20these%20%22arguments%22%20as%20the,updating%20them%20using%20SGD%20mechanism
        """
        for name, param in self.named_parameters():
            assert "backbone" not in name, f"Backbone parameter '{name}' is part of the linear model's parameters."

    def _assert_backbone_params_grads_are_not_set(self):
        for param in self.backbone.parameters():
            assert param.grad is None or (param.grad == 0).all(), "Gradient is being set for backbone's parameter."

    def _print_num_params_self_and_w_lin(self, verify: bool = False):
        # Print the number of parameters for self, self.w_lin, and backbone
        print(
            f"Total parameters: {sum(p.numel() for p in self.parameters())}, Linear layer parameters: {sum(p.numel() for p in self.w_lin.parameters())}, Backbone parameters: {sum(p.numel() for p in self.backbone.parameters())}")
        # assert the number of parameters of self equals self.w_lin
        if verify:
            assert sum(p.numel() for p in self.parameters()) == sum(
                p.numel() for p in
                self.w_lin.parameters()), "Number of parameters for self and self.w_lin are not equal."


# -

def get_feature_size_of_model_embedding_feature_extractor(model: nn.Module) -> int:
    if isinstance(model, nn.Linear):
        # toy example to just run code
        return model.out_features
    from transformers import CLIPModel
    if isinstance(model, CLIPModel):
        return get_feature_size_of_clip_model_embedding_feature_extractor(model)
    else:
        raise ValueError(f"The provided model {type(model)=} is not supported.")


def get_embedding_from_feature_extractor(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, nn.Linear):
        # toy example to just run code
        if hasattr(model, 'backbone'):
            return model.backbone[0]
        else:
            return model[0]
    from transformers import CLIPModel
    if isinstance(model, CLIPModel):
        return
    else:
        raise ValueError(f"The provided model {type(model)=} is not supported.")


# - clip specific code, note I am not importing clip the file level so that ppl are not forced to use hf to use this file

def get_clip_model() -> nn.Module:
    from transformers import CLIPModel
    model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print(f'{type(model)=}')
    return model


def get_feature_size_of_clip_model_embedding_feature_extractor(clip: nn.Module) -> int:
    return clip.config.hidden_size


# -- test, examples, unit tests, etc.

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, Din, Dout):
        self.num_samples = num_samples

    def __getitem__(self, index):
        data = torch.randn(1, 1)  # generate random data
        label = random.randint(0, 1)  # generate random label
        return data, label

    def __len__(self):
        return self.num_samples


def _example_task2vec_fim_div_using_nn_lin():
    """

    nearly worked, we need to fix the code for the linear model above so the task2vec code works. Error:
    File "/Users/brandomiranda/opt/anaconda3/envs/diversity-for-predictive-success-of-meta-learning/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1130, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
    AttributeError: 'LinearModelFromFixedNNFeatures' object has no attribute 'layers'
    """
    # - create minimal an args object needed for code to work
    from argparse import Namespace

    Din, Dout = 5, 3
    batch_size = 6
    args = Namespace(batch_size=batch_size)
    args.classifier_opts = None
    dataset = RandomDataset(100, Din, Dout)  # 100 samples in data set

    # - build linear model from out class using backbone nn.linear
    # - build backbone nn.linear
    model: nn.Module = nn.Linear(Din, Dout)
    # - build linear model from out class using backbone nn.linear
    linear_model: nn.Module = LinearModelFromFixedNNFeatures(model, Dout)
    args.probe_network = linear_model

    # # - compute task2vec embeddings
    split = 'train'
    # - get your pytorch data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.dataloaders = {'train': dataloader, 'val': dataloader, 'test': dataloader}

    # - get task embeddings from normal pytorch data loader
    from uutils.torch_uu.metrics.diversity.diversity import get_task_embeddings_from_normal_dataloader
    from uutils.torch_uu.metrics.diversity.task2vec_based_metrics import task2vec, task_similarity

    embeddings: list[task2vec.Embedding] = get_task_embeddings_from_normal_dataloader(args,
                                                                                      args.dataloaders,
                                                                                      args.probe_network,
                                                                                      num_tasks_to_consider=args.batch_size,
                                                                                      split=split,
                                                                                      classifier_opts=args.classifier_opts,
                                                                                      )

    # - compute distance matrix & task2vec based diversity, to demo` task2vec, this code computes pair-wise distance between task embeddings
    distance_matrix: np.ndarray = task_similarity.pdist(embeddings, distance='cosine')
    print(f'{distance_matrix=}')
    from uutils.numpy_uu.common import get_diagonal

    distances_as_flat_array, _, _ = get_diagonal(distance_matrix, check_if_symmetric=True)

    # - compute div
    from uutils.torch_uu.metrics.diversity.diversity import \
        get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks, \
        get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks

    div_tot = float(distances_as_flat_array.sum())
    print(f'Diversity: {div_tot=}')
    div, ci = get_task2vec_diversity_coefficient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Diversity: {(div, ci)=}')
    standardized_div: float = get_standardized_diversity_coffecient_from_pair_wise_comparison_of_tasks(distance_matrix)
    print(f'Standardised Diversity: {standardized_div=}')


def _example_task2vec_fim_div_using_clip():
    pass


if __name__ == '__main__':
    # print time passed to run code using time
    import time

    start_time = time.time()
    # ---
    # run code
    _example_task2vec_fim_div_using_nn_lin()
    # _example_div_fim_task2vec_using_clip()
    # print time taken
    print("--- %s seconds ---" % (time.time() - start_time))
