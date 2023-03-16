"""
ViT code for uutils.

refs:
    - docs: https://huggingface.co/docs/transformers/model_doc/vit
    - fine-tuning ViT with hf trainer: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb
    - looks like a great tutorial for training the model manually: https://colab.research.google.com/drive/1Z1lbR_oTSaeodv9tTm11uEhOjhkUx1L4?usp=sharing#scrollTo=5ql2T5PDUI1D
    - nice forward pass tutorial: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb


Qs:
- "ViTModel: This is the base model that is provided by the HuggingFace transformers library and is the core of the vision transformer. Note: this can be used like a regular PyTorch layer." https://blog.roboflow.com/how-to-train-vision-transformer/
- Q: why do we need both the feature extractor and ViTModel?
- Q: "Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this."

Original transformer did the following for dropout:
```
Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop = 0.1
```
**so before the addition and norm, we do dropout of multi-headed self attention (MHSE) sublayer & FC/MLP sublayer.** (for vasmani trans)

For ViT:
```
Dropout, when used, is applied after
every dense layer except for the the qkv-projections and directly after adding positional- to patch
embeddings.
```

# pseudocode for original transformer:
x = get_seq()
x = x + PosEnc()
# - transformer block
x = MHSA(x) = SF(KVQ(x)) = SubLayer(x)
x = LNorm(x + DropOut(x)) = LNorm(x + DropOut(SubLayer(x))
x = FC(x) = RelU(xW1+b1)W2+b2 = SubLayer(x)
x = LNorm(x + DropOut(FC(x))) = LNorm(x + DropOut(SubLayer(x))
x = output of encoder # decoder is similar but it has a masked MHSA & then MHSA & then FC.

"""
from argparse import Namespace
from pathlib import Path
from typing import Callable, Optional, Union

import torch

from torch import nn, Tensor

from learn2learn.vision.benchmarks import BenchmarkTasksets

from transformers import ViTFeatureExtractor, ViTModel, ViTConfig

from pdb import set_trace as st

from transformers.modeling_outputs import BaseModelOutputWithPooling


class ViTForImageClassificationUU(nn.Module):
    def __init__(self,
                 num_classes: int,
                 image_size: int,  # 224 inet, 32 cifar, 84 mi, 28 mnist, omni...
                 criterion: Optional[Union[None, Callable]] = None,
                 # Note: USL agent does criterion not model usually for me e.g nn.Criterion()
                 cls_p_dropout: float = 0.0,
                 pretrained_name: str = None,
                 vitconfig: ViTConfig = None,
                 ):
        """
        :param num_classes:
        :param pretrained_name: 'google/vit-base-patch16-224-in21k'  # what the diff with this one: "google/vit-base-patch16-224"
        """
        super().__init__()
        if vitconfig is not None:
            raise NotImplementedError
            self.vitconfig = vitconfig
            print(f'You gave a config so everyone other param given is going to be ignored.')
        elif pretrained_name is not None:
            raise NotImplementedError
            # self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.model = ViTModel.from_pretrained(pretrained_name)
            print('Make sure you did not give a vitconfig or this pretrained name will be ignored.')
        else:
            self.num_classes = num_classes
            self.image_size = image_size
            self.vitconfig = ViTConfig(image_size=self.image_size)
            self.model = ViTModel(self.vitconfig)
        assert cls_p_dropout == 0.0, 'Error, for now only p dropout for cls is zero until we figure out if we need to ' \
                                     'change all the other p dropout layers too.'
        self.dropout = nn.Dropout(cls_p_dropout)
        self.cls = nn.Linear(self.model.config.hidden_size, num_classes)
        self.criterion = None if criterion is None else criterion

    def forward(self, batch_xs: Tensor, labels: Tensor = None) -> Tensor:
        """
        Forward pass of vit. I added the "missing" cls (and drouput layer before it) to act on the first cls
        token embedding. Remaining token embeddings are ignored/not used.

        I think the feature extractor only normalizes the data for you, doesn't seem to even make it into a seq, see:
        ...
        so idk why it's needed but an example using it can be found here:
            - colab https://colab.research.google.com/drive/1Z1lbR_oTSaeodv9tTm11uEhOjhkUx1L4?usp=sharing#scrollTo=cGDrb1Q4ToLN
            - blog with trainer https://huggingface.co/blog/fine-tune-vit
            - single PIL notebook https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb
        """
        outputs: BaseModelOutputWithPooling = self.model(pixel_values=batch_xs)
        output: Tensor = self.dropout(outputs.last_hidden_state[:, 0])
        logits: Tensor = self.cls(output)
        if labels is None:
            assert logits.dtype == torch.float32
            return logits  # this is what my usl agent does ;)
        else:
            raise NotImplementedError
            assert labels.dtype == torch.long
            #   loss = self.criterion(logits.view(-1, self.num_classes), labels.view(-1))
            loss = self.criterion(logits, labels)
            return loss, logits

    def _assert_its_random_model(self):
        from uutils.torch_uu import norm
        pre_trained_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        print(f'----> {norm(pre_trained_model)=}')
        print(f'----> {norm(self)=}')
        assert norm(pre_trained_model) > norm(self), f'Random models usually have smaller weight size but got ' \
                                                     f'{norm(pre_trained_model)}{norm(self)}'


def get_vit_model_and_model_hps(num_classes: int,
                                image_size: int,
                                ) -> tuple[nn.Module, dict]:
    """get vit."""
    model_hps: dict = dict(num_classes=num_classes, image_size=image_size)
    model: nn.Module = ViTForImageClassificationUU(num_classes, image_size)
    return model, model_hps


def get_vit_get_vit_model_and_model_hps_mi(num_classes: int = 5,  # or 64, etc. for data set's num_cls
                                           image_size: int = 84,  # 224 inet, 32 cifar, 84 mi, 28 mnist, omni...
                                           criterion: Optional[Union[None, Callable]] = None,
                                           ) -> tuple[nn.Module, dict]:
    """get vit for mi, only num_classes = 5 and image size 84 is needed. """
    model_hps: dict = dict(num_classes=num_classes, image_size=image_size, criterion=criterion)
    model: nn.Module = ViTForImageClassificationUU(num_classes, image_size)
    return model, model_hps


def get_vit_get_vit_model_and_model_hps(vitconfig: ViTConfig = None,
                                        num_classes: int = 5,
                                        image_size: int = 84,  # 224 inet, 32 cifar, 84 mi, 28 mnist, omni...
                                        criterion: Optional[Union[None, Callable]] = None,  # for me agent does it
                                        cls_p_dropout: float = 0.0,
                                        pretrained_name: str = None,
                                        ) -> tuple[nn.Module, dict]:
    """get vit for mi, only num_classes = 5 and image size 84 is needed. """
    model_hps: dict = dict(vitconfig=vitconfig,
                           num_classes=num_classes,
                           image_size=image_size,
                           criterion=criterion,
                           cls_p_dropout=cls_p_dropout,
                           pretrained_name=pretrained_name)
    model: nn.Module = ViTForImageClassificationUU(**model_hps)
    return model, model_hps


# - tests

def does_feature_extractor_have_params():
    """
    what is the HF feature extractor doing?
    1. Is it only cutting into patches HxWxC -> Nx(P^2*C)
    2. Or is it doing 1 AND embedding layer ... -> Nx(P^2*C) *_mat (P^2*C)xD
    notation: size notation A*B == [A,B]

    answer: after printing out the fields (using vars(model) or model.__dict__) I saw this in the arch text rep:
    ```
...
  (patch_embeddings): PatchEmbeddings(
    (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  )
  (dropout): Dropout(p=0.0, inplace=False)
)), ('encoder', ViTEncoder(
  (layer): ModuleList(
    (0): ViTLayer(
      (attention): ViTAttention(
        (attention): ViTSelfAttention(
          (query): Linear(in_features=768, out_features=768, bias=True)
...
    ```
    no params for feature extractor makes me think the "feature extractor" only reshapes the img (3D) to a sequence
    according to the patch: HxWxC -> Nx(P^2*C) s.t. P=HW/P^2.

    ref: https://arxiv.org/abs/2010.11929
    """
    # - get model (for inference)
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    print(f'{type(model)=}')
    # type(model)=<class 'transformers.models.vit.modeling_vit.ViTForImageClassification'>

    from transformers import ViTModel
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    print(f'{type(model)=}')
    print(f'{isinstance(model, nn.Module)=}')
    # type(model)=<class 'transformers.models.vit.modeling_vit.ViTModel'>
    model.eval()
    model.to(device)

    # - get an image
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    # im

    # - get encoding/embedding of im
    from transformers import ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    print(f'{feature_extractor=}')
    print(type(feature_extractor))
    print(f'{isinstance(feature_extractor, nn.Module)=}')  # False! :D
    # class 'transformers.models.vit.feature_extraction_vit.ViTFeatureExtractor'>
    encoding = feature_extractor(images=im, return_tensors="pt")
    print(f'{encoding=}')
    encoding.keys()

    print(f"{encoding['pixel_values'].shape=}")

    # - forward pass (get logits!)
    pixel_values = encoding['pixel_values'].to(device)
    outputs = model(pixel_values)
    print(f'{outputs=}')
    logits = outputs.logits
    print(f'{logits.shape}=')

    # - (extra, get label name), empty function in class, point to tutorial
    prediction = logits.argmax(-1)
    print("Predicted class:", model.config.id2label[prediction.item()])


def cifar_vit():
    """meant for something completely self contained...if I have time. """
    # - get data and data loader
    import torchvision
    from pathlib import Path
    root = Path('~/data/').expanduser()
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.long))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train, batch_size=4)

    # - get vit model & patcher
    # matching models according to: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb
    # from transformers import ViTForImageClassification
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    # custom data set tutorial: https://colab.research.google.com/drive/1Z1lbR_oTSaeodv9tTm11uEhOjhkUx1L4?usp=sharing#scrollTo=5ql2T5PDUI1D
    from transformers import ViTModel
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    from uutils.torch_uu import norm
    print(f'----> {norm(model)=}')
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # this blog also uses the in21k, https://huggingface.co/blog/fine-tune-vit
    # model with with input size for cifar
    from transformers import ViTConfig
    vitconfig = ViTConfig(image_size=32)
    model = ViTModel(vitconfig)
    from uutils.torch_uu import norm
    print(f'----> {norm(model)=}')
    # cls layer
    num_classes, cls_p_dropout = 100, 0.1  # seems they use 0.1 p_dropout for cosine scheduler, but also use weight decay...
    dropout = nn.Dropout(cls_p_dropout)
    cls = nn.Linear(model.config.hidden_size, num_classes)
    # loss
    criterion = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # to device
    model.to(device)
    cls.to(device)  # put as field to class so it's moved automatically
    # -
    epochs = 1
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            # - process batch
            x, y = batch
            x, y = x.to(device), y.to(device)
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            print(f'{x.size()=}')

            # - forward pass
            # n case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a number of channels, H and W are image height and width.
            # assert x.size() == torch.Size([C, H, W)) # if passed to the weird hf feature extractor module
            # assert x.size(1) == 3
            # x = feature_extractor(x)
            # print(f'{x.size()=}')
            out = model(x)
            # todo: Q, so do we do dropout after cls linear layer? Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings.
            cls_token_emebd = out.last_hidden_state[:, 0]
            print(f'{cls_token_emebd.size()=}')
            output = dropout(cls_token_emebd)
            logits = cls(output)
            # loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            loss = criterion(logits, y)
            print(f'{loss=}')
            # then you can do backward()
            break
        break
    print('--- success vit cifar ---')
    return


def hdb1_vit():
    """
    testing hdb1 with vit.

    """
    # - for determinism
    # random.seed(0)
    # torch.manual_seed(0)
    # np.random.seed(0)

    # - options for number of tasks/meta-batch size
    batch_size = 5
    num_iterations = 1
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    # - get benchmark
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import hdb1_mi_omniglot_tasksets
    benchmark: BenchmarkTasksets = hdb1_mi_omniglot_tasksets()
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - get my vit model
    model = ViTForImageClassificationUU(num_classes=64 + 1100, image_size=84)
    criterion = nn.CrossEntropyLoss()
    # to device
    model.to(device)
    criterion.to(device)
    # -
    for i, taskset in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        # - train loop
        for iteration in range(num_iterations):
            for task_num in range(batch_size):
                print(f'{task_num=}')

                x, y = taskset.sample()
                x, y = x.to(device), y.to(device)
                print(f'{x.size()=}')
                print(f'{y.size()=}')
                print(f'{y=}')
                assert isinstance(x, torch.Tensor)
                assert isinstance(y, torch.Tensor)

                # - forward pass
                logits = model(x)
                loss = criterion(logits, y)
                print(f'{loss=}')
                # then you can do backward()
                break
            break
        break
    return


def mi_vit():
    """test for vit on mini-imagenet."""
    # from uutils.torch_uu.dataloaders.meta_learning.l2l_ml_tasksets import get_l2l_tasksets
    #
    # args = Namespace(k_shots=5, n_cls=5, k_eval=15, data_option='cifarfs', data_path=Path('~/data/l2l_data/'))
    # args.batch_size = 5
    #
    # tasksets: BenchmarkTasksets = get_l2l_tasksets(args)
    # # - loop through tasks
    # device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    # criterion = nn.CrossEntropyLoss()
    # for i, taskset in enumerate(tasksets):
    #     print(f'-- {splits[i]=}')
    #     for task_num in range(batch_size):
    pass

def vit_forward_pass():
    # - for determinism
    import random
    import numpy as np
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # - options for number of tasks/meta-batch size
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    # - get my vit model
    vitconfig: ViTConfig = ViTConfig()
    # model = ViTForImageClassificationUU(num_classes=64 + 1100, image_size=84)
    model = get_vit_get_vit_model_and_model_hps(vitconfig, num_classes=64 + 1100, image_size=84)
    criterion = nn.CrossEntropyLoss()
    # to device
    model.to(device)
    criterion.to(device)

    # - forward pass
    x = torch.rand(5, 3, 84, 84)
    y = torch.randint(0, 64 + 1100, (5,))
    logits = model(x)
    loss = criterion(logits, y)
    print(f'{loss=}')

# -- Run experiment
"""
python ~/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/models/hf_uu/vit_uu.py
"""

if __name__ == "__main__":
    import time
    from uutils import report_times

    start = time.time()
    # - run experiment
    # does_feature_extractor_have_params()
    # cifar_vit()
    # hdb1_vit()
    # mi_vit()
    vit_forward_pass()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
