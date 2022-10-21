"""
ViT code for uutils.

refs:
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

import torch

from torch import nn

from learn2learn.vision.benchmarks import BenchmarkTasksets

from transformers import ViTFeatureExtractor, ViTModel

from pdb import set_trace as st


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
                                          target_transform=lambda data: torch.tensor(data, dtype=torch.int))
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         target_transform=lambda data: torch.tensor(data, dtype=torch.int))
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
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # this blog also uses the in21k, https://huggingface.co/blog/fine-tune-vit
    # model with with input size for cifar
    from transformers import ViTConfig
    vitconfig = ViTConfig(image_size=32)
    model = ViTModel(vitconfig)
    # cls layer
    num_labels, cls_p_dropout = 100, 0.1  # seems they use 0.1 p_dropout for cosine scheduler, but also use weight decay...
    dropout = nn.Dropout(cls_p_dropout)
    classifier = nn.Linear(model.config.hidden_size, num_labels)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # -
    epochs = 1
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            # - process batch
            x, y = batch
            x.to(device), y.to(device)
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
            print(f'{out.size()=}')
            # todo: Q, so do we do dropout after cls linear layer? Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings.
            cls_token_emebd = out.last_hidden_state[:, 0]
            print(f'{cls_token_emebd.size()=}')
            output = dropout(cls_token_emebd)
            logits = classifier(output)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = criterion(logits, y)
            print(f'{loss=}')
            # then you can do backward()
            st()


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

    # - get benchmark
    from diversity_src.dataloaders.hdb1_mi_omniglot_l2l import hdb1_mi_omniglot_tasksets
    benchmark: BenchmarkTasksets = hdb1_mi_omniglot_tasksets()
    splits = ['train', 'validation', 'test']
    tasksets = [getattr(benchmark, split) for split in splits]

    # - loop through tasks
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    # model = get_model('resnet18', pretrained=False, num_classes=5).to(device)
    #
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    # The bare ViT Model transformer outputting raw hidden-states without any specific head on top.
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    criterion = nn.CrossEntropyLoss()
    for i, taskset in enumerate(tasksets):
        print(f'-- {splits[i]=}')
        # - train loop
        for iteration in range(num_iterations):
            for task_num in range(batch_size):
                print(f'{task_num=}')

                X, y = taskset.sample()
                print(f'{X.size()=}')
                print(f'{y.size()=}')
                print(f'{y=}')

                image = X

                inputs = feature_extractor(image, return_tensors="pt")
                outputs = model(**inputs)
                print(f'{outputs=}')
                last_hidden_states = outputs.last_hidden_state
                st()

                y_pred = model(X)
                loss = criterion(y_pred, y)
                print(f'{loss=}')
                print()

    print('-- end of test --')


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
    cifar_vit()
    # hdb1_vit()
    # mi_vit()
    # - Done
    print(f"\nSuccess Done!: {report_times(start)}\a")
