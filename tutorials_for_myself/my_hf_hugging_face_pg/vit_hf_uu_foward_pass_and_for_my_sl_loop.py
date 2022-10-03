"""
- tutorial: https://huggingface.co/docs/transformers/model_doc/vit#transformers.FlaxViTForImageClassification
- notebooks: https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer
"""
# %%

# from transformers import T5Tokenizer, T5ForConditionalGeneration
#
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
#
# input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
# labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
#
# # the forward function automatically creates the correct decoder_input_ids
# loss = model(input_ids=input_ids, labels=labels).loss
# loss.item()

# %%

from datasets import load_dataset

# load cifar10 (only small portion for demonstration purposes)
from transformers.modeling_outputs import ImageClassifierOutput

train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

id2label = {id: label for id, label in enumerate(train_ds.features['label'].names)}
print(f'{id2label=}')
label2id = {label: id for id, label in id2label.items()}
print(f'{label2id=}')

from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=10,
                                                  id2label=id2label,
                                                  label2id=label2id)

# import torch
# x = torch.randn(4, 32, 32, 3)
# x = train_ds[0]['img']
# x = x.convert("RGB")
# print(f'{x.size()=}')
# logits = model(x)
# print(f'{logits=}')

# %%

from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
    [
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)


def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


x = train_ds[0]['img']
x = _train_transforms(x.convert("RGB"))
print(f'{x.size()=}')

x = x.unsqueeze(0)  # add batch size 1
out_cls: ImageClassifierOutput = model(x)
print(f'{out_cls.logits=}')

print()

# %%
# can't download vitt for some reason?
# https://stackoverflow.com/questions/73939929/how-to-resolve-the-hugging-face-error-importerror-cannot-import-name-is-tokeni

from pathlib import Path
import torchvision
from typing import Callable

root = Path("~/data/").expanduser()
# root = Path(".").expanduser()
train = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
test = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
img2tensor: Callable = torchvision.transforms.ToTensor()

from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

x, y = train_ds[0]
print(f'{y=}')
print(f'{type(x)=}')
x = img2tensor(x)

x = x.unsqueeze(0)  # add batch size 1
out_cls: ImageClassifierOutput = model(x)
print(f'{out_cls.logits=}')

# %%
