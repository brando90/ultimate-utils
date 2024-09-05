"""
learn2learn examples: https://github.com/learnables/learn2learn/tree/master/examples/vision

4CNN l2l hack:
- since SL needs to have 64 output units, I unfortuantely, hardcoded mdl.cls = nn.Linear(...,64).
doing the setter does change the .classifier to point to the right thing (see the setter decorator, also, I asserted
the pointer to be the same and the weight norms, they are the same even print(self.model) shows a mismatch in out_features)
and doing .X = Y in pytorch populates the modules. So now all models will have a .classifier and .cls modules. This
means that the state_dict of the model will have both. So when you create the model you will either need to make sure
to repalce_final_layer so that .model.cls = nn.Linear(...) is set and thus when you load the checkpoint both the
cls and classifier layer will be registered by pytorch.
Or (which is the solution I choose) is to have self.cls = self.classifier in the init so that it always has both modules.
"""
import learn2learn
import torch
from learn2learn.vision.models import CNN4Backbone, maml_init_
from torch import nn


def cnn4_cifarsfs(ways: int,
                  hidden_size=64,
                  embedding_size=64 * 4,
                  ) -> tuple[nn.Module, dict]:
    """
    Based on: https://github.com/learnables/learn2learn/blob/master/examples/vision/anil_fc100.py
    """
    model_hps: dict = dict(ways=ways, hidden_size=hidden_size, embedding_size=embedding_size)
    # model = learn2learn.vision.models.CNN4(output_size=ways, hidden_size=hidden_size, embedding_size=embedding_size, )
    model = CNN4(output_size=ways, hidden_size=hidden_size, embedding_size=embedding_size, )
    return model, model_hps


class CNN4(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    Instantiate `CNN4Backbone` if you only need the feature extractor.

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    * **channels** (int, *optional*, default=3) - The number of channels in input.
    * **max_pool** (bool, *optional*, default=True) - Whether ConvBlocks use max-pooling.
    * **embedding_size** (int, *optional*, default=None) - Size of feature embedding.
        Defaults to 25 * hidden_size (for mini-Imagenet).

    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(
            self,
            output_size,
            hidden_size=64,
            layers=4,
            channels=3,
            max_pool=True,
            embedding_size=None,
    ):
        super().__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        self.features = CNN4Backbone(
            hidden_size=hidden_size,
            channels=channels,
            max_pool=max_pool,
            layers=layers,
            max_pool_factor=4 // layers,
        )
        self.classifier = torch.nn.Linear(
            embedding_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size
        assert self.cls is self.classifier

    def forward(self, x):
        assert self.cls is self.classifier  # this just makes sure that we are running final layer we want
        x = self.features(x)
        x = self.classifier(x)
        assert self.cls is self.classifier  # this just makes sure that we are running final layer we want
        return x

    # https://stackoverflow.com/questions/71654047/how-does-one-get-the-object-in-a-python-object-inside-a-decorated-function-witho
    # unfortuantely needed, otherwise pytorch seems to add it to the modules and then if it does a backwards pass it
    # think there are parameters not being trained, although self.cls is self.classifier should return True
    @property
    def cls(self):
        return self.classifier

    @cls.setter
    def cls(self, new_cls):
        self.classifier = new_cls


# - tests

def wider_net_test():
    model, _ = cnn4_cifarsfs(ways=64, hidden_size=1024, embedding_size=1024 * 4)

    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print(y)
    print(y.size())


def _reproduce_bug():
    model, _ = cnn4_cifarsfs(ways=64, hidden_size=1024, embedding_size=1024 * 4)
    model.cls = model.classifier
    print(model)

    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print(y)
    print(y.size())
    y.sum().backward()
    print()


if __name__ == '__main__':
    # wider_net_test()
    _reproduce_bug()
    print('Done\a')
