"""
learn2learn examples: https://github.com/learnables/learn2learn/tree/master/examples/vision
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
        )#is this for the transfer learning stuff/probing the second-to-last layer?
        maml_init_(self.classifier)#random xaiver uniform initialization
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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
