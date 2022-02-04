"""
learn2learn examples: https://github.com/learnables/learn2learn/tree/master/examples/vision
"""
import learn2learn
from torch import nn


def cnn4_cifarsfs(ways: int = 5,
                  hidden_size=64,
                  embedding_size=64 * 4,
                  ) -> tuple[nn.Module, dict]:
    """
    Based on: https://github.com/learnables/learn2learn/blob/master/examples/vision/anil_fc100.py
    """
    model_hps: dict = dict()
    model = learn2learn.vision.models.CNN4(output_size=ways, hidden_size=hidden_size, embedding_size=embedding_size, )
    return model, model_hps
