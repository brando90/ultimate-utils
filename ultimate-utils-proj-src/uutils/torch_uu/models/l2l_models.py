"""
learn2learn examples: https://github.com/learnables/learn2learn/tree/master/examples/vision
"""
import learn2learn
from torch import nn


def cnn4_cifarsfs(ways: int,
                  hidden_size=64,
                  embedding_size=64 * 4,
                  ) -> tuple[nn.Module, dict]:
    """
    Based on: https://github.com/learnables/learn2learn/blob/master/examples/vision/anil_fc100.py
    """
    model_hps: dict = dict(ways=ways, hidden_size=hidden_size, embedding_size=embedding_size)
    model = learn2learn.vision.models.CNN4(output_size=ways, hidden_size=hidden_size, embedding_size=embedding_size, )
    model.cls = model.classifier
    # replace_final_layer(args, n_classes=ways)  # for meta-learning, this is done at the user level not data set
    return model, model_hps
