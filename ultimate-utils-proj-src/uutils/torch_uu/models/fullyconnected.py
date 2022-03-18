"""
Implements 3-layer fully connected network. Mainly used for 5-way gaussian classification task
Modified by Patrick 3/14/22
"""
import learn2learn
import torch
from learn2learn.vision.models import CNN4Backbone, maml_init_
from torch import nn

def fnn3_gaussian(ways: int,
                  input_size: int,
                  hidden_layer1 = 15,
                  hidden_layer2 = 15,
                  ) -> tuple[nn.Module, dict]:
    model_hps : dict = dict(ways = ways, input_size = input_size, hidden_layer1=hidden_layer1, hidden_layer2 = hidden_layer2)
    model = FNN3(output_size=ways, input_size = input_size, hidden_layer1=hidden_layer1, hidden_layer2=hidden_layer2, )
    return model, model_hps

class FNN3(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_layer1=15,
            hidden_layer2=15,
    ):
        super().__init__()

        '''self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_layer1),
            #nn.BatchNorm1d(hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            #nn.BatchNorm1d(hidden_layer2),
            nn.ReLU(),
            nn.Linear(hidden_layer2, output_size)
        )'''

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_layer1),
            nn.BatchNorm1d(hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.BatchNorm1d(hidden_layer2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_layer2, output_size)



    def forward(self, x):
        #x = x.view(-1,1).float()
        x = self.features(x)
        x = self.classifier(x)
        return x#.float()

    # unfortuantely needed, otherwise pytorch seems to add it to the modules and then if it does a backwards pass it
    # think there are parameters not being trained, although self.cls is self.classifier should return True
    @property
    def cls(self):
        return self.classifier

    @cls.setter
    def cls(self, new_cls):
        self.classifier = new_cls


# - tests

def fnn_test():
    model, _ = fnn3_gaussian(ways=5, input_size=1, hidden_layer1=15,hidden_layer2=15)

    x = torch.randn(1)
    y = model(x)
    print(y)
    print(y.size())

if __name__ == '__main__':
    fnn_test()
    print('Done\a')
