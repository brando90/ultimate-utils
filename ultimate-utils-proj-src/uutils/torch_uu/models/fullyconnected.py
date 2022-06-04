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
                  hidden_layers = [15,15],
                  #hidden_layer1 = 15,
                  #hidden_layer2 = 15,
                  ) -> tuple[nn.Module, dict]:
    model_hps : dict = dict(ways = ways, input_size = input_size, hidden_layers=hidden_layers)
    model = FNN3(output_size=ways, input_size = input_size, hidden_layers=hidden_layers)
    return model, model_hps

class FNN3(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_layers=[15, 15],
            #hidden_layer1=15,
            #hidden_layer2=15,
    ):
        super().__init__()

        assert len(hidden_layers) >= 2, "Need at least 2 hidden layers"

        # Start of our FNN: input -> hidden_layer[0]
        modules = [
            nn.Flatten(),
            nn.Linear(input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU()
        ]
        # Intermediate layers
        for i in range(len(hidden_layers)-1):
            layer = [
                nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                nn.BatchNorm1d(hidden_layers[i+1]),
                nn.ReLU()
            ]
            modules.extend(layer)
        # Put all start and intermediate layers together
        self.features = nn.Sequential(*modules)

        # Last layer is "Classifier" layer: from hidden_layers[-1] -> output
        self.classifier = nn.Linear(hidden_layers[-1], output_size)
        '''
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_layers[1], output_size)
        '''
    #@property
    #def classifier(self):
    #    return self.classifier


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
    model, _ = fnn3_gaussian(ways=5, input_size=1, hidden_layers=[15,15,15])#hidden_layer1=15,hidden_layer2=15)

    x = torch.randn([10,1,1,1])
    y = model(x)
    print(y)
    print(y.size())

if __name__ == '__main__':
    fnn_test()
    print('Done\a')
