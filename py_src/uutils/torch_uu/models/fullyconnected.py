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
                  test_skip=False,
                  #hidden_layer1 = 15,
                  #hidden_layer2 = 15,
                  ) -> tuple[nn.Module, dict]:
    model_hps : dict = dict(ways = ways, input_size = input_size, hidden_layers=hidden_layers)
    model = FNN3(output_size=ways, input_size = input_size, hidden_layers=hidden_layers,test_skip=test_skip)
    return model, model_hps

class FNN3(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_layers=[15, 15],
            test_skip=False
            #hidden_layer1=15,
            #hidden_layer2=15,
    ):
        super().__init__()

        assert len(hidden_layers) >= 2, "Need at least 2 hidden layers"

        #Test if Resnet/skip connection can improve convergence
        if(test_skip == True):
            self.testskip=True
            self.f1 = nn.Flatten()
            self.l1 = nn.Linear(1, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.relu1 = nn.ReLU()
            self.l2 = nn.Linear(128, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.relu2 = nn.ReLU()
            self.l3 = nn.Linear(128, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.relu3 = nn.ReLU()
            self.l4 = nn.Linear(128, 128)
            self.bn4 = nn.BatchNorm1d(128)
            self.relu4 = nn.ReLU()
            # self.fc = nn.Linear(128, 5)
            self.classifier = nn.Linear(128, output_size)
            return

        self.testskip=False
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
        if(self.testskip==True):
            x = self.f1(x)
            x_skip_1 = self.l1(x)
            x = self.bn1(x_skip_1)
            x = self.relu1(x)
            x = self.l2(x)
            x = self.bn2(x)
            x = self.relu2(x + x_skip_1)
            x_skip_2 = self.l3(x)
            x = self.bn3(x_skip_2)
            x = self.relu3(x)
            x = self.l4(x)
            x = self.bn4(x)
            x = self.relu4(x + x_skip_2)
            x = self.classifier(x)
            return x
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
