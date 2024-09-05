# Recommended pattern

To allow the code to be extendible and re-usable we are going to adopt the uu-agent pattern.
In particular what it means is that we will not hardcode the agents training in the agent itself, but instead
have a function that takes in an optional agent and trains. The advantage is that we do NOT have to create an agent
to use the training functions (and other re-usable functions e.g. inference, test).

e.g.

```python
import torch.nn as nn

class Agent(nn.Module):

    def __init__(self):
        pass  # default agent doesn't need anything!

    def train(self):
        train(self.model, self.dataloaders, self.opt, self.scheduler)
```

The advantage of this vs only using `train(self, dataloaders)` is that now you don't have to create an `Agent` if you
want to re-use the train code. If the train could needed to use custom code then you can put it in it's own train 
function and then call train. If the `train` code "needed" an agent to do a forward pass, then you can use
the `Agent` to wrap the model (e.g. in a meta-learner) and then you can write the custom code in the agent.
If you do this make sure you overwrite the `nn.Module` functions so that it functions correctly e.g.
`.load_state_dict, .forward` come to mind. Make sure your model processes the batch e.g. passing to gpu inside it's
own code properly.