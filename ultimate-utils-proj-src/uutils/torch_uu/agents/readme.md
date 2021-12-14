# Recommended pattern

To allow the code to be extendible and re-usable we are going to adopt the uu-agent pattern.
In particular what it means is that we will not hardcode the agents training in the agent itself, but instead
have a function that takes in an optional agent and trains. The advantage is that we do NOT have to create an agent
to use the training functions (and other re-usable functions e.g. inference, test).

e.g.

```python
class Agent:

    def __init__(self):
        pass  # default agent doesn't need anything!

    def train(self, dataloaders: dict):
        train(self, dataloaders)
```

The advantage of this vs only using `train(self, dataloaders)` is that now you can custumize `Agent` and when other
cose that uses an agent (e.g. a meta-learner that has a forward) then code can use that agent. 
Note however, most code is encouraged to find a way to not have to create an agent if it can be avoided.