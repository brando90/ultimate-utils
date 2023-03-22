# Ultimate Utils Wandb Sweeps Tutorial

## Sweeps: An Overview

Sweeps: An Overview
Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

Define the sweep: we do this by creating a dictionary or a YAML file that specifies the parameters to search through, the search strategy, the optimization metric et all.

Initialize the sweep: with one line of code we initialize the sweep and pass in the dictionary of sweep configurations: sweep_id = wandb.sweep(sweep_config)

Run the sweep agent: also accomplished with one line of code, we call wandb.agent() and pass the sweep_id to run, along with a function that defines your model architecture and trains it: wandb.agent(sweep_id, function=train)


## Basic usage

```python
import wandb
import pprint
import math
import torch

sweep_config: dict = {
 "name": "my-ultimate-sweep",
 "metric": 
     {"name": "train_loss", 
      "goal": "minimize" }
    ,
 "method": "random",
 "parameters": None,  # not set yet
}
# paramameters == hyperparameters here in sweeps
parameters = {
    'optimizer': {
        'values': ['adam', 'adafactor']}
    ,
    'scheduler': {
        'values': ['cosine', 'none']}  # todo, think how to do
    ,
    'lr': {        
        # # a flat distribution between 0 and 0.1
        # 'distribution': 'uniform',
        # 'min': 0,
        # 'max': 0.1
        "distribution": "log_uniform",
        "min": math.log(1e-6),
        "max": math.log(0.2)}
    ,
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    ,
    # it's often the case that some hps we don't want to vary in the run e.g. num_its
    'num_its': {'value': 200_000}
    }
sweep_config['parameters'] = parameters

pprint.pprint(sweep_config)

def my_train_func():
 # read the current value of parameter "a" from wandb.config
    wandb.init(project="project-name", entity="entity-name")
    lr = wandb.config.lr

    train_loss: float = 8.0
    for i in range(5):
        # get a random update step from the range [0.0, 1.0] using torch
        update_step: float = lr * torch.rand(1).item()
        wandb.log({"lr": lr, "train_loss": train_loss - update_step})

# We can wind up a Sweep Controller by calling wandb.sweep with the appropriate sweep_config and project name.
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
# relation to above: wandb sweep config.yaml ?

# run the sweep, The cell below will launch an agent that runs train 5 times, usingly the randomly-generated hyperparameter values returned by the Sweep Controller.
wandb.agent(sweep_id, function=my_train_func, count=5)
````

## Using one of Ultimate Utils' main function

todo: use the hdb4 maml l2l main with sweeps for lr (and maybe optimizer?)

idea/plan:
- run a sweep of the above code without anything complicated
- do `wandb.agent(sweep_id, function=train)`, pass args to train in this wandb.agent. Then make sure you get the right
hp by doing wandb.config.lr etc. or the ones being tested. Put wandb.config.lr to args dict. In load_args, if sweeps
put the args.lr = wandb.config.lr. Then train it.
- change the args to receive optional sweeps. I think it makes most sense to take a yaml file per sweep so name it
hdb4 maml l2l sweep. Put it in a sweeps folder in div folder (or here for demo). Pass the file path of the sweep config
in the argparse. Default argparse is none. Create both meta-L & SL argparse flag.
- Figure out how to use different CUDAs in same server in SNAP.
- would be nice to compare with rylans
- (later using more servers in sweep in SNAP)

## References

ref:
    - youtube video: https://www.youtube.com/watch?v=9zrmUIlScdY
    - colab: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb
    - doc: https://docs.wandb.ai/ref/python/sweep 