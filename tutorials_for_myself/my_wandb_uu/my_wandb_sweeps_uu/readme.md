# Ultimate Utils Wandb Sweeps Tutorial
Recommended: see your sweep_in_python_yaml_config folder here in uutils. 

## Sweeps: An Overview
Sweeps: An Overview
Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

Define the sweep: we do this by creating a dictionary or a YAML file that specifies the parameters to search through, the search strategy, the optimization metric et all.

Initialize the sweep: with one line of code we initialize the sweep and pass in the dictionary of sweep configurations: sweep_id = wandb.sweep(sweep_config)

Run the sweep agent: also accomplished with one line of code, we call wandb.agent() and pass the sweep_id to run, along with a function that defines your model architecture and trains it: wandb.agent(sweep_id, function=train)

## References

ref:
    - Recommended: see your sweep_in_python_yaml_config folder here in uutils. 
    - youtube video: https://www.youtube.com/watch?v=9zrmUIlScdY
    - colab: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb
    - doc: https://docs.wandb.ai/ref/python/sweep 