# Run Wandb Sweep in python but use a yaml config file

## Main take-aways

- you need to specify the path to config somewhere. 
- If you run sweep & agents in python then:
  - you specify config in wand.sweep(...) call
  - then agent uses sweep id to choose hyperparms to run your train file

- note:
  - for now if you have usl & maml (different methods) on different files, then you need to have a separate sweep for each method,
  likely each file/train method specified in it's own config
  - shouldn't be to hard to create configs thanks to GPT-4. 