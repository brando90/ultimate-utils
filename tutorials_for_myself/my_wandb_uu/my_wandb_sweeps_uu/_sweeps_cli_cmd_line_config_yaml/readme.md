
# Quick Demo

```
export SWEEPID=$(wandb sweep config.yaml)
NUM=10
wandb agent train_sl.py --count $NUM $SWEEPID
wandb agent train_maml.py --count $NUM $SWEEPID
```

# TODO

todo:
    - using two files in the same sweep (config): https://community.wandb.ai/t/how-do-i-have-two-different-run-files-to-log-to-the-same-sweep/4150
