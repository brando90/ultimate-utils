#!/bin/bash

local_dir=$(python /Users/brando/ultimate-utils/ultimate-utils-project/execute_tensorboard.py $1)
tensorboard --logdir $local_dir
