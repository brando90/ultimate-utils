#!/bin/bash

#conda install -y -c conda-forge tensorboard

echo "Friendly reminder to download the logs from remote first"

alias tbb="sh ${HOME}/ultimate-utils/run_tb.sh"

# - create the cmd that does the prefix substitution (once given a string as an argument at thend)
path2cmd_that_puts_right_prefix=$(pwd)/ultimate-utils/ultimate-utils-proj-src/execute_tensorboard.py

# - get the path from the local directory to the logs for your tb experiments
# the line bellow gets the output/string from the python command
local_dir2current_logs=$(python $path2cmd_that_puts_right_prefix $1)

# -- run tb using the remote data but locally
echo $local_dir2current_logs
tensorboard --logdir local_dir2current_logs
