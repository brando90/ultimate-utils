#!/bin/bash

# Tests
# sh /Users/brando/ultimate-utils/run_tb.sh /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb
# tbb /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb
# local_dir=$(python -c "import sys; print('ho' + str(sys.argv[1]))" $1)

# -- transform the remote tb path to local path
local_dir=$(python /Users/brando/ultimate-utils/ultimate-utils-project/execute_tensorboard.py $1)
# local_dir=$(python -c "from pathlib import Path; import sys; print(Path(sys.argv[1].replace('/home/miranda9/', '~/')).expanduser())" $1)

# -- run the new remote tb locally
echo $local_dir
tensorboard --logdir $local_dir
