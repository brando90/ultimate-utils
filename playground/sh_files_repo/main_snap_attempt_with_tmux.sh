# small attempt
# - get a job id for this tmux session
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo SLURM_JOBID = $SLURM_JOBID
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
export ERR_FILE=$PWD/main.sh.e$SLURM_JOBID

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
# pkill -9 reauth -u brando9

# - start tmux,
https://unix.stackexchange.com/questions/724877/custom-kerberos-tmux-doesnt-let-me-name-my-sessions-help-forcing-it
tmux new -s $SLURM_JOBID
# /afs/cs/software/bin/krbtmux new -s $SLURM_JOBID
# cat /afs/cs/software/bin/krbtmux

# - reauth
# /afs/cs/software/bin/reauth
echo $SU_PASSWORD | /afs/cs/software/bin/reauth
# echo 'Secret' | /afs/cs/software/bin/reauth
# echo 'totally secret password' | kinit user@DOMAIN.EDU
# to see reauth running
# top -u brando9

# - expt python script
python expts.py &
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE &
export JOB_PID=$!
echo JOB_PID = $JOB_PID

# - Echo tmux id, should be jobid
tmux display-message -p '#S'
echo SLURM_JOBID = $SLURM_JOBID

# - detach current tmux session
tmux detach &

# - wait for pid from python to be done, if done kill this tmux sess
wait $JOB_PID
tmux ls
tmux kill-session -t $SLURM_JOBID
# check it was killed
tmux ls

# - Done
echo "Done with bash script (experiment or dispatched daemon experiments). "

# real attempt

#!/bin/bash
# pkill -9 python

# - CAREFUL, if a job is already running it could do damage to it, rm reauth process, qian doesn't do it so skip it
# top -u brando9
# pkill -9 reauth -u brando9

# - set experiment id
export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))")
echo SLURM_JOBID = $SLURM_JOBID

# - start tmux, https://unix.stackexchange.com/questions/724877/custom-kerberos-tmux-doesnt-let-me-name-my-sessions-help-forcing-it
tmux new -s $SLURM_JOBID
# /afs/cs/software/bin/krbtmux new -s $SLURM_JOBID
# tmux is a new process so let's check that env variable is really there
echo SLURM_JOBID = $SLURM_JOBID

# -- setup up for condor_submit background script in vision-cluster
#source /etc/bashrc
#source /etc/profile
#source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile
source ~/.bashrc.user
echo HOME = $HOME

#export OUT_FILE="$($PWD)/main.sh.o$($SLURM_JOBID)_gpu$($CUDA_VISIBLE_DEVICES)_$(hostname)"
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
echo OUT_FILE = $OUT_FILE

#export OUT_FILE="$($PWD)/main.sh.err$($SLURM_JOBID)_gpu$($CUDA_VISIBLE_DEVICES)_$(hostname)"
export ERR_FILE=$PWD/main.sh.e$SLURM_JOBID
echo ERR_FILE = $ERR_FILE

#module load gcc/9.2.0
source cuda11.1

# activate conda
conda init bash
conda activate metalearning_gpu

# some quick checks
nvcc --version
hostname
which python
python -c "import uutils; print(uutils); uutils.hello()"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# - test shared
pwd .
realpath .

# - check wanbd dir
echo WANDB_DIR = $WANDB_DIR
echo TEMP = $TEMP

# - choose GPU
#export CUDA_VISIBLE_DEVICES=$(((RANDOM%8)))
#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
#export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=3
export CUDA_VISIBLE_DEVICES=4
#export CUDA_VISIBLE_DEVICES=5
#export CUDA_VISIBLE_DEVICES=6
#export CUDA_VISIBLE_DEVICES=7
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=4,5,7
#export CUDA_VISIBLE_DEVICES=4,5,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=1,2,3
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,6,7

echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.get_device_name(0));"

# - reauth
# /afs/cs/software/bin/reauth
echo $SU_PASSWORD | /afs/cs/software/bin/reauth
# echo 'Secret' | /afs/cs/software/bin/reauth
# echo 'totally secret password' | kinit user@DOMAIN.EDU
# to see reauth running
# top -u brando9

# -- Run Experiment
echo ---- Running your python main.py file ----

# hdb1 scaling expts with 5CNN
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size
#nohup python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size > $OUT_FILE 2> $ERR_FILE
#nohup python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_hdb1_adam_cs_filter_size > $OUT_FILE 2> $ERR_FILE &

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_hdb1_5cnn_adam_cl_filter_size --filter_size 4 > $OUT_FILE 2> $ERR_FILE &
export JOB_PID=$!
echo JOB_PID = $JOB_PID
# tail -f main.sh.o$SLURM_JOBID
cat $OUT_FILE

# - Echo tmux id, should be jobid
tmux display-message -p '#S'
echo SLURM_JOBID = $SLURM_JOBID

# - detach current tmux session
tmux detach &

# - wait for pid from python to be done, if done kill this tmux sess
wait $JOB_PID
tmux ls
tmux kill-session -t $SLURM_JOBID
tmux ls

# - Done
echo "Done with bash script (experiment or dispatched daemon experiments). "

#pip install wandb --upgrade
