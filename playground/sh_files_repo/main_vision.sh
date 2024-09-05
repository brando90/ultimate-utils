##!/bin/bash

# -- set up vision job (a submission job is usually empty and has the root of the submission so you probably need your HOME env var)
export HOME=/home/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile
#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1

host_v=$(hostname)
if [ $host_v = vision-21.cs.illinois.edu ]; then
    conda activate metalearning11.1
else
    conda activate metalearningpy1.7.1c10.2
fi

# -- quick checks
#nvidia-smi
nvcc --version
hostname
echo $PATH
which python
#conda list

# run experiment
echo ---- Running your python main ----
python ~/automl-meta-learning/automl-proj/experiments/meta_learning/main_metalearning.py

echo ---- Running your python main ----
