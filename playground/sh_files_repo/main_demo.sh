#!/bin/bash

echo JOB STARTED

# a submission job is usually clean and has the root of the submission so you probably need this env var
export HOME=/home/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

conda activate metalearningpy1.7.1c10.2
#conda activate metalearning11.1

module load cuda-toolkit/10.2
#module load cuda-toolkit/11.1

#nvidia-smi
nvcc --version
##conda list
echo hostname
echo $PATH
which python
#echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

# - run script
#python ~/ultimate-utils/ultimate-utils-proj-src/uutils/torch/distributed.py
#echo MINIMUM TEST WORKS
python ~/ML4Coq/ml4coq-proj/embeddings_zoo/tree_nns/main_brando.py

echo JOB ENDED
