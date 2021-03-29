#!/bin/bash

echo JOB STARTED

source /etc/bashrc
#source /etc/profile
#source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

conda activate metalearning11.1

#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1

nvidia-smi
nvcc --version
#conda list
echo $PATH
which python
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

# - run script
python ~/ultimate-utils/ultimate-utils-project/uutils/torch/distributed.py
echo MINIMUM TEST WORKS
# python ~/ML4Coq/ml4coq-proj/embeddings_zoo/tree_nns/main_brando.py

echo JOB ENDED
