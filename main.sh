#!/bin/bash -l

echo JOB STARTED

source /etc/bashrc
#source /etc/profile
#source /etc/profile.d/modules.sh
#source ~/.bashrc
#source ~/.bash_profile

#conda activate metalearning11.1

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=<YOUR_IFACE>

#export SHELL=/bin/bash
#export _CONDOR_SHELL=/bin/bash

#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1

#/usr/local/cuda/bin:/home/miranda9/miniconda3/envs/automl-meta-learning/bin:/home/miranda9/miniconda3/condabin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/home/miranda9/my_bins:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/miranda9/my_bins:/home/miranda9/bin
#/usr/local/cuda/bin:/home/miranda9/miniconda3/envs/automl-meta-learning/bin:/home/miranda9/miniconda3/condabin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/my_bins:/bin:/my_bins:/my_bins:/bin
# export PATH=/usr/local/cuda/bin:/home/miranda9/miniconda3/envs/automl-meta-learning/bin:/home/miranda9/miniconda3/condabin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/home/miranda9/my_bins:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/miranda9/my_bins:/home/miranda9/bin

nvidia-smi
nvcc --version
#conda list
echo $PATH
which python
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
#echo ---
#env
#echo ---

# - run script
python ~/ultimate-utils/ultimate-utils-project/uutils/torch/distributed.py
#python ~/ML4Coq/ml4coq-proj/embeddings_zoo/tree_nns/main_brando.py
#python ~/ML4Coq/ml4coq-proj/embeddings_zoo/tree_nns/main_brando.py --debug --num_epochs 5 --batch_size 2 --term_encoder_embedding_dim 8
# python ~/ultimate-utils/ultimate-utils-project/uutils/torch/distributed.py

echo JOB ENDED
