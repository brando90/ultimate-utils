#!/bin/bash

echo JOB STARTED

# a submission job is usually empty and has the root of the submission so you probably need your HOME env var
export HOME=/home/miranda9
# to have modules work and the conda command work
source /etc/bashrc
source /etc/profile
source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/.bash_profile

# https://linuxize.com/post/how-to-compare-strings-in-bash/#:~:text=When%20comparing%20strings%20in%20Bash,%5B%5B%20command%20for%20pattern%20matching.
conda activate metalearningpy1.7.1c10.2
#conda activate coq-tactician-graph

host_v=$(hostname)
if [ $host_v = vision-21.cs.illinois.edu ]; then
    conda activate metalearning11.1
else
    conda activate metalearningpy1.7.1c10.2
fi

#module load cuda-toolkit/10.2
module load cuda-toolkit/11.1

#nvidia-smi
nvcc --version
#conda list
hostname
echo $PATH
which python

# - run script
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic
python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic --term_encoder_embedding_dim 1024
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic --split test_debug
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic --split train
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic --split train --train_set_length 23_644
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic --split train --train_set_length 163_218

#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode syntactic --num_epoch 2
#python -u ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --mode semantic --batch_siz 64 --debug
#python ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --debug --serial --num_epochs 2 --batch_size 2 --term_encoder_embedding_dim 8
#python ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --debug --num_epochs 2 --batch_size 2 --term_encoder_embedding_dim 8

# MEMORY debugging
#python -m memory_profiler ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --serial
#mprof run ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --serial
#mprof run ~/ML4Coq/ml4coq-proj-src/embeddings_zoo/tree_nns/main_brando.py --serial --batch_size 128 --term_encoder_embedding_dim 32

echo JOB ENDED