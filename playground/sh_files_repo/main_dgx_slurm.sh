#!/bin/bash
#SBATCH --job-name="job_bmg"
#SBATCH --output="main.sh.o%j.%N"
#SBATCH --partition=x86
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=16
#SBATCH --threads-per-core=2
#SBATCH --mem-per-cpu=4000
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1

# email: https://www.hpc.caltech.edu/documentation/faq/why-am-i-not-getting-job-notification-emails-how-do-i-make-them-stop

# hal-dgx website: https://wiki.ncsa.illinois.edu/display/ISL20/Access+hal-dgx+and+overdrive+with+hal-login3+Node

# -- Running 1 cpu core and many cpu cores:
srun --partition=x86 --time=48:00:00 --pty /bin/bash
srun --partition=x86 --time=48:00:00 --nodes=1 --ntasks-per-node=32 --sockets-per-node=1 --cores-per-socket=16 --threads-per-core=2 --mem-per-cpu=4000 --wait=0 --export=ALL --pty /bin/bash

# -- Example 0
#Request 1x GPU along with 32x CPU cores for 48 hours
srun --partition=x86 --time=48:00:00 --nodes=1 --ntasks-per-node=32 --sockets-per-node=1 --cores-per-socket=16 --threads-per-core=2 --mem-per-cpu=4000 --wait=0 --export=ALL --gres=gpu:a100:1 --pty /bin/bash
#srun --partition=x86 --time=48:00:00 --nodes=1 --ntasks-per-node=128 --sockets-per-node=1 --cores-per-socket=16 --threads-per-core=2 --mem-per-cpu=4000 --wait=0 --export=ALL --gres=gpu:a100:4 --pty /bin/bash
#srun --partition=x86 --time=48:00:00 --nodes=1 --ntasks-per-node=32 --sockets-per-node=1 --cores-per-socket=16 --threads-per-core=2 --mem-per-cpu=4000 --wait=0 --export=ALL --gres=gpu:a100:8 --pty /bin/bash

# -- Example 1
# Request 2x GPU along with 64x CPU cores for 12 hours

# -- Example 2
# Request 4x GPU along with 128x CPU cores for 24 hours


# -- Running sbatch
# sbatch ~/diversity-for-predictive-success-of-meta-learning/main_dgx_slurm.sh
# sbatch main_dgx_slurm.sh

# - Print stuff
cd ~
echo STARTING `date`
# why srun inside sbatch: https://stackoverflow.com/questions/53636752/slurm-why-use-srun-inside-sbatch
srun hostname
nvidia-smi
python -c "import torch;print(torch.version.cuda)"
nvcc -V

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo torch.cuda.device_count is:
python -c "import torch; print(torch.cuda.device_count())"
echo ---- Running your python main ----

# -- Run Experiment
# - SL
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_32_filter_size
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_16_filter_size
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_8_filter_size
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_4_filter_size

# - MAML
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_32_filter_size
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_16_filter_size
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_8_filter_size
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size_4_filter_size

#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k
python -m torch.distributed.run --nproc_per_node=1 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_hdb1_100k
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet18task2vec_hdb1_100k
#python -m torch.distributed.run --nproc_per_node=1 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet18task2vec_hdb1_100k

# - Data analysis
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/_main_distance_sl_vs_maml.py
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/diversity/task2vec_based_metrics/diversity_task2vec/diversity_for_few_shot_learning_benchmark.py

# - task2vec div
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py

# - task2vec div individual jobs
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_cifarfs_resnet18_random
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_cifarfs_resnet34_pretrained_imagenet
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_cifarfs_resnet34_random

# - small hdb1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/diversity/task2vec_based_metrics/diversity_task2vec/small_datasets_example.py

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/diversity/task2vec_based_metrics/diversity_task2vec/high_diversity_benchmarks.py

# - other wrap up
/home/miranda9/miniconda3/envs/meta_learning_a100/bin/python -m pi install --upgrade pip
pip install wandb --upgrade
conda update conda
conda update --all
pip install --upgrade pip
pip3 install --upgrade pip

echo -- Done submitting job in dgx A100-SXM4-40G
