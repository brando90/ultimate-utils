#!/bin/bash

# used this before dgx used slurm

echo -- Start my submission file

export SLURM_JOBID=$(((RANDOM)))
echo SLURM_JOBID = $SLURM_JOBID

#export CUDA_VISIBLE_DEVICES=$(((RANDOM%8)))
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=3
#export CUDA_VISIBLE_DEVICES=4
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

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo torch.cuda.device_count is:
python -c "import torch; print(torch.cuda.device_count())"
python -c "import uutils; uutils.torch_uu.gpu_test()"

echo ---- Running your python main ----

pip install wandb --upgrade
/home/miranda9/miniconda3/envs/meta_learning_a100/bin/python -m pip install --upgrade pip

#export SLURM_JOBID=-1
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_metalearning.py --manual_loads_name manual_load_cifarfs_resnet12rfs_maml > $OUT_FILE &

# - SL
export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_600 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_600 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam_cl_600 > $OUT_FILE &

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_resnet_rfs_mi_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_rfs_resnet12rfs_adam_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_rfs_resnet12rfs_adam_cl_600 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_rfs_4cnn_adam_cl_200 > $OUT_FILE &


#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_sgd_cl_600 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_rfs_5cnn_sgd_cl_600 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_resnet12rfs_sgd_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_resnet12rfs_sgd_cl_200 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_500 > $OUT_FILE &

#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_4cnn_hidden_size_1024_adam_rfs_1000 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_4cnn_hidden_size_1024_adam_no_scheduler_1000 > $OUT_FILE &
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_cifarfs_4cnn_hidden_size_1024_adam_no_scheduler_many_epochs > $OUT_FILE &

python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_sl_with_ddp.py --manual_loads_name sl_mi_rfs_5cnn_adam > $OUT_FILE &
echo pid = $!
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
echo SLURM_JOBID = $SLURM_JOBID

# - MAML
#export OUT_FILE=$PWD/main.sh.o$SLURM_JOBID
#python -m torch.distributed.run --nproc_per_node=5 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_cifarfs_rfs_adam_cl_100k > $OUT_FILE &
#python -m torch.distributed.run --nproc_per_node=5 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_4CNNl2l_cifarfs_rfs_adam_cl_70k > $OUT_FILE &

#python -m torch.distributed.run --nproc_per_node=8 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_adam_cl_100k > $OUT_FILE &
#python -m torch.distributed.run --nproc_per_node=8 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNNl2l_mi_rfs_adam_cl_70k > $OUT_FILE &


#python -m torch.distributed.run --nproc_per_node=5 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNNl2l_mi_rfs_sgd_cl_100k > $OUT_FILE &
#python -m torch.distributed.run --nproc_per_node=5 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNNl2l_cifarfs_rfs_sgd_cl_100k > $OUT_FILE &

#python -m torch.distributed.run --nproc_per_node=5 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_cifarfs_rfs_sgd_cl_100k > $OUT_FILE &
#python -m torch.distributed.run --nproc_per_node=5 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_rfs_sgd_cl_100k > $OUT_FILE &

#python -m torch.distributed.run --nproc_per_node=3 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_4CNNl2l_1024_cifarfs_rfs_adam_cl_100k > $OUT_FILE &
#python -m torch.distributed.run --nproc_per_node=4 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_resnet12rfs_mi_adam_no_scheduler_100k > $OUT_FILE &

#python -m torch.distributed.run --nproc_per_node=3 ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_dist_maml_l2l.py --manual_loads_name l2l_5CNN_mi_adam_filter_size > $OUT_FILE &
#echo pid = $!
#echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
#echo SLURM_JOBID = $SLURM_JOBID

# - Data analysis
#python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main2_distance_sl_vs_maml.py
python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/_main_distance_sl_vs_maml.py

echo -- Done submitting job in dgx A100-SXM4-40G
