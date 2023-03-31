Rylan — Today at 2:19 PM
Ok, so suppose you have some W&B sweep e.g.:

program: notebooks/vision/train.py
project: emergence-vision-train
method: grid
metric:
  goal: minimize
  name: test_loss
parameters:
  batch_size_times_seq_len_train:
    values: [250]
  batch_size_times_seq_len_test:
    values: [250]
  dataset:
    values: ['cifar10']
  network_type:
    values: ["conv+affine"]
  n_conv_filters_per_layer:
    values: [
      "[32, 64, 64, 64]",
      "[32, 64, 64, 128]",
      "[32, 64, 128, 128]",
      "[32, 64, 128, 256]",
      "[]",
      "[32]",
      "[32, 32]",
      "[32, 64]",
      "[64, 64]",
      "[64, 128]",
      "[32, 32, 64]",
      "[32, 64, 64]",
      "[32, 64, 128]",
    ]
  n_affine_units_per_layer:
    values: [
      "[32]",
      "[32, 32]",
      "[64, 32]",
      "[64, 64]",
      "[128, 64]",
      "[128, 128]",
      "[256, 128]",
      "[256, 256]",
      "[512, 256]",
      "[1024, 512]",
      "[64, 64, 64]",
      "[128, 64, 64]",
      "[128, 128, 64]",
      "[256, 128, 64]",
      "[256, 256, 64]",
      "[256, 256, 128]",
      "[1024, 512, 128]",
      "[1024, 1024, 512]",
      "[1024, 1024, 512, 128]",
    ]
  n_epochs:
    values: [40]
  seq_len:
    values: [ 5 ]
  task:
    values: ["classification"]
  use_max_pool:
    values: [ True, False ]
 
At the command line, call wandb sweep <path to that ^ yaml file>. This will output a Sweep ID e.g. v0cepfg0 
Then I have a bash script to launch a single agent (this is a SLURM script, but SLURM isn't necessary whatsoever):
#!/bin/bash
#SBATCH -n 1                    # one node
#SBATCH --mem=8G               # RAM
#SBATCH -p gpu                  # Run on GPU partition
#SBATCH -G 1                    # Request 1 GPU
#SBATCH -C GPU_MEM:16GB         # Request 16GB GPU memory
#SBATCH --time=36:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

module load math  # I think this is necessary on Sherlock for cudnn
module load devel  # I think this is necessary on Sherlock for cuda
module load cuda/11.5.0
module load cudnn/8.6.0.163

DATA='/home/groups/sanmi/rschaef/KoyejoLab-Rotation'

id=${1}

# Activate virtual environment.
source /home/groups/sanmi/rschaef/KoyejoLab-Rotation/emergence_venv/bin/activate

# This makes the code inside koyejo_lab available to import.
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

wandb agent rylan/emergence-vision-train/${id}
Suppose that file is called train_gpu.sh
Then I run the script, either directly or via slurm's sbatch e.g. sbatch train_gpu.sh <sweep ID>
so if I want to launch multiple SLURM jobs, I'll do something like:
for i in {1..20}
do
sbatch train_gpu.sh v0cepfg0
done
That's it!