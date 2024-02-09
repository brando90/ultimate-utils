#!/bin/bash
# - snap: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers and support il-action@cs.stanford.edu
# - live server stats: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-gpu-servers-stats
#8 a100 80GB
ssh brando9@ampere1.stanford.edu
ssh brando9@skampere1.stanford.edu
#10 Quadro RTX 8000 48GB
ssh brando9@hyperturing1.stanford.edu
ssh brando9@hyperturing2.stanford.edu
#10 RTX A4000 16GB
ssh brando9@mercury1.stanford.edu
ssh brando9@mercury2.stanford.edu

tput rmcup

source $AFS/.bashrc.lfs
conda activate beyond_scale
export CUDA_VISIBLE_DEVICES=5; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))"); echo $SLURM_JOBID;
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES; echo SLURM_JOBID = $SLURM_JOBID; echo hostname = $(hostname)
ulimit -n 120000; ulimit -Sn; ulimit -Hn;
nvidia-smi;hostname
(echo "GPU_ID PID UID APP" ; for GPU in 0 1 2 3 ; do for PID in $( nvidia-smi -q --id=${GPU} --display=PIDS | awk '/Process ID/{print $NF}') ; do echo -n "${GPU} ${PID} " ; ps -up ${PID} | awk 'NR-1 {print $1,$NF}' ; done ; done) | column -t

export CUDA_VISIBLE_DEVICES=3,4,5,6; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))"); echo $SLURM_JOBID;
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; export SLURM_JOBID=$(python -c "import random;print(random.randint(0, 1_000_000))"); echo $SLURM_JOBID;
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES; echo SLURM_JOBID = $SLURM_JOBID; echo hostname = $(hostname)

python -c "import uutils; uutils.torch_uu.gpu_test()"
python -c "import torch; print(torch.cuda.get_device_capability());print('if >=8 you can use bfloat16');"
python -c "import torch; print(torch.bfloat16);"

# - start krbtmux
#pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;
krbtmux
reauth

tmux ls
tmux new -s 1
reautexport CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')h

source $AFS/.bashrc
conda activate beyond_scale
export CUDA_VISIBLE_DEVICES=6
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
export HF_TOKEN=$(cat ~/keys/brandos_hf_token.txt)
echo $HF_TOKEN

# -- Run
# python ~/beyond-scale-language-data-diversity/src/diversity/div_coeff.py

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
python ~/beyond-scale-language-data-diversity/src/training/train.py

# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
# python ~/beyond-scale-language-data-diversity/src/diversity/embeddings/div_act_based.py


export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
python ~/beyond-scale-language-data-diversity/src/training/eval.py

# -- other option is to run `echo $SU_PASSWORD | /afs/cs/software/bin/reauth` inside of python, right?
export JOB_PID=$!
echo $OUT_FILE
echo $ERR_FILE
echo JOB_PID = $JOB_PID
echo SLURM_JOBID = $SLURM_JOBID
