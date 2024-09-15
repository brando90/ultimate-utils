# Training

# 1 GPU train

Run:
```bash
# if you need to kill a proc
nvidia-smi
kill -9 <pid>

source ~/.virtualenvs/snap_cluster_setup/bin/activate
cd ~/snap-cluster-setup
python py_src/train/simple_train2.py
```

# 1-8 GPUs 1 node train

Run:
```bash
source ~/.virtualenvs/snap_cluster_setup/bin/activate
cd ~/snap-cluster-setup
accelerate launch --config_file ~/snap-cluster-setup/configs/accelerate/ddp/ddp_config_hf_acc.yaml py_src/train/simple_train2.py  
```