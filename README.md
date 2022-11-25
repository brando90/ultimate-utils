# Ultimate-utils

Ulitmate-utils (or uutils) is collection of useful code that Brando has collected through the years that has been useful accross his projects.
Mainly for machine learning and programming languages tasks.

## Installing Ultimate-utils

## Standard pip install [Recommended]

If you are going to use a gpu the do this first before continuing 
(or check the offical website: https://pytorch.org/get-started/locally/):
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --upgrade
```
Otherwise, just doing the following should work.
```bash
pip install ultimate-utils
```
If that worked, then you should be able to import is as follows:
```python
import uutils
```
note the import statement is shorter than the library name (`ultimate-utils` vs `uutils`).

Note, for an older version of uutils you might need to downgrade pytorch related stuff by doing:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Manual installation [for Development]

To use uutils first get the code from this repo (e.g. fork it on github):

```
git clone git@github.com:brando90/ultimate-utils.git
```

Then install it in development mode in your python env with python >=3.9
(read `modules_in_python.md` to learn about python envs).
E.g. create your env with conda:

```
conda create -n uutils_env python=3.9
conda activate uutils_env
```

Then install uutils in edibable mode and all it's depedencies with pip in the currently activated conda environment:

```
pip install -e ~/ultimate-utils/ultimate-utils-proj-src
```

No error should show up from pip.
To test the installation uutils do:

```
python -c "import uutils; uutils.hello()"
python -c "import uutils; uutils.torch_uu.hello()"
```

it should print something like the following:

```

hello from uutils __init__.py in:
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>


hello from torch_uu __init__.py in:
<module 'uutils.torch_uu' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/__init__.py'>

```

To test (any) pytorch do:
```
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```
output:
```
(meta_learning_a100) [miranda9@hal-dgx diversity-for-predictive-success-of-meta-learning]$ python -c "import uutils; uutils.torch_uu.gpu_test()"
device name: A100-SXM4-40GB
Success, no Cuda errors means it worked see:
out=tensor([[ 0.5877],
        [-3.0269]], device='cuda:0')
(meta_learning_a100) [miranda9@hal-dgx diversity-for-predictive-success-of-meta-learning]$ python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
device name: A100-SXM4-40GB
Success, torch works with whatever device is shown in the output tensor:
out=tensor([[-1.9061],
        [ 1.3525]], device='cuda:0')

```

GPU TEST: To test if pytorch works with gpu do (it should fail if no gpus are available):
```
python -c "import uutils; uutils.torch_uu.gpu_test()"
```
output should be something like this:
```
(meta_learning_a100) [miranda9@hal-dgx diversity-for-predictive-success-of-meta-learning]$ python -c "import uutils; uutils.torch_uu.gpu_test()"
device name: A100-SXM4-40GB
Success, no Cuda errors means it worked see:
out=tensor([[ 0.5877],
        [-3.0269]], device='cuda:0')
```

### [Adavanced] If using pygraphviz functions 

If you plan to use the functions that depend on `pygraphviz` you will likely need to install `graphviz` first. 
On mac, `brew install graphviz`. 
On Ubuntu, `sudo apt install graphviz`. 

Then install `pygraphviz` with 
```
pip install pygraphviz
```

If the previous steps didn't work you can also try installing using conda
(which seems to install both `pygraphviz and `graphviz`):
```
conda install -y -c conda-forge pygraphviz
```
to see details on that approach see the following stack overflow link question: 
https://stackoverflow.com/questions/67509980/how-does-one-install-pygraphviz-on-a-hpc-cluster-without-errors-even-when-graphv

To test if pygraphviz works do:
```
python -c "import pygraphviz"
```

Nothing should return if successful.

## Contributing

Feel free to push code with pull request.
Please include at least 1 self-contained test (that works) before pushing.

### How modules are imported in a python project

Read the `modules_in_python.md` to have an idea of the above development/editable installation commands. 

## Executing tensorboard experiment logs from remote

- visualize the remote logs using pycharm and my code (TODO: have the download be automatic...perhaps not needed)

1. Download the code from the cluster using pycharm remote
2. Then copy paste the *remote path* (from pycharm, browse remote)
3. Using the copied path run `tbb path2log` e.g. `tbbb /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb`

to have `tbbb` work as the command add to your `.zshrc` (or `.bashrc`):
```
alias tbb="sh ${HOME}/ultimate-utils/run_tb.sh"
```

then the command `tbb path2log` should work.

ref: see files
- https://github.com/brando90/ultimate-utils/blob/master/run_tb.sh
- https://github.com/brando90/ultimate-utils/blob/master/ultimate-utils-proj-src/execute_tensorboard.py

## Pushing to pypi

For full details see
```
~/ultimate-utils/tutorials_for_myself/pushing_to_pypi/README.md
```
For quick push do:
```bash
cd ~/ultimate-utils/
rm -rf build
rm -rf dist
cd ~/ultimate-utils/
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```
then:
```bash
cd ~/ultimate-utils/
rm -rf build
rm -rf dist
```

## Testing pip install with docker

Create & pull image:
```
docker run -ti continuumio/miniconda3
```
then do
```
pip install ultimate-utils
```
it should be installed. You can import and test the print as mentioned above. 

## Citation
If you use this implementation consider citing us:

```
@software{brando2021ultimateutils,
    author={Brando Miranda},
    title={Ultimate Utils - the Ultimate Utils library for Machine Learning and Artificial Intelligence},
    url={https://github.com/brando90/ultimate-utils},
    year={2021}
}
```

A permanent link lives here: https://www.ideals.illinois.edu/handle/2142/112797


# Docker container
If you have issues with docker & space
run [warning this will remove all your docker images](https://stackoverflow.com/questions/44664900/oserror-errno-28-no-space-left-on-device-docker-but-i-have-space):
```bash
docker system prune -af
```

To build the docker image form the computer that will run the docker container
```bash
# for adm/x86
docker pull brandojazz/ultimate-utils:test
# for adm/x86
docker pull brandojazz/ultimate-utils:test_arm
```

or do the standard docker build for the image for an x86 machine:
```bash
# for adm/x86
docker build -t brandojazz/ultimate-utils:test ~/ultimate-utils/
# for arm/m1
docker build -f ~/ultimate-utils/Dockerfile_arm -t brandojazz/ultimate-utils:test_arm ~/ultimate-utils/

docker login
# for adm/x86
docker push brandojazz/ultimate-utils:test
# for arm/m1
docker push brandojazz/ultimate-utils:test_arm

docker images
```

Run container:
```bash
docker run -ti brandojazz/ultimate-utils:test_arm bash
```
or in development mode:
```bash
docker run -v ~/ultimate-utils:/home/bot/ultimate-utils \
           -v ~/data:/home/bot/data \
           -ti brandojazz/ultimate-utils:test_arm bash
```

Inside the docker container:
```bash
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```

### Appendix

#### Data storage at UIUC vision cluster IC

```bash
home -> /home/miranda9 (export HOME=/home/miranda9, echo $HOME)
code/git -> home
ssh -> home
push on save -> home
.bashrc -> home
conda -> shared (miniconda)

data & logs & ckpts -> home (mkdir ~/data)
wandb -> home but delete folder after each run (rm -rf $WANDB_DIR)
```

My goal is to put the large heavy stuff (e.g. conda, data, ) at `/shared/rsaas/miranda9/`.
Warning: due to the vpn if you run one of this commands and you lose connection you will have to do it again and might
have half a transfer of files. 
So run them in a job.sub command or re pull them from git then do a soft link.
```bash
mv ~/data /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/data ~/data 

mv ~/miniconda3 /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/miniconda3 ~/miniconda3

mv ~/data_folder_fall2020_spring2021 /shared/rsaas/miranda9/
ln -s /shared/rsaas/miranda9/data_folder_fall2020_spring2021 ~/data_folder_fall2020_spring2021

# --

# mv ~/diversity-for-predictive-success-of-meta-learning /shared/rsaas/miranda9
cd /shared/rsaas/miranda9/
git clone git@github.com:brando90/diversity-for-predictive-success-of-meta-learning.git 
# ln -s file1 link1
ln -s /shared/rsaas/miranda9/diversity-for-predictive-success-of-meta-learning ~/diversity-for-predictive-success-of-meta-learning 

mv ~/Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective ~/Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective 

mv ~/ultimate-anatome /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/ultimate-anatome ~/ultimate-anatome 

mv ~/ultimate-aws-cv-task2vec /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/ultimate-aws-cv-task2vec ~/ultimate-aws-cv-task2vec 

# mv ~/ultimate-utils /shared/rsaas/miranda9
cd /shared/rsaas/miranda9/
git clone git@github.com:brando90/ultimate-utils.git
ln -s /shared/rsaas/miranda9/ultimate-utils ~/ultimate-utils 

mv ~/pycoq /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/pycoq ~/pycoq 

mv ~/rfs /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/rfs ~/rfs 

mv ~/automl-meta-learning /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/automl-meta-learning ~/automl-meta-learning 

mv ~/wandb /shared/rsaas/miranda9
ln -s /shared/rsaas/miranda9/wandb ~/wandb 
```
to check real path (for soft links) do e.g. `realpath /home`.

# SNAP servers

Logging in/sshing:
```bash
ssh brando9@whale.stanford.edu
mkdir /dfs/scratch0/brando9
cd /dfs/scratch0/brando9
# ln -s file1 link1
#ln -s /dfs/scratch0/brando9 /afs/cs.stanford.edu/u/brando9/dfs/scratch0/brando9

# -- Titan X, 12 GB
ssh brando9@hyperion1.stanford.edu
ssh brando9@hyperion3.stanford.edu

# -- 2080, 11 GB
ssh brando9@turing1.stanford.edu
ssh brando9@turing2.stanford.edu
ssh brando9@turing3.stanford.edu
ssh brando9@turing4.stanford.edu

# -- A4000 16 GB [SK]
mercury1
mercury2

# -- RTX 8000 48GB, local storage 11TB
ssh brando9@hyperturing1.stanford.edu
ssh brando9@hyperturing2.stanford.edu

# -- a100 80GB, local storage 56TB
ssh brando9@ampere1.stanford.edu
...
ssh brando9@ampere4.stanford.edu

# -- a100 80GB, local storage ... [SK]
ssh brando9@...stanford.edu
```

My folder set up:
```bash
chmod a+x ~/diversity-for-predictive-success-of-meta-learning/main.sh
#chmod a+x /shared/rsaas/miranda9/diversity-for-predictive-success-of-meta-learning/main.sh

HOME -> /dfs/scratch0/brando9/ so that ckpting/logging works (wanbd logs to local lfs)
AFS -> /afs/cs.stanford.edu/u/brando9, export AFS=/afs/cs.stanford.edu/u/brando9, alias afs='cd $AFS' (where code is, need to fix data -> symlink to dfs data folder)
bashrc.user -> afs with symlynk to dfs (since HOME is at dfs), cat ~/.bashrc.user
ssh -> for now in afs because previous stuff broke when it was in dfs, need to test a little more
code/git -> code in /afs/cs.stanford.edu/u/brando9, so that push/sftp on save works in pycharm. Symlink them to afs so that they are visible at home=dfs/...
push on save -> root of projs /afs/cs.stanford.edu/u/brando9, make sure if you have root set automatically that you give the relative path on the deployment mapping (avoid putting root of hpc twice by accident)
wandb -> to local lfs of cluster, since that file really doesnt matter to me, just has to be somewhere so wandb works, see echo $LOCAL_MACHINE_PWD or/and ls $LOCAL_MACHINE_PWD, (if it fails put to dfs)

# conda ->  dfs and two back ups 
conda -> /dfs/scratch0/brando9 so any server has access to it, plus they are big so dont want to overwhelm afs (does symlinking conda to afs makes sense?), ls /dfs/scratch0/brando9/miniconda/envs & python -c "import uutils;uutils.get_home_pwd_local_machine_snap()" should work 

data -> /dfs/scratch0/brando9/ but with a symlink to /afs/cs.stanford.edu/u/brando9/data, TODO: https://intellij-support.jetbrains.com/hc/en-us/requests/4447850
# ln -s file1 link1
ln -s /afs/cs.stanford.edu/u/brando9/data /dfs/scratch0/brando9/data
```

Getting started:
```bash
# - create home
mkdir /dfs/scratch0/brando9
ls /dfs/scratch0/brando9

# note, export sets it for current shell and all subshells created
# - cd to home dfs
cd /dfs/scratch0/brando9
export HOME=/dfs/scratch0/brando9

# - to edit .bashrc.user
vim /afs/cs.stanford.edu/u/brando9/.bashrc.user
# ln -s file1 link1
ln -s /afs/cs.stanford.edu/u/brando9/.bashrc.user ~/.bashrc.user 

# - get hostname without stanford to set lfs for wandb
# export LOCAL_MACHINE_PWD=$(python -c "import uutils;uutils.get_home_pwd_local_machine_snap()")
export HOSTNAME=$(hostname)
export LOCAL_MACHINE_PWD="/lfs/${HOSTNAME::-13}/0/brando9"
echo LOCAL_MACHINE_PWD = $LOCAL_MACHINE_PWD
```

Installing conda:
```bash
# - in dfs when home is dfs
echo $HOME
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
ls -lah ~

export PATH="$HOME/miniconda/bin:$PATH"
echo $PATH
conda

source ~/miniconda/bin/activate
conda init
conda init bash
conda update -n base -c defaults conda
conda install conda-build

conda create -n metalearning_gpu python=3.9
conda activate metalearning_gpu
conda create -n iit_synthesis python=3.9
conda activate iit_synthesis
conda list

# - in afs when home is dfs but miniconda is installed to afs 
# WARNING (DISK QUOTA, DONT USE)
echo $AFS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $AFS/miniconda.sh
bash $AFS/miniconda.sh -b -p $AFS/miniconda
ls -lah $AFS

export PATH="$AFS/miniconda/bin:$PATH"
echo $PATH
conda

source $AFS/miniconda/bin/activate
conda init
conda init bash
conda update -n base -c defaults conda
conda install conda-build

conda create -n metalearning_gpu python=3.9
conda activate metalearning_gpu
conda info -e

conda create -n iit_synthesis python=3.9
conda activate iit_synthesis
conda info -e

conda list

# - installing full anaconda
echo $HOME
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh
#wget https://repo.continuum.io/conda/Anaconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
#bash ~/anaconda.sh -b -p $HOME/anaconda
nohup bash ~/anaconda.sh -b -p $HOME/anaconda > anaconda_install.out &
tail -f anaconda_install.out
ls -lah ~

export PATH="$HOME/anaconda/bin:$PATH"
echo $PATH
conda

source ~/anaconda/bin/activate
conda init
conda init bash
conda update -n base -c defaults conda
conda install conda-build

conda create -n metalearning_gpu python=3.9
conda activate metalearning_gpu

nohup sh ~/diversity-for-predictive-success-of-meta-learning/install.sh > div_install.out &
tail -f div_install.out

nohup sh ~/diversity-for-predictive-success-of-meta-learning/install.sh > div_install_miniconda.out &
tail -f div_install_miniconda.out

conda create -n iit_synthesis python=3.9
conda activate iit_synthesis
conda list


# - making a backup
# cp -R <source_folder> <destination_folder>
nohup cp -R anaconda anaconda_backup &
nohup cp -R miniconda miniconda_backup &

# /dfs/scratch0/brando9
nohup cp -R /dfs/scratch0/brando9/anaconda /dfs/scratch0/brando9/anaconda_backup &
nohup cp -R /dfs/scratch0/brando9/miniconda /dfs/scratch0/brando9/miniconda_backup &


nohup pip install -e ~/ultimate-utils/ &
nohup pip install -e ~/diversity-for-predictive-success-of-meta-learning/ &

mkdir /lfs/madmax/0/brando9
nohup cp -R ~/anaconda /lfs/madmax/0/brando9/anaconda_backup &
nohup cp -R ~/miniconda /lfs/madmax/0/brando9/miniconda_backup &

# - in case you need the path if you broke it
export PATH=/afs/cs.stanford.edu/u/brando9/miniconda/bin:/afs/cs.stanford.edu/u/brando9/miniconda/bin:/afs/cs.stanford.edu/u/brando9/miniconda/condabin:/afs/cs.stanford.edu/u/brando9/miniconda/bin:/dfs/scratch0/brando9/miniconda/bin:/usr/local/cuda-11.1/bin:/home/miranda9/miniconda3/bin:/usr/kerberos/sbin:/usr/kerberos/bin:/afs/cs/software/sbin:/afs/cs/software/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/dfs/scratch0/brando9/my_bins
# Use this command to check your afs quota and usage 
fs lq ~/
fs lq $AFS

# - try using pyenv
conda deactivate metalearning_gpu
virtualenv -p /usr/local/bin/python3.p metalearning_gpu_pyenv
source metalearning_gpu_pyenv/bin/activate
which pip

nohup sh ~/diversity-for-predictive-success-of-meta-learning/install.sh > div_install_pyenv.out &
tail -f div_install_pyenv.out
```

Git cloning your code & SHH keys:
```bash
echo $HOME
mkdir /afs/cs.stanford.edu/u/brando9/.ssh
cd /afs/cs.stanford.edu/u/brando9/.ssh
#touch ~/.ssh/id_ed25519
ssh-keygen -t ed25519 -C "brandojazz@gmail.com"
# copy paste the bellow path (needs absolute path to $HOME unfortuantely)
### (no for now) /dfs/scratch0/brando9/.ssh/id_ed25519
# just enter send it to /afs/cs.stanford.edu/u/brando9/.ssh/id_ed25519
# press enter twice or type passphrase twice

eval "$(ssh-agent -s)"
### (no for now) ssh-add /dfs/scratch0/brando9/.ssh/id_ed25519
ssh-add /afs/cs.stanford.edu/u/brando9/.ssh/id_ed25519

# add public key to your github
### (no for now) cat /dfs/scratch0/brando9/.ssh/id_ed25519.pub
cat /afs/cs.stanford.edu/u/brando9/.ssh/id_ed25519.pub
# copy contents of terminal
# paste to your github keys under settings under SSH and GPG keys
```

gitclone projs:
```bash
git clone git@github.com:brando90/ultimate-utils.git
git clone git@github.com:brando90/diversity-for-predictive-success-of-meta-learning.git
git clone git@github.com:brando90/pycoq.git
git clone git@github.com:FormalML/iit-term-synthesis.git
mkdir /dfs/scratch0/brando9/data
mkdir /dfs/scratch2/brando9/data

/afs/cs.stanford.edu/u/brando9
/dfs/scratch0/brando9
# ln -s file1 link1
ln -s /afs/cs.stanford.edu/u/brando9/ultimate-utils /dfs/scratch0/brando9/ultimate-utils
ln -s /afs/cs.stanford.edu/u/brando9/diversity-for-predictive-success-of-meta-learning /dfs/scratch0/brando9/diversity-for-predictive-success-of-meta-learning 
ln -s /afs/cs.stanford.edu/u/brando9/pycoq /dfs/scratch0/brando9/pycoq  
ln -s /afs/cs.stanford.edu/u/brando9/iit-term-synthesis /dfs/scratch0/brando9/iit-term-synthesis 
ln -s /dfs/scratch0/brando9/data /afs/cs.stanford.edu/u/brando9/data 
```

Using kerberos tmux (https://ilwiki.stanford.edu/doku.php?id=hints:long-jobs):
note:
- need to run krbtmux per server
- to reattach and other tmux commands use tmux prefix not krbtmux
```bash
# - run expt with krbtmux 
ssh ...

tmux ls
pkill -9 tmux -u brando9; pkill -9 krbtmux -u brando9; pkill -9 reauth -u brando9; pkill -9 python -u brando9; pkill -9 wandb-service* -u brando9;

# - start krbtmux
krbtmux
reauth
#source ~/.bashrc.user
# now you have a krbtmux that won't die, so you can 0. perhaps running reauth inside the main_krbtmux.sh script 1. run a python job for each krbtmux or 2. run multiple python jobs inside this krbtmux session

# - run expt
sh main_krbtmux.sh &
# - end of run expt

tmux detach
tmux attach -t 0
# 

# --
# if the above doesn't work then I will have to have a script that sets up the environment variables correctly for me & then manually type the python main:
# sh main_setup_jobid_and_out_files.sh
# python main.py &
# sh echo_jobid_and_out_files.sh

# type password, later how to pass it from terminal automatically https://unix.stackexchange.com/questions/724880/how-to-run-a-job-in-the-background-using-tmux-without-interacting-with-tmux-lik

# note that settip up the jobid & outfile for err and stdout can be done within python too if the current sh main_krbtmux.sh & doesn't work and we move to python main.py &

# -- some useful tmux commands & keystrokes
tmux new -s mysession
tmux kill-session -t mysession
tmux switch -t <session name or number>
tmux attach -t <session number>
tmux ls
tmux detach

C-b ) = next session
C-b ( = previous session
C-b [ = to scroll history
C-b d = detach from tmux without closing/killing the session but return normally to the terminal :) 
```
todo: https://unix.stackexchange.com/questions/724880/how-to-run-a-job-in-the-background-using-tmux-without-interacting-with-tmux-lik



Using nohup for long running jobs in snap TODO: https://unix.stackexchange.com/questions/724902/how-does-one-send-new-commands-to-run-to-an-already-running-nohup-process-or-run 
```bash
TODO bellow doesn't work
##nohup sh -c 'echo $SU_PASSWORD | /afs/cs/software/bin/reauth; python -u ~/diversity-for-predictive-success-of-meta-learning/div_src/diversity_src/experiment_mains/main_diversity_with_task2vec.py --manual_loads_name diversity_ala_task2vec_hdb1_mio > $OUT_FILE 2> $ERR_FILE' > $PWD/main.sh.nohup.out$SLURM_JOBID &
```
todo: perhaps opening a `krbtmux` and running commands there e.g. `python main.sh &` would work? (perhaps we don't even need nohup)

Using gpus snap: https://ilwiki.stanford.edu/doku.php?id=hints:gpu
```bash
source cuda11.1
# To see Cuda version in use
nvcc -V
```

Check disk space/quota:
```bash
df -h /dfs/scratch0
```

TODO, to reuse code, have something that checks
the name of the cluster (if it's one of SNAPS, i.e. put the prefixes of snap ampere etc
and run the code bellow, else do nothing let slurm, condor do it):
```python
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# likely as above, set HOME on the script you are running to log correctly
```

Creating folders in scratch with lots of space.
```bash
cd /dfs/scratch0/ 
mkdir brando9
cd /dfs/scratch0/brando9
```

check quota in afs
```bash 
(metalearning_gpu) brando9~ $ fs lq $AFS
Volume Name                    Quota       Used %Used   Partition
user.brando9                 5000000    1963301   39%          6% 
```

weird bug activate doesn't work
```bash
export PATH="$HOME/miniconda/bin:$PATH"
echo $PATH
source ~/miniconda/bin/activate
conda activate metalearning_gpu
conda

#mv activate.c~ activate.c
#mv deactivate.c~ deactivate.c
#mv conda.c~ conda.c
#
#gcc -Wall activate.c -o activate
#gcc -Wall deactivate.c -o deactivate
#gcc -Wall conda.c -o conda

sh activate.c
sh deactivate.c
python conda.c
```

## CPU

ref: https://ilwiki.stanford.edu/doku.php?id=snap-servers:snap-servers#compute_servers