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
docker run -v /Users/brandomiranda/ultimate-utils:/home/bot/ultimate-utils \
           -v /Users/brandomiranda/data:/home/bot/data \
           -ti brandojazz/ultimate-utils:test_arm bash
```

Inside the docker container:
```bash
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```