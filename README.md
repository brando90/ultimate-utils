# Ultimate-utils

Ulitmate-utils (or uutils) is collection of useful code that Brando has collected through the years that has been useful accross his projects.
Mainly for machine learning and programming languages tasks.

## Installing Ultimate-utils

## Standard pip install [Recommended]

If you are going to use a gpu the do this first before continuing 
(or check the offical website: https://pytorch.org/get-started/locally/):
```angular2html
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Otherwise, just doing the follwoing should work.
```
pip install ultimate-utils
```
If that worked, then you should be able to import is as follows:
```
import uutils
```

note the import statement is shorter than the library name (`ultimate-utils` vs `uutils`).

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

To test pytorch do:
```
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```
To test if pytorch works with gpu do (it should fail if no gpus are available):
```
python -c "import uutils; uutils.torch_uu.gpu_test()"
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

## Citation
If you use this implementation consider citing us:

```
@software{brando2021ultimateutils,
    author={Brando Miranda},
    title={The ultimate utils library for Machine Learning and Artificial Intelligence},
    url={https://github.com/brando90/ultimate-utils},
    year={2021}
}
```
