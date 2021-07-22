# Ultimate-utils

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

## Installing Ultimate-utils

I am a collection of useful code that Brando has collected through the years that has been useful accross his projects.

To use me get a copy of this repo (I recommend to fork it on github) clone it as usual in your workspace. 
Then install it in development mode (
google it if you don't know what it is or see the `modules_in_python.md`) with:

```
conda develop ~/ultimate-utils/ultimate-utils-project
```

or (pip command not tested)

```
pip install -e ~/ultimate-utils/ultimate-utils-project
```

depending on which package manager you are using.
I usually use conda.

## Contributing

Feel free to push code with pull request.
Please include at least 1 self-contained test (that works) before pushing.

### How modules are imported in a python project

Read the `modules_in_python.md` to have an idea of the above development/editable installation commands. 

## Executing tensorboard stuff from remote

- visualize the remote logs using pycharm and my code (TODO: have the download be automatic...perhaps not needed)

1. Download the code from the cluster using pycharm remote
2. Then copy paste the *remote path* (from pycharm, browse remote)
3. Using the copied path run `tbb path2log` e.g. `tbbb /home/miranda9/data/logs/logs_Mar06_11-15-02_jobid_0_pid_3657/tb`

to have `tbbb` work as the command add to your `.zshrc` (or `.bashrc`):
```
alias tbb='sh /Users/brando/ultimate-utils/run_tb.sh'
```

then the command `tbb path2log` should work.

ref: see files
- https://github.com/brando90/ultimate-utils/blob/master/run_tb.sh
- https://github.com/brando90/ultimate-utils/blob/master/ultimate-utils-proj-src/execute_tensorboard.py
