# %%
import wandb
from uutils import run_bash_command

run_bash_command('pip install wandb --upgrade')
run_bash_command('echo $WANDB_API_KEY')
run_bash_command('cat ~/.netrc')

wandb.init(project="proof-term-synthesis", entity="brando", name='run_name', group='expt_name')

print('success!\a')

#%%
'''
conda create -n myenv python=3.9
'''