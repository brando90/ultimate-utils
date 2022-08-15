#%%
import wandb
from uutils import cat_file, run_bash_command

run_bash_command('pip install wandb --upgrade')
cat_file('~/.zshrc')
cat_file('~/.netrc')

wandb.init(project="proof-term-synthesis", entity="brando", name='run_name', group='expt_name')

print('success!\a')

