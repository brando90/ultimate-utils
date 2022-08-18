"""
ref: https://github.com/wandb/wandb/issues/1745
"""
# %%

import wandb

api: wandb.Api = wandb.Api()
run = api.run("brando/proj.../run_name...")
run.delete()