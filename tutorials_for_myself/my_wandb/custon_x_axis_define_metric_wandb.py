"""
https://docs.wandb.ai/guides/track/log#customize-axes-and-summaries-with-define_metric


https://colab.research.google.com/drive/1uegSY1HRGlKfK-07Uuw-ZxPJsNA9BN_9#scrollTo=2Lkko6FUOXqS

https://colab.research.google.com/drive/1pLe4peekE5ixKqCTfjVQofKB6w3mdEDP#scrollTo=HfV3zKwXN2yd

"""

#
# %%capture
# !pip install wandb

import numpy as np
import wandb

wandb.init()

# define our custom x axis metric
wandb.define_metric("outer_step")
# define which metrics will be plotted against it
wandb.define_metric("outer_train_loss", step_metric="outer_step")
wandb.define_metric("inner_train_losses", step_metric="outer_step")


# outer loop
for i_o in range(10):
  # inner loop
  i_is = []
  for i_i in range(5):
    i_is.append(i_i/5)
  # log
  log_dict = {
      "outer_train_loss": i_o,
      "outer_step": i_o,
      "inner_train_losses": wandb.Histogram(i_is)
  }
  wandb.log(log_dict, commit=True)
print('Done')
# wandb.finish()

# !cat wandb/latest-run/files/wandb-summary.json