#%%
"""
I think artifacts are used per run or something but I will just create one run upload of the zip of the data and
figs I made. Then create a report that links to the artifact run.

refs:
https://docs.wandb.ai/guides/artifacts
https://community.wandb.ai/t/how-does-one-manually-upload-a-single-artifact/841

failed artifact push: https://wandb.ai/brando/meta-learning-neurips-workshop/runs/221fe0xw
"""
from argparse import Namespace
from pathlib import Path

import wandb

import uutils

import time

start = time.time()

# uutils.setup_args_for_experiment() does the following:
# if hasattr(args, 'log_to_wandb'):
#     if args.log_to_wandb:
#         # os.environ['WANDB_MODE'] = 'offline'
#         import wandb
#
#         print(f'{wandb=}')
#
#         # - set run name
#         run_name = None
#         # if in cluster use the cluster jobid
#         if hasattr(args, 'jobid'):
#             # if jobid is actually set to something, use that as the run name in ui
#             if args.jobid is not None and args.jobid != -1 and str(args.jobid) != '-1':
#                 run_name: str = f'jobid={str(args.jobid)}'
#         # if user gives run_name overwrite that always
#         if hasattr(args, 'run_name'):
#             run_name = args.run_name if args.run_name is not None else run_name
#         args.run_name = run_name
#         # - initialize wandb
#         wandb.init(project=args.wandb_project,
#                    entity=args.wandb_entity,
#                    # job_type="job_type",
#                    name=run_name,
#                    group=args.experiment_name
#                    )
#         wandb.config.update(args)

def get_args_for_experiment() -> Namespace:
    # - get my default args
    args = uutils.parse_basic_meta_learning_args()
    args.log_to_wandb = False
    args.log_to_wandb = True
    args.wandb_project = 'meta-learning-neurips-workshop'
    args.experiment_name = 'upload-of-zip-files-synthetic-data-set-all-figs-and-hps'
    args.run_name = f'{args.experiment_name}_1'
    args = uutils.setup_args_for_experiment(args)
    return args

def get_zips_paths():
    log_root: Path = Path('~/Desktop/').expanduser()
    dataset_filename: str = 'dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15.zip'
    all_figs_zip_filename: str = 'all_ckpts_and_figures.zip'
    return log_root, dataset_filename, all_figs_zip_filename

# wandb.init(job_type="dataset-creation")  # done in uutils.setup_args_for_experiment()
print('-- getting args')
args = get_args_for_experiment()

# https://docs.wandb.ai/ref/python/artifact for Artifact api
print('-- creating artifacts')
artifact_data = wandb.Artifact('dataset_LS_fully_connected_NN_with_BN_nb_tasks200_data_per_task1000_l_4_nb_h_layes3_out1_H15.zip', type='dataset-as-zip')
artifact_figs = wandb.Artifact('all_figs_for_paper', type='figs-as-zip')

# - get zip files to log as artifacts
print('-- getting path to zipz')
log_root, dataset_filename, all_figs_zip_filename = get_zips_paths()

# todo - Imagine more lines of text were added to this text file: (what does this mean?)
print('-- wandb artifact logging1 (artifact_data.add_file)')
# artifact.add_file('my-dataset.txt')
artifact_data.add_file(log_root / dataset_filename)
artifact_figs.add_file(log_root / all_figs_zip_filename)

# Log that artifact, and we identify the changed file
print('-- wandb artifact logging2 (wandb.log_artifact)')
wandb.log_artifact(artifact_data)
wandb.log_artifact(artifact_figs)
# todo - Now you have a new version of the artifact, tracked in W&B (don't get it)

# - wandb
if args.log_to_wandb:
    print('-- finishing wandb')
    wandb.finish()
    print('-- wandb finished')

print(f'time_passed_msg = {uutils.report_times(start)}')
print('Done!\a')