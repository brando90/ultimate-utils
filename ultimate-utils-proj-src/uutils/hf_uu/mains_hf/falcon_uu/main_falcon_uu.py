from argparse import Namespace


def main_falcon_qlora():
    """
python -m pdb -c continue ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to none --path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/sweep_configs_falcon_qlora/falcon_debug_config.yaml
python -m pdb -c continue ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to wandb --path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/sweep_configs_falcon_qlora/falcon_sweep_config.yaml
    """
    from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep, setup_wandb_for_train_with_hf_trainer
    from uutils.hf_uu.hf_argparse.common import get_simple_args

    # - get most basic hf args
    args: Namespace = get_simple_args()  # just report_to, path2sweep_config, path2debug_seep
    print(args)

    # - run train
    from uutils.hf_uu.train.sft.qlora_ft import train_falcon_qlora_ft  # <-- modify this for other model
    report_to = args.report_to
    if report_to == "none":
        train: callable = train_falcon_qlora_ft
        train(args)
    elif report_to == "wandb":
        path2sweep_config = args.path2sweep_config
        train = lambda: train_falcon_qlora_ft(args)
        exec_run_for_wandb_sweep(path2sweep_config, train)
    else:
        raise ValueError(f'Invalid hf report_to option: {report_to=}.')


def main_falcon_7b_fp32_28gb_gpu_mem_mdl():
    """
python -m pdb -c continue ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to none --path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/sweep_configs_falcon7b_fft/falcon_debug_config.yaml
python -m pdb -c continue ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to wandb --path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/sweep_configs_falcon7b_fft/falcon_sweep_config.yaml
    """
    from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep, setup_wandb_for_train_with_hf_trainer
    from uutils.hf_uu.hf_argparse.common import get_simple_args
    import socket
    print(f'{socket.gethostname()=}')

    # - get most basic hf args
    args: Namespace = get_simple_args()  # just report_to, path2sweep_config, path2debug_seep
    print(args)

    # - run train
    from uutils.hf_uu.train.full_fine_tuning import train_falcon_7b_32fp_28gb_mem  # <-- modify this 4 diff train
    report_to = args.report_to
    if report_to == "none":
        train: callable = train_falcon_7b_32fp_28gb_mem
        train(args)
    elif report_to == "wandb":
        path2sweep_config = args.path2sweep_config
        train = lambda: train_falcon_7b_32fp_28gb_mem(args)
        exec_run_for_wandb_sweep(path2sweep_config, train)
    else:
        raise ValueError(f'Invalid hf report_to option: {report_to=}.')

def main_falcon_7b_fp32_28gb_gpu_mem_mdl():
    """
python -m pdb -c continue ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to none --path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/sweep_configs_falcon7b_fft/falcon_debug_config.yaml
python -m pdb -c continue ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/main_falcon_uu.py --report_to wandb --path2config ~/ultimate-utils/ultimate-utils-proj-src/uutils/hf_uu/mains_hf/falcon_uu/sweep_configs_falcon7b_fft/falcon_sweep_config.yaml
    """
    from uutils.wandb_uu.sweeps_common import exec_run_for_wandb_sweep, setup_wandb_for_train_with_hf_trainer
    from uutils.hf_uu.hf_argparse.common import get_simple_args
    import socket
    print(f'{socket.gethostname()=}')

    # - get most basic hf args
    args: Namespace = get_simple_args()  # just report_to, path2sweep_config, path2debug_seep
    print(args)

    # - run train
    from uutils.hf_uu.train.full_fine_tuning import train_falcon_7b_32fp_28gb_mem  # <-- modify this 4 diff train
    report_to = args.report_to
    if report_to == "none":
        train: callable = train_falcon_7b_32fp_28gb_mem
        train(args)
    elif report_to == "wandb":
        path2sweep_config = args.path2sweep_config
        train = lambda: train_falcon_7b_32fp_28gb_mem(args)
        exec_run_for_wandb_sweep(path2sweep_config, train)
    else:
        raise ValueError(f'Invalid hf report_to option: {report_to=}.')


if __name__ == '__main__':
    import time

    start_time = time.time()
    main_falcon_7b_fp32_28gb_gpu_mem_mdl()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")
