def exec_train(args: tuple, run: wandb.wandb_run.Run):
    """
    note:
        - decided against named obj to simplify code i.e. didn't know model_args, data_args, training_args, general_args
        how to have the code write the variables on it's own. Would Namespace(**tup) work? Dont want to do d['x'] = x manually.
        I don't think automatic nameing obj is possible in python: https://chat.openai.com/share/b1d58369-ce27-4ee3-a588-daf28137f774
        better reference maybe some day.
        - seperates logic of wandb setup from the actual training code a little bit for cleaner (to reason) code.
        - passes run var just in case it's needed.
    """
    pass  # actualy traing code here


def train(args: tuple):
    """
    Runs train but seperates the wandb setup from the actual training code.
    """
    # - init wanbd run
    run = wandb.init()
    print(f'{wandb.get_sweep_url()}=')
    # - exec run
    # args[3].run = run  # just in case the GeneralArguments has a pointer to run. Decided against this to avoid multiple pointers to the same object.
    exec_train(args)
    # - finish wandb
    run.finish()



def exec_run_from_cmd_args_or_sweep():
    """
    Simply execs a run either from a wand sweep file or from the command line arguments. Ignore the wandb sweep details
    if it confuses you.
    """
    # 1. parse all the arguments from the command line
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, GeneralArguments))
    _, _, _, general_args = parser.parse_args_into_dataclasses()  # default args is to parse sys.argv
    path2sweep_config: str = general_args.path2sweep_config
    # 2. if the wandb_config option is on, then overwrite run cmd line configuration in favor of the sweep_config.
    if path2sweep_config:  # None => False => not getting wandb_config
        # overwrite run configuration with the wandb_config configuration (get config and create new args)
        config_path = Path(general_args.path2sweep_config).expanduser()
        with open(config_path, 'r') as file:
            sweep_config = dict(yaml.safe_load(file))
        sweep_args: list[str] = [item for pair in [[f'--{k}', str(v)] for k, v in sweep_config.items()] for item in
                                 pair]
        model_args, data_args, training_args, general_args = parser.parse_args_into_dataclasses(args=sweep_args)
        args: tuple = (model_args, data_args, training_args, general_args)  # decided against named obj to simplify code
        # 3. execute run from sweep
        # Initialize the sweep in Python which create it on your project/eneity in wandb platform and get the sweep_id.
        sweep_id = wandb.sweep(sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
        # Finally, once the sweep_id is acquired, execute the sweep using the desired number of agents in python.
        train = lambda: train(args)  # pkg train with args i.e., when you call train() it will all train(args).
        wandb.agent(sweep_id, function=train, count=general_args.count)
        # print(f"Sweep URL: https://wandb.ai/{sweep_config['entity']}/{sweep_config['project']}/sweeps/{sweep_id}")
        wandb.get_sweep_url()
    else:
        # use the args from the command line
        parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, GeneralArguments))
        model_args, data_args, training_args, general_args = parser.parse_args_into_dataclasses()
        # 3. execute run
        args: tuple = (model_args, data_args, training_args, general_args)  # decided against named obj to simplify code
        train(args)


if __name__ == '__main__':
    import time

    start_time = time.time()
    exec_run_from_cmd_args_or_sweep()
    print(f"The main function executed in {time.time() - start_time} seconds.\a")