import wandb


def try_printing_wandb_url() -> str:
    """
    Try to print the wandb url and return it as a string if it succeeds.
    If it fails, return the error message as a string.

    todo: refactor wrt try_printing_wandb_url in logging code
    """
    try:
        # print(f'{wandb.run.dir=}')
        print(f'{wandb.run.get_url()=}')
        # print(_get_sweep_url_hardcoded())
        print(f'{wandb.get_sweep_url()=}')
        return str(wandb.run.get_url())
    except Exception as e:
        err_msg: str = f'Error from wandb url get {try_printing_wandb_url=}: {e=}'
        print(err_msg)
        import logging
        logging.warning(err_msg)
        return str(e)

def _print_sweep_url(sweep_config: dict,
                     sweep_id: str,
                     verbose: bool = False,
                     ) -> str:
    """
    Legacy use, wandb.get_sweep_url() is better https://stackoverflow.com/questions/75852199/how-do-i-print-the-wandb-sweep-url-in-python
    ref:
        - SO: https://stackoverflow.com/questions/75852199/how-do-i-print-the-wandb-sweep-url-in-python
        - wandb discuss: https://community.wandb.ai/t/how-do-i-print-the-wandb-sweep-url-in-python/4133
    """
    entity = sweep_config['entity']
    project = sweep_config['project']
    if verbose:
        print(f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    return f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"