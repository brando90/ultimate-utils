def report_to2wandb_init_mode(report_to: str) -> str:
    if report_to == 'none':
        return 'disabled'
    else:
        assert report_to == 'wandb', f'Err {report_to=}.'
        return 'online'
