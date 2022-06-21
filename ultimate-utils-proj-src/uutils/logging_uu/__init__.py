from typing import Optional

from pathlib import Path

import logging

# https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
FORMATTER = '--> [%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s <--'


def config_logging(out_file: Optional[str] = '~/.uu_default_out.log'):
    """
    Configures python's logging module to print to stdout and logging file.

    ref:
        - https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
    """
    handlers = [logging.StreamHandler()]
    if out_file == '' or out_file is None:
        handlers += logging.FileHandler(Path('.uu_default_out.log').expanduser()),

    logging.basicConfig(
        format=FORMATTER,
        level=logging.INFO,
        # ensure logs to stdout todo: confirm that INFO level is the one to always print to both https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
        handlers=handlers
    )

# if uncomment sets the default logging as given by that func.
# config_logging()
