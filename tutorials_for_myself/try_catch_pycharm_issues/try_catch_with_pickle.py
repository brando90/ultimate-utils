"""
trying to resolve:
- https://intellij-support.jetbrains.com/hc/en-us/requests/3764538


You will need to run
    pip install transformers
    pip install fairseq

On mac
    pip3 install torch torchvision torchaudio
on linux
    pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
on windows
    pip3 install torch torchvision torchaudio
"""
import argparse
from argparse import Namespace
from typing import Any

from fairseq import optim
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers.optimization import AdafactorSchedule


def invoke_handled_exception():
    try:
        1 / 0
    except ZeroDivisionError:
        print('exception caught')


def make_args_pickable(args: Namespace) -> Namespace:
    """
    Returns a copy of the args namespace but with unpickable objects as strings.

    note: implementation not tested against deep copying.
    ref:
        - https://stackoverflow.com/questions/70128335/what-is-the-proper-way-to-make-an-object-with-unpickable-fields-pickable
        - pycharm halting all the time issues: https://stackoverflow.com/questions/70761481/how-to-stop-pycharms-break-stop-halt-feature-on-handled-exceptions-i-e-only-b
        - stop progressbar from printing progress when checking if it's pickable: https://stackoverflow.com/questions/70762899/how-does-one-stop-progressbar-from-printing-eta-progress-when-checking-if-the
    """
    pickable_args = argparse.Namespace()
    # - go through fields in args, if they are not pickable make it a string else leave as it
    # The vars() function returns the __dict__ attribute of the given object.
    for field in vars(args):
        # print(f'-----{field}')
        field_val: Any = getattr(args, field)
        if not is_picklable(field_val):
            field_val: str = str(field_val)
        # - after this line the invariant is that it should be pickable, so set it in the new args obj
        setattr(pickable_args, field, field_val)
        # print('f-----')
    return pickable_args

def is_picklable(obj: Any) -> bool:
    """
    Checks if somehting is pickable.

    Ref:
        - https://stackoverflow.com/questions/70128335/what-is-the-proper-way-to-make-an-object-with-unpickable-fields-pickable
        - pycharm halting all the time issue: https://stackoverflow.com/questions/70761481/how-to-stop-pycharms-break-stop-halt-feature-on-handled-exceptions-i-e-only-b
    """
    import pickle
    try:
        pickle.dumps(obj)
    except pickle.PicklingError:
        return False
    return True

def invoke_handled_exception_brandos_pickle_version():
    mdl: nn.Module = nn.Linear(4, 3)
    optimizer: Optimizer = optim.adafactor.Adafactor(params=mdl.parameters())
    scheduler: _LRScheduler = AdafactorSchedule(optimizer)
    args: Namespace = Namespace(scheduler=scheduler, optimizer=optimizer, model=mdl)
    make_args_pickable(args)
    print('Success if this line printed! Args was made into a pickable args without error')

# -- tests
invoke_handled_exception()
invoke_handled_exception_brandos_pickle_version()