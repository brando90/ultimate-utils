"""

# - scheduling rate tips
Do "But the most commonly used method is when the validation loss does not improve for a few epochs." according to https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

"""
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_scheduler(opt: Optimizer) -> _LRScheduler:
    """
    e.g.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(10):
            train(...)
            val_loss = validate(...)
            # Note that step should be called after validate()
            scheduler.step(val_loss)
    """
    pass