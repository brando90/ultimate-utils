from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR

# class AdafactorSchedule(LambdaLR):
#     """
#     Since :class:`~transformers.optimization.Adafactor` performs its own scheduling, if the training loop relies on a
#     scheduler (e.g., for logging), this class creates a proxy object that retrieves the current lr values from the
#     optimizer.
#     It returns ``initial_lr`` during startup and the actual ``lr`` during stepping.
#     """
#
#     def __init__(self, optimizer, initial_lr=0.0):
#         def lr_lambda(_):
#             return initial_lr
#
#         for group in optimizer.param_groups:
#             group["initial_lr"] = initial_lr
#         super().__init__(optimizer, lr_lambda)
#         for group in optimizer.param_groups:
#             del group["initial_lr"]
#
#     def get_lr(self):
#         opt = self.optimizer
#         lrs = [
#             opt._get_lr(group, opt.state[group["params"][0]])
#             for group in opt.param_groups
#             if group["params"][0].grad is not None
#         ]
#         if len(lrs) == 0:
#             lrs = self.base_lrs  # if called before stepping
#         return lrs

def get_hugging_face_adafactor_scheduler(optimizer, initial_lr: float = 0.0) -> _LRScheduler:
    from transformers.optimization import AdafactorSchedule
    # scheduler = AdafactorSchedule(optimizer)
    scheduler = AdafactorSchedule(optimizer, initial_lr=initial_lr)
    return scheduler


class _AdafactorSchedulerUU(LambdaLR):
    """
    TODO: I think this should work for the torch_optimizer library...
        - perhaps its better to return the hugging face one? i.e. scheduler = AdafactorSchedule(optimizer)
        - https://github.com/jettify/pytorch-optimizer/issues/404
    """

    def __init__(self, optimizer, initial_lr=0.0):
        assert False, 'untested'
        def lr_lambda(_):
            return initial_lr

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr

        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs