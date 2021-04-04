from typing import *

import torch
from fvcore.common.param_scheduler import *
from torch.optim import Optimizer


class LRMultiplier(torch.optim.lr_scheduler._LRScheduler):
    """
    A LRScheduler which uses fvcore  `ParamScheduler` to multiply the
    learning rate of each param in the optimizer.
    Every step, the learning rate of each parameter becomes its initial value
    multiplied by the output of the given `ParamScheduler`.
    The absolute learning rate value of each parameter can be different.
    This scheduler can be used as long as the relative scale among them do
    not change during training.

    Source: https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/lr_scheduler.py
    """

    # NOTES: in the most general case, every LR can use its own scheduler.
    # Supporting this requires interaction with the optimizer when its parameter
    # group is initialized. For example, classyvision implements its own optimizer
    # that allows different schedulers for every parameter group.
    # To avoid this complexity, we use this class to support the most common cases
    # where the relative scale among all LRs stay unchanged during training.  In this
    # case we only need a total of one scheduler that defines the relative LR multiplier.

    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: ParamScheduler,
        max_iter: int,
        last_iter: int = -1,
    ):
        """
        Args:
            optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            multiplier: a fvcore ParamScheduler that defines the multiplier on
                every LR of the optimizer
            max_iter: the total number of training iterations
        """
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(
                "_LRMultiplier(multiplier=) must be an instance of fvcore "
                f"ParamScheduler. Got {multiplier} instead."
            )
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        # fvcore schedulers are stateless. Only keep pytorch scheduler states
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]


# Cell
def FlatCosScheduler(optimizer: Optimizer, pct_start: float, max_iters: int):
    """
    Schedule the LearningRate at flat `lr` for `pct_start` of `max_iters` before cosine annealing.

    Inspired From - https://docs.fast.ai/callback.schedule.html#Learner.fit_flat_cos.
    """

    schedulers = [LinearParamScheduler(1, 1), CosineParamScheduler(1, 0)]

    sched = CompositeParamScheduler(
        schedulers,
        lengths=[pct_start, 1 - pct_start],
        interval_scaling=["rescaled", "rescaled"],
    )

    # Wrap Param Scheduler under LRMultiplier class
    sched = LRMultiplier(optimizer, sched, max_iter=max_iters)
    return sched