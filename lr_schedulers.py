from functools import partial

import transformers
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


def get_schedule_fn(scheduler, num_training_steps=None, warmup_steps=None, dim_embed=None):
    """
    Returns a callable scheduler_fn(optimizer).
    """
    if scheduler == "cosine-decay":
        scheduler_fn = partial(
            CosineAnnealingLR,
            T_max=num_training_steps,
            eta_min=1e-7,
        )
    elif scheduler == "transformer":
        scheduler_fn = partial(TransformerScheduler, dim_embed=dim_embed, warmup_steps=warmup_steps)
    elif scheduler == "cosine-with-warmup":
        scheduler_fn = partial(
            transformers.get_wsd_schedule,
            num_warmup_steps=warmup_steps,
            num_stable_steps=0,
            num_decay_steps=num_training_steps - warmup_steps,
            min_lr_ratio=0.01,
        )
    elif scheduler == "inverse-sqrt":
        scheduler_fn = partial(
            transformers.get_inverse_sqrt_schedule,
            num_warmup_steps=warmup_steps,
        )
    else:
        raise ValueError(f"Invalid schedule {scheduler} given.")

    return scheduler_fn


class TransformerScheduler(_LRScheduler):
    def __init__(self, optimizer, dim_embed, warmup_steps, last_epoch=-1, verbose=False):
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lr = self.dim_embed ** (-0.5) * min(
            self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5)
        )
        return [lr] * self.num_param_groups
