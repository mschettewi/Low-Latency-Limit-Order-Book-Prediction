import torch
import numpy as np

def get_lr_scheduler(optimizer, config, steps_per_epoch):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(step):
        warmup_steps = config.warmup_epochs * steps_per_epoch
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, config.num_epochs * steps_per_epoch - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)