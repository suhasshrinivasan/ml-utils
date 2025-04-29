from dataclasses import dataclass

import torch


@dataclass
class Backend:
    device: torch.device
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
