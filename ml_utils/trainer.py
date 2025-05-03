from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import torch

from ml_utils.backend import Backend


@dataclass
class StepResult:
    loss: float
    extra: dict = field(default_factory=dict)


def default_train_step_fn(model, batch, backend, loss_fn) -> StepResult:
    batch = [x.to(backend.device) for x in batch]
    backend.optimizer.zero_grad()
    loss = loss_fn(model, batch).mean()
    loss.backward()
    backend.optimizer.step()
    return StepResult(loss=loss.item())


@dataclass
class TrainState:
    epoch: int = 0
    logs: dict = field(default_factory=dict)
    stop_training: bool = False
    batch_count: int = 0
    total_steps: int = 0


@dataclass
class TrainResult:
    train_losses: List
    state: TrainState


class Trainer:
    def __init__(
        self,
        backend: Backend,
        loss_fn: Callable,
        train_step_fn: Callable = default_train_step_fn,
        callbacks: List[Callable] = None,
    ):
        self.backend = backend
        self.loss_fn = loss_fn
        self.train_step_fn = train_step_fn
        self.callbacks = callbacks or []
        self.state = TrainState()

    def train(self, model, train_loader, n_epochs):
        train_losses = []

        model = model.to(self.backend.device)

        for cb in self.callbacks:
            cb.on_run_begin(model, self.state)

        for epoch in range(n_epochs):
            if self.state.stop_training:
                break

            self.state.epoch = epoch
            self.state.batch_count = 0

            train_loss = self._train_epoch(model, train_loader)
            train_losses.append(train_loss)

            self.state.logs = {"loss.mean/train": train_loss}
            for cb in self.callbacks:
                cb.on_epoch_end(model, self.state)

        for cb in self.callbacks:
            cb.on_run_end(model, self.state)

        return TrainResult(train_losses=train_losses, state=self.state)

    def _train_epoch(self, model, train_loader):
        model.train()
        losses = []
        for batch in train_loader:
            step_result = self.train_step_fn(model, batch, self.backend, self.loss_fn)
            losses.append(step_result.loss)
            self.state.batch_count += 1
            self.state.total_steps += 1
        return np.mean(losses)
