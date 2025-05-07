from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from ml_utils.backend import Backend
from ml_utils.evaluation import EvalSpec, evaluate
from ml_utils.trainer import TrainState


class Callback(ABC):
    """
    Abstract base class for all callbacks.
    """

    def __init__(self):
        """
        Initialize the callback.
        """
        pass

    @abstractmethod
    def on_run_begin(self, model, state):
        """
        Called at the beginning of run.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Called at the end of each epoch.
        """
        pass

    @abstractmethod
    def on_run_end(self, logs: dict = None):
        """
        Called at the end of run.
        """
        pass


class LoggerCallback(Callback):
    """
    Callback to log run progress.
    """

    def __init__(self, log_fn: Callable):
        """
        Initialize the logger callback.

        Args:
            log_fn (Callable): Function to log run progress.
        """
        super().__init__()
        self.log_fn = log_fn

    def on_run_begin(self, model, state):
        """
        Called at the beginning of run.
        """
        self.log_fn("Run started.")

    def on_epoch_end(self, model, state):
        """
        Called at the end of each epoch.
        """
        self.log_fn(f"Epoch {state.epoch} ended.")
        self.log_fn(f"Logs:\n{state.logs}")

    def on_run_end(self, model, state):
        """
        Called at the end of run.
        """
        self.log_fn("Run ended.")
        self.log_fn(f"Final logs:\n{state.logs}")


class StdoutLoggerCallback(LoggerCallback):
    """
    Callback to log run progress to stdout.
    """

    def __init__(self):
        """
        Initialize the stdout logger callback.
        """
        super().__init__(log_fn=print)


class EvaluatorCallback(Callback):
    """
    Evaluate a model and update TrainState logs after each loader evaluation.
    """

    def __init__(
        self,
        backend,
        eval_loaders: List[Tuple[str, Iterable]],
        eval_specs: List[EvalSpec],
    ):
        super().__init__()
        self.backend = backend
        self.eval_specs = eval_specs
        self.eval_loaders = eval_loaders

    def _evaluate(self, model, state):
        results = evaluate(
            model=model,
            eval_loaders=self.eval_loaders,
            eval_specs=self.eval_specs,
            backend=self.backend,
        )
        state.logs.update(results)

    def on_run_begin(self, model, state):
        self._evaluate(model, state)

    def on_epoch_end(self, model, state):
        self._evaluate(model, state)

    def on_run_end(self, model, state):
        self._evaluate(model, state)


class EarlyStopperCallback(Callback):
    """
    Stop training if the monitored metric does not improve for a given number of epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        threshold: float = 0.0,
        metric: str = "loss.mean/val",
        higher_is_better: bool = False,
    ):
        """
        Initialize the early stopping callback.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            threshold (float): Minimum change to qualify as an improvement.
        """
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.best_train_state = None
        self.best_model_state = None
        self.higher_is_better = higher_is_better

    def on_run_begin(self, model, state):
        state.stop_training = False
        self.best_train_state = deepcopy(state)
        if (
            self.best_train_state.logs.get(self.metric) is None
        ):  # model evaluation not performed
            self.best_train_state.logs[self.metric] = self._worst_value
        else:
            self.best_model_state = model.state_dict()

    def on_epoch_end(self, model, state):
        current_value = state.logs[self.metric]
        if self._metric_has_improved(current_value):
            self.best_train_state = deepcopy(state)
            self.best_train_state = model.state_dict()
            state.stop_training = False
        else:
            self.patience -= 1
            if self.patience == 0:
                state.stop_training = True

    def on_run_end(self, _, state):
        if self.best_train_state[self.metric] != self._worst_value:
            state.logs["early_stopping"] = {
                f"best_{self.metric}": self.best_train_state.logs[self.metric],
                "best_model_state": self.best_model_state,
                "best_epoch": self.best_train_state.epoch,
                "best_total_steps": self.best_train_state.total_steps,
            }

    @property
    def _worst_value(self):
        val = float("-inf") if self.higher_is_better else float("inf")
        return val

    def _metric_has_improved(self, current_value):
        best_so_far = self.best_train_state.logs[self.metric]
        delta = current_value - best_so_far
        if self.higher_is_better:
            if delta > self.threshold:
                return True
        else:
            if delta < self.threshold:
                return True
        return False
