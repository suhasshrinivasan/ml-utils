from abc import ABC, abstractmethod
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
