from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple

import torch

from ml_utils.backend import Backend


def mean_agg(values):
    values = torch.cat(values, dim=0)
    return values.mean().item()


@dataclass
class EvalSpec:
    name: str
    fn: Callable  # Function to compute the metric
    agg_fns: List[Callable] = field(default_factory=lambda: [mean_agg])

    def __post_init__(self):
        if isinstance(self.agg_fns, Callable):
            self.agg_fns = [
                self.agg_fns
            ]  # Automatically wrap single function into list


def evaluate(
    model: torch.nn.Module,
    eval_loaders: List[Tuple[str, Iterable]],
    eval_specs: List[EvalSpec],
    backend: Backend,
) -> dict:
    """
    Evaluate a model across multiple loaders and metrics.

    Args:
        model: The model to evaluate.
        eval_loaders: List of (name, dataloader) pairs.
        eval_specs: List of EvalSpecs.
        backend: backend.

    Returns:
        Dict of evaluation results.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for loader_name, eval_loader in eval_loaders:
            buffers = {spec.name: [] for spec in eval_specs}

            for batch in eval_loader:
                for spec in eval_specs:
                    value = spec.fn(model, batch, backend)
                    buffers[spec.name].append(value)

            for spec in eval_specs:
                for agg_fn in spec.agg_fns:
                    aggregated = agg_fn(buffers[spec.name])
                    agg_fn_name = getattr(agg_fn, "__name__", "agg")
                    key = f"{spec.name}.{agg_fn_name}/{loader_name}"
                    results[key] = aggregated

    return results
