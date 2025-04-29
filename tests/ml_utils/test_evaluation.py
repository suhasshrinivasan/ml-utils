# test_evaluate.py

from dataclasses import dataclass
from typing import Callable, Iterable, List

import pytest
import torch

from ml_utils.backend import Backend
from ml_utils.evaluation import EvalSpec, evaluate


def abs_diff_fn(model, batch, backend):
    inputs, targets = batch
    inputs = inputs.to(backend.device)
    targets = targets.to(backend.device)
    outputs = model(inputs)
    loss = torch.abs(outputs - targets)
    return loss.squeeze(-1)


def mse(model, batch, backend):
    inputs, targets = batch
    inputs = inputs.to(backend.device)
    targets = targets.to(backend.device)
    outputs = model(inputs)
    loss = ((outputs - targets) ** 2).squeeze(-1)
    return loss


def dummy_mean_agg(values):
    values = torch.stack(values, dim=0)
    return values.mean().item()


def dummy_sum_agg(values):
    values = torch.stack(values, dim=0)
    return values.sum().item()


def dummy_stdev_agg(values):
    values = torch.stack(values, dim=0)
    return values.std().item()


class TestEvaluateFunction:
    def _build_loader(self):
        torch.manual_seed(42)
        x = torch.randn(9, 2)
        y = torch.randn(9, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=3), x, y

    def test_single_spec_default_agg(self, dummy_model, dummy_backend):
        loader, x, y = self._build_loader()

        eval_specs = [
            EvalSpec(name="abs_diff", fn=abs_diff_fn),
        ]

        results = evaluate(
            model=dummy_model.to(dummy_backend.device),
            eval_loaders=[("loader", loader)],
            eval_specs=eval_specs,
            backend=dummy_backend,
        )

        outputs = dummy_model(x.to(dummy_backend.device))
        diffs = torch.abs(outputs - y.to(dummy_backend.device)).squeeze(-1)
        expected_mean = diffs.mean().item()

        key = "abs_diff.mean_agg/loader"
        assert key in results
        assert pytest.approx(results[key], rel=1e-5) == expected_mean

    def test_single_spec_single_agg(self, dummy_model, dummy_backend):
        loader, x, y = self._build_loader()

        eval_specs = [
            EvalSpec(name="abs_diff", fn=abs_diff_fn, agg_fns=dummy_mean_agg),
        ]

        results = evaluate(
            model=dummy_model.to(dummy_backend.device),
            eval_loaders=[("loader", loader)],
            eval_specs=eval_specs,
            backend=dummy_backend,
        )

        outputs = dummy_model(x.to(dummy_backend.device))
        diffs = torch.abs(outputs - y.to(dummy_backend.device)).squeeze(-1)
        expected_mean = diffs.mean().item()

        key = "abs_diff.dummy_mean_agg/loader"
        assert key in results
        assert pytest.approx(results[key], rel=1e-5) == expected_mean

    def test_single_spec_multiple_agg(self, dummy_model, dummy_backend):
        loader, x, y = self._build_loader()

        eval_specs = [
            EvalSpec(
                name="abs_diff",
                fn=abs_diff_fn,
                agg_fns=[dummy_mean_agg, dummy_sum_agg, dummy_stdev_agg],
            ),
        ]

        results = evaluate(
            model=dummy_model.to(dummy_backend.device),
            eval_loaders=[("loader", loader)],
            eval_specs=eval_specs,
            backend=dummy_backend,
        )

        outputs = dummy_model(x.to(dummy_backend.device))
        diffs = torch.abs(outputs - y.to(dummy_backend.device)).squeeze(-1)
        expected_mean = diffs.mean().item()
        expected_sum = diffs.sum().item()
        expected_std = diffs.std(unbiased=True).item()

        mean_key = "abs_diff.dummy_mean_agg/loader"
        assert mean_key in results
        assert pytest.approx(results[mean_key], rel=1e-5) == expected_mean

        sum_key = "abs_diff.dummy_sum_agg/loader"
        assert sum_key in results
        assert pytest.approx(results[sum_key], rel=1e-5) == expected_sum

        std_key = "abs_diff.dummy_stdev_agg/loader"
        assert std_key in results
        assert pytest.approx(results[std_key], rel=1e-5) == expected_std

    def test_multiple_specs_multiple_agg(self, dummy_model, dummy_backend):
        loader, x, y = self._build_loader()

        eval_specs = [
            EvalSpec(
                name="abs_diff",
                fn=abs_diff_fn,
                agg_fns=[dummy_mean_agg, dummy_sum_agg, dummy_stdev_agg],
            ),
            EvalSpec(
                name="mse",
                fn=mse,
                agg_fns=[dummy_mean_agg, dummy_sum_agg, dummy_stdev_agg],
            ),
        ]

        results = evaluate(
            model=dummy_model.to(dummy_backend.device),
            eval_loaders=[("loader", loader)],
            eval_specs=eval_specs,
            backend=dummy_backend,
        )

        outputs = dummy_model(x.to(dummy_backend.device))

        diffs = torch.abs(outputs - y.to(dummy_backend.device)).squeeze(-1)
        expected_mean = diffs.mean().item()
        expected_sum = diffs.sum().item()
        expected_std = diffs.std(unbiased=True).item()

        mses = ((outputs - y.to(dummy_backend.device)) ** 2).squeeze(-1)
        expected_mse_mean = mses.mean().item()
        expected_mse_sum = mses.sum().item()
        expected_mse_std = mses.std(unbiased=True).item()

        mean_key = "abs_diff.dummy_mean_agg/loader"
        assert mean_key in results
        assert pytest.approx(results[mean_key], rel=1e-5) == expected_mean

        sum_key = "abs_diff.dummy_sum_agg/loader"
        assert sum_key in results
        assert pytest.approx(results[sum_key], rel=1e-5) == expected_sum

        std_key = "abs_diff.dummy_stdev_agg/loader"
        assert std_key in results
        assert pytest.approx(results[std_key], rel=1e-5) == expected_std

        mse_mean_key = "mse.dummy_mean_agg/loader"
        assert mse_mean_key in results
        assert pytest.approx(results[mse_mean_key], rel=1e-5) == expected_mse_mean

        mse_sum_key = "mse.dummy_sum_agg/loader"
        assert mse_sum_key in results
        assert pytest.approx(results[mse_sum_key], rel=1e-5) == expected_mse_sum

        mse_std_key = "mse.dummy_stdev_agg/loader"
        assert mse_std_key in results
        assert pytest.approx(results[mse_std_key], rel=1e-5) == expected_mse_std

    def test_multiple_specs_multiple_agg_multiple_loaders(
        self, dummy_model, dummy_backend
    ):
        loader1, x1, y1 = self._build_loader()
        loader2, x2, y2 = self._build_loader()

        eval_specs = [
            EvalSpec(
                name="abs_diff",
                fn=abs_diff_fn,
                agg_fns=[dummy_mean_agg, dummy_sum_agg, dummy_stdev_agg],
            ),
            EvalSpec(
                name="mse",
                fn=mse,
                agg_fns=[dummy_mean_agg, dummy_sum_agg, dummy_stdev_agg],
            ),
        ]

        results = evaluate(
            model=dummy_model.to(dummy_backend.device),
            eval_loaders=[("loader1", loader1), ("loader2", loader2)],
            eval_specs=eval_specs,
            backend=dummy_backend,
        )

        for loader_name, x, y in [("loader1", x1, y1), ("loader2", x2, y2)]:
            outputs = dummy_model(x.to(dummy_backend.device))
            diffs = torch.abs(outputs - y.to(dummy_backend.device)).squeeze(-1)
            mses = ((outputs - y.to(dummy_backend.device)) ** 2).squeeze(-1)

            expected = {
                f"abs_diff.dummy_mean_agg/{loader_name}": diffs.mean().item(),
                f"abs_diff.dummy_sum_agg/{loader_name}": diffs.sum().item(),
                f"abs_diff.dummy_stdev_agg/{loader_name}": diffs.std(
                    unbiased=True
                ).item(),
                f"mse.dummy_mean_agg/{loader_name}": mses.mean().item(),
                f"mse.dummy_sum_agg/{loader_name}": mses.sum().item(),
                f"mse.dummy_stdev_agg/{loader_name}": mses.std(unbiased=True).item(),
            }

            for key, expected_value in expected.items():
                assert key in results
                assert pytest.approx(results[key], rel=1e-5) == expected_value
