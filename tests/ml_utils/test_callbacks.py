from unittest.mock import MagicMock

import pytest
import torch

from ml_utils.callbacks import EvaluatorCallback, LoggerCallback
from ml_utils.evaluation import EvalSpec, mean_agg
from ml_utils.trainer import TrainState


class TestLoggerCallback:

    log_fn = MagicMock()
    logger = LoggerCallback(log_fn=log_fn)

    def test_init_with_log_fn(self):
        assert self.logger.log_fn == self.log_fn

    def test_on_run_begin(self, dummy_model, dummy_run_begin_state):
        self.logger.on_run_begin(dummy_model, dummy_run_begin_state)
        self.logger.log_fn.assert_called_with("Run started.")

    def test_on_epoch_end(self, dummy_model, dummy_epoch_end_state):
        self.logger.on_epoch_end(dummy_model, dummy_epoch_end_state)
        calls = [call.args[0] for call in self.log_fn.call_args_list]
        assert any("Epoch 4 ended." in c for c in calls)
        assert any(
            "Logs:\n{'loss': 0.123, 'accuracy': 0.456, 'val_loss': 0.789, 'val_accuracy': 0.101}"
            in c
            for c in calls
        )

    def test_on_run_end(self, dummy_model, dummy_run_end_state):
        self.logger.on_run_end(dummy_model, dummy_run_end_state)
        calls = [call.args[0] for call in self.log_fn.call_args_list]
        assert any("Run ended." in c for c in calls)
        assert any(
            "Final logs:\n{'loss': 0.045, 'accuracy': 0.999, 'val_loss': 0.067, 'val_accuracy': 0.89}"
            in c
            for c in calls
        )


def dummy_metric_fn(model, batch, backend):
    """
    A tiny metric: returns a tensor full of ones so the mean is 1.0.
    Keeps the interface (model, batch, backend) that evaluate() expects.
    """
    x, _ = batch
    return torch.ones(x.shape[0], device=backend.device)


class TestEvaluatorCallback:

    def _build_loader(self):
        x = torch.randn(6, 2)
        y = torch.randn(6, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(dataset, batch_size=3)

    def test_constructor_stores_arguments(self, dummy_backend):
        loaders = [("val", self._build_loader())]
        specs = [EvalSpec(name="ones", fn=dummy_metric_fn)]
        cb = EvaluatorCallback(
            backend=dummy_backend,
            eval_loaders=loaders,
            eval_specs=specs,
        )

        assert cb.backend is dummy_backend
        assert cb.eval_loaders is loaders
        assert cb.eval_specs is specs  # catches wrong attribute name

    def test_on_run_begin_updates_logs(self, dummy_model, dummy_backend):
        loader = self._build_loader()
        spec = EvalSpec(name="ones", fn=dummy_metric_fn)
        cb = EvaluatorCallback(dummy_backend, [("loader", loader)], [spec])

        state = TrainState()
        cb.on_run_begin(dummy_model.to(dummy_backend.device), state)

        key = "ones.mean_agg/loader"
        assert key in state.logs
        assert isinstance(state.logs[key], float)

    def test_on_epoch_end_keeps_previous_logs(self, dummy_model, dummy_backend):
        loader = self._build_loader()
        spec = EvalSpec(name="ones", fn=dummy_metric_fn)
        cb = EvaluatorCallback(dummy_backend, [("loader", loader)], [spec])

        state = TrainState()
        state.logs = {"pre_existing": 42.0}
        cb.on_epoch_end(dummy_model.to(dummy_backend.device), state)

        assert "pre_existing" in state.logs  # untouched
        assert "ones.mean_agg/loader" in state.logs  # new metric added

    def test_on_run_end_writes_metrics(self, dummy_model, dummy_backend):
        loader = self._build_loader()
        spec = EvalSpec(name="ones", fn=dummy_metric_fn)
        cb = EvaluatorCallback(dummy_backend, [("loader", loader)], [spec])

        state = TrainState()
        cb.on_run_end(dummy_model.to(dummy_backend.device), state)

        assert "ones.mean_agg/loader" in state.logs

    def test_multiple_loaders_and_specs(self, dummy_model, dummy_backend):
        loader1 = self._build_loader()
        loader2 = self._build_loader()

        def squared_error(model, batch, backend):
            x, y = batch
            return ((model(x.to(backend.device)) - y.to(backend.device)) ** 2).squeeze(
                -1
            )

        specs = [
            EvalSpec(name="ones", fn=dummy_metric_fn, agg_fns=[mean_agg]),
            EvalSpec(name="mse", fn=squared_error, agg_fns=[mean_agg]),
        ]

        cb = EvaluatorCallback(
            backend=dummy_backend,
            eval_loaders=[("loader1", loader1), ("loader2", loader2)],
            eval_specs=specs,
        )

        state = TrainState()
        cb.on_run_begin(dummy_model.to(dummy_backend.device), state)

        expected_keys = {
            "ones.mean_agg/loader1",
            "ones.mean_agg/loader2",
            "mse.mean_agg/loader1",
            "mse.mean_agg/loader2",
        }
        assert expected_keys.issubset(state.logs.keys())

    def test_handles_empty_loader_list(self, dummy_model, dummy_backend):
        spec = EvalSpec(name="ones", fn=dummy_metric_fn)
        cb = EvaluatorCallback(dummy_backend, [], [spec])

        state = TrainState()
        cb.on_run_begin(dummy_model.to(dummy_backend.device), state)

        # No loaders => no metrics, but also no crash
        assert state.logs == {}
