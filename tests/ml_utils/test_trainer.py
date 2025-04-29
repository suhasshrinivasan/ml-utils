from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ml_utils.trainer import Trainer, TrainResult, TrainState, default_train_step_fn


@pytest.fixture
def dummy_loss_fn():
    def loss_fn(model, batch):
        x, y = batch
        output = model(x)
        return torch.nn.MSELoss()(output, y).unsqueeze(0)  # keep it batched

    return loss_fn


class DummyCallback:
    def __init__(self):
        self.run_begin_called = False
        self.epoch_end_called = False
        self.run_end_called = False

    def on_run_begin(self, model, state):
        self.run_begin_called = True

    def on_epoch_end(self, model, state):
        self.epoch_end_called = True

    def on_run_end(self, model, state):
        self.run_end_called = True


class TestTrainer:
    def test_trainer_returns_correct_types(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        assert isinstance(result, TrainResult)
        assert isinstance(result.state, TrainState)

    def test_trainer_tracks_losses(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        assert len(result.train_losses) == 2
        assert all(isinstance(loss, float) for loss in result.train_losses)

    def test_trainer_updates_state(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        assert result.state.epoch == 1
        assert result.state.total_steps > 0
        assert result.state.batch_count == len(dummy_data_loader)
        assert not result.state.stop_training

    def test_trainer_updates_model_weights(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        initial_model = dummy_model.__class__().to(dummy_backend.device)
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        weights_changed = not torch.equal(
            dummy_model.linear.weight.data, initial_model.linear.weight.data
        )
        assert (
            weights_changed
        ), "Model weights did not change — training may have failed."

    def test_trainer_populates_logs(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        assert "loss.mean/train" in result.state.logs
        assert isinstance(result.state.logs["loss.mean/train"], float)

    def test_trainer_moves_model_to_device(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        model_device = next(dummy_model.parameters()).device
        assert (
            model_device.type == dummy_backend.device.type
        ), f"Model is not on device {dummy_backend.device}"

    def test_trainer_leaves_model_in_train_mode(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        trainer.train(dummy_model, dummy_data_loader, n_epochs=2)

        assert (
            dummy_model.training is True
        ), "Model should be left in train mode after training."

    def test_trainer_calls_on_run_begin_callbacks(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        callback = DummyCallback()
        trainer = Trainer(dummy_backend, dummy_loss_fn, callbacks=[callback])
        trainer.train(dummy_model, dummy_data_loader, n_epochs=1)

        assert callback.run_begin_called, "on_run_begin was not called"

    def test_trainer_calls_on_epoch_end_callbacks(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        callback = DummyCallback()
        trainer = Trainer(dummy_backend, dummy_loss_fn, callbacks=[callback])
        trainer.train(dummy_model, dummy_data_loader, n_epochs=1)

        assert callback.epoch_end_called, "on_epoch_end was not called"

    def test_trainer_calls_on_run_end_callbacks(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        callback = DummyCallback()
        trainer = Trainer(dummy_backend, dummy_loss_fn, callbacks=[callback])
        trainer.train(dummy_model, dummy_data_loader, n_epochs=1)

        assert callback.run_end_called, "on_run_end was not called"

    def test_trainer_stops_training_when_requested(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        class StopTrainingCallback:
            def on_run_begin(self, model, state):
                pass

            def on_epoch_end(self, model, state):
                state.stop_training = True

            def on_run_end(self, model, state):
                pass

        callback = StopTrainingCallback()
        trainer = Trainer(dummy_backend, dummy_loss_fn, callbacks=[callback])
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=5)

        assert result.state.epoch == 0, "Training should have stopped after first epoch"

    def test_trainer_tracks_batch_count_correctly(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=1)

        assert result.state.batch_count == len(
            dummy_data_loader
        ), "Batch count mismatch"

    def test_trainer_tracks_total_steps_correctly(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=3)

        expected_steps = len(dummy_data_loader) * 3
        assert result.state.total_steps == expected_steps, "Total steps mismatch"

    def test_trainer_handles_zero_epochs(
        self, dummy_model, dummy_data_loader, dummy_backend, dummy_loss_fn
    ):
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, dummy_data_loader, n_epochs=0)

        assert result.train_losses == []
        assert result.state.total_steps == 0
        assert result.state.epoch == 0
        assert result.state.batch_count == 0

    def test_trainer_handles_empty_dataloader(
        self, dummy_model, dummy_backend, dummy_loss_fn
    ):
        empty_loader = []
        trainer = Trainer(dummy_backend, dummy_loss_fn)
        result = trainer.train(dummy_model, empty_loader, n_epochs=1)

        assert len(result.train_losses) == 1
        assert (
            np.isnan(result.train_losses[0]) or result.train_losses[0] == 0.0
        ), "Loss should be NaN or 0 for empty loader"
        assert result.state.batch_count == 0
        assert result.state.total_steps == 0
        assert result.state.batch_count == 0
