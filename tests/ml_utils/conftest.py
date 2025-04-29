import pytest
import torch

from ml_utils.backend import Backend
from ml_utils.trainer import TrainState

# === Test Constants ===
BATCH_SIZE = 3
INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_SAMPLES = 9
TOTAL_STEPS = 10
LEARNING_RATE = 0.01

# === Set random seed ===
SEED = 42
torch.manual_seed(SEED)


# === Fixtures ===
@pytest.fixture
def dummy_model():
    """
    Fixture to create a dummy model for testing.
    """

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.LazyLinear(OUTPUT_DIM)

        def forward(self, x):
            return self.linear(x)

    return DummyModel()


@pytest.fixture
def dummy_run_begin_state():
    state = TrainState()
    return state


@pytest.fixture
def dummy_epoch_end_state():
    state = TrainState()
    state.epoch = 4
    state.logs = {
        "loss": 0.123,
        "accuracy": 0.456,
        "val_loss": 0.789,
        "val_accuracy": 0.101,
    }
    state.stop_training = False
    state.batch_count = BATCH_SIZE
    state.total_steps = 43
    return state


@pytest.fixture
def dummy_run_end_state():
    state = TrainState()
    state.epoch = 100
    state.logs = {
        "loss": 0.045,
        "accuracy": 0.999,
        "val_loss": 0.067,
        "val_accuracy": 0.89,
    }
    state.stop_training = True
    state.batch_count = BATCH_SIZE
    state.total_steps = TOTAL_STEPS
    return state


@pytest.fixture
def dummy_data_loader():
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    y = torch.randn(BATCH_SIZE, OUTPUT_DIM)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)


@pytest.fixture
def dummy_backend():
    return Backend(
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.01),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
