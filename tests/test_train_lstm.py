"""Regression tests for time-ordered LSTM data splitting."""

import torch
from torch.utils.data import TensorDataset

from train_lstm import sequential_train_val_split


def test_sequential_train_val_split_preserves_time_order() -> None:
    """Train samples should be the prefix and validation samples the suffix."""
    dataset = TensorDataset(
        torch.arange(10, dtype=torch.float32).view(10, 1),
        torch.arange(10, dtype=torch.float32).view(10, 1),
    )

    train_dataset, val_dataset = sequential_train_val_split(dataset, 0.8)

    assert train_dataset.indices == list(range(8))
    assert val_dataset.indices == [8, 9]
