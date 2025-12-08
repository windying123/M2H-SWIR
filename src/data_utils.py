"""Utility functions for loading and splitting data."""

from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED


def load_numpy_pair(x_path, y_path) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path)
    Y = np.load(y_path)
    return X, Y


def split_simulated_lut(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split LUT into train/val/test."""
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=(test_size + val_size), random_state=random_state, shuffle=True
    )
    rel_val_size = val_size / (test_size + val_size)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=(1 - rel_val_size),
        random_state=random_state, shuffle=True
    )
    return {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
    }


def train_val_test_indices_by_year(years: np.ndarray, leave_out_year: int):
    """Return indices for leave-one-year-out CV."""
    test_idx = np.where(years == leave_out_year)[0]
    train_idx = np.where(years != leave_out_year)[0]
    return {"train": train_idx, "test": test_idx}
