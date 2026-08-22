"""
Module: spectral_dataset
Description: Custom dataset class managing spectral data and labels.
"""

import numpy as np
import pandas as pd


class SpectralDataset:
    """Class representing a dataset of Raman spectra for composition/aggregation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Initialize the SpectralDataset.

        :param X: Feature matrix of shape (n_samples, n_features).
        :param y: Target array of shape (n_samples,).
        :param feature_names: List of Raman shift frequencies.
        """
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} labels.")
        
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=str)
        self.feature_names = feature_names if feature_names is not None else list(range(X.shape[1]))

    def __len__(self) -> int:
        """Operator Overload: returns total number of spectral samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Allows indexing into dataset."""
        return self.X[idx], self.y[idx]

    def sample_generator(self, batch_size: int = 16):
        """
        Part 2 Requirement: Generator function yielding data batches.

        :param batch_size: Size of each data batch.
        """
        n_samples = len(self)
        for i in range(0, n_samples, batch_size):
            yield self.X[i:i + batch_size], self.y[i:i + batch_size]
