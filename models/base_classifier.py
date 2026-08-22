"""
Module: base_classifier
Description: Abstract Base Class for spectroscopic classifiers.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSpectraClassifier(ABC):
    """Abstract parent class defining the interface for spectral classifiers."""

    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize the base classifier.

        :param model_name: Name of the model algorithm.
        """
        if not isinstance(model_name, str):
            raise TypeError("Model name must be a string.")
        
        self.model_name = model_name
        self.is_trained = False
        self.classes_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Abstract method to train the classifier."""
        pass

    @abstractmethod
    def predict_with_confidence(self, X: np.ndarray):
        """Abstract method to predict label and return confidence score."""
        pass

    def __str__(self) -> str:
        """String representation of the classifier object."""
        status = "Trained" if self.is_trained else "Untrained"
        return f"{self.model_name} Classifier [{status}]"
