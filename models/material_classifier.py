"""
Module: material_classifier
Description: Machine Learning model classifier for API Raman Material Identification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from models.base_classifier import BaseSpectraClassifier
from utils.spectral_dataset import SpectralDataset


class MaterialClassifier(BaseSpectraClassifier):
    """
    Raman Material Classifier implementing Random Forest with confidence estimation.
    Inherits from BaseSpectraClassifier (OOP Inheritance).
    Uses SpectralDataset internally (OOP Composition).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the MaterialClassifier.

        :param n_estimators: Number of trees in the Random Forest.
        :param random_state: Random state seed.
        """
        super().__init__(model_name="RandomForest Material Classifier")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
            
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.dataset = None  # Composition placeholder

    def load_dataset(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """Loads data into an internal SpectralDataset instance (Composition)."""
        self.dataset = SpectralDataset(X, y, feature_names)

    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
        """
        Trains the Random Forest model on dataset.

        :param X: Feature matrix (optional if loaded via load_dataset).
        :param y: Target array.
        """
        if X is None or y is None:
            if self.dataset is None:
                raise RuntimeError("No data provided to fit. Call load_dataset or supply X and y.")
            X, y = self.dataset.X, self.dataset.y

        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")

        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        self.is_trained = True
        return self

    def predict_with_confidence(self, X: np.ndarray) -> list:
        """
        Predicts material class and computes prediction confidence.

        :param X: Feature matrix of spectra to predict.
        :return: List of tuples (predicted_compound, confidence_score).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict_with_confidence.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probabilities = self._model.predict_proba(X)
        predictions = self._model.predict(X)

        results = []
        # Part 2 Requirement: Special functions (zip, map, lambda)
        confidences = list(map(lambda prob: float(np.max(prob)), probabilities))
        
        for pred, conf in zip(predictions, confidences):
            results.append({"material": str(pred), "confidence": round(conf * 100, 2)})

        return results

    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates model performance metrics.

        :param X_test: Test features.
        :param y_test: Test ground truth labels.
        :return: Dictionary containing evaluation metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")

        preds = self._model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # Part 2 Requirement: Dictionary Comprehension
        class_accuracies = {
            cls: report[cls]["precision"] for cls in self.classes_ if cls in report
        }

        return {
            "accuracy": float(acc),
            "class_precision": class_accuracies,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    def __add__(self, other):
        """
        Operator Overload (__add__): Combines tree counts of two classifiers.
        """
        if not isinstance(other, MaterialClassifier):
            raise TypeError("Can only add two MaterialClassifier instances together.")
        new_estimators = self.n_estimators + other.n_estimators
        return MaterialClassifier(n_estimators=new_estimators, random_state=self.random_state)
