"""
Pytest Test Suite for MaterialClassifier.
"""

import numpy as np
import pytest
from models.material_classifier import MaterialClassifier
from utils.spectral_dataset import SpectralDataset


@pytest.fixture
def dummy_data():
    """Generates synthetic spectral dataset for testing."""
    X = np.random.rand(20, 50)
    y = np.array(["Aspirin"] * 10 + ["Ibuprofen"] * 10)
    return X, y


def test_model_training_and_prediction(dummy_data):
    """Validates training and confidence scoring."""
    X, y = dummy_data
    clf = MaterialClassifier(n_estimators=10)
    clf.fit(X, y)

    assert clf.is_trained is True

    predictions = clf.predict_with_confidence(X[:2])
    assert len(predictions) == 2
    assert "material" in predictions[0]
    assert "confidence" in predictions[0]
    assert 0.0 <= predictions[0]["confidence"] <= 100.0


def test_untrained_exception_handling():
    """Validates exception handling on untrained model."""
    clf = MaterialClassifier()
    dummy_input = np.random.rand(1, 50)

    with pytest.raises(RuntimeError):
        clf.predict_with_confidence(dummy_input)


def test_operator_overloading():
    """Validates __add__ operator overloading."""
    clf1 = MaterialClassifier(n_estimators=20)
    clf2 = MaterialClassifier(n_estimators=30)
    combined = clf1 + clf2

    assert combined.n_estimators == 50
