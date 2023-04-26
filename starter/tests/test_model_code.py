import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from starter.ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope="session")
def data():
    X = np.ones((2, 14), dtype=np.float32)
    X[0, :] = 0.0
    y = np.array([0, 1], dtype=np.float32)
    return X, y

@pytest.fixture(scope="session")
def model(data):
    X, y = data
    return train_model(X, y)

@pytest.fixture(scope="session")
def pred(data, model):
    X, _ = data
    return inference(model, X)

def test_train_model(model):
    assert isinstance(model, RandomForestClassifier)

def test_inference(data, pred):
    _, y = data
    assert pred.shape == y.shape

def test_compute_model_metrics(data, pred):
    _, y = data
    precision, recall, fbeta = compute_model_metrics(y, pred)
    assert all(isinstance(n, float) for n in [precision, recall, fbeta])
    assert all((n >= 0 and n <= 1) for n in [precision, recall, fbeta])
