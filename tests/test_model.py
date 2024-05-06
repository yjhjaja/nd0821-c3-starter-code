import pytest, sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, os.path.join(dir, ml))
from model import train_model, compute_model_metrics, inference

@pytest.fixture
def sample_data():
    X_train = np.random.rand(100, 5)
    X_test  = np.random.rand(100, 5)
    y_train = np.random.binomial(n = 1, p = .5, size = [100])
    return X_train, X_test, y_train

def test_train_model(sample_data):
    X_train, X_test, y_train = sample_data
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    assert all(np.unique(preds)==[0]) or all(np.unique(preds)==[1]) or all(np.unique(preds)==[0,1])

def test_compute_model_metrics():
    y     = np.random.binomial(n = 1, p = .5, size = [100])
    preds = np.random.binomial(n = 1, p = .5, size = [100])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    mi = min([precision, recall, fbeta])
    ma = max([precision, recall, fbeta])
    assert mi>=0 and ma<=1

def test_inference(sample_data):
    X_train, X_test, y_train = sample_data
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    preds = inference(gb, X_test)
    assert all(np.unique(preds)==[0,1])
