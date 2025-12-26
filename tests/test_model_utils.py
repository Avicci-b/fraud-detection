import numpy as np
from sklearn.dummy import DummyClassifier
from src.model_utils import evaluate_model

def test_evaluate_model_outputs():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=100)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert "auc_pr" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
