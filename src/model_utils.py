from sklearn.metrics import average_precision_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model using AUC-PR and F1-score.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_pr": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics
