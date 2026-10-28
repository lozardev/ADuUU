import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _scale(train, test):
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X, X_test, mean, std = _scale(data.train_features, data.test_features)
    y = np.asarray(data.train_target, dtype=float).ravel()
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0
    history = []
    for _ in range(data.epochs):
        err = X @ w + b - y
        history.append(float(np.mean(err**2)))
        w -= data.learning_rate * (2 / n) * (X.T @ err)
        b -= data.learning_rate * 2 * err.mean()

    pred = X_test @ w + b

    return RegressionResult(
        weights=w, bias=float(b), loss_history=history, predictions=pred
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X, X_test, mean, std = _scale(data.train_features, data.test_features)
    y = np.asarray(data.train_target, dtype=float).ravel()
    n, d = X.shape
    eps = 1e-12

    w = np.zeros(d)
    b = 0.0
    history = []
    for _ in range(data.epochs):
        p = _sigmoid(X @ w + b)
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        history.append(float(loss))
        grad = p - y
        w -= data.learning_rate * (X.T @ grad) / n
        b -= data.learning_rate * grad.mean()

    proba = _sigmoid(X_test @ w + b)
    pred = (proba >= 0.5).astype(int)
    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=history,
        probabilities=proba,
        predictions=pred,
    )


def _metrics(y_true, proba, pred):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    y_test = np.asarray(data.test_target).ravel()
    y_train = np.asarray(data.train_target).ravel()

    own = train_logistic_classifier(data)
    own_m = _metrics(y_test, own.probabilities, own.predictions)

    X, X_test, _, _ = _scale(data.train_features, data.test_features)

    tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y_train)
    tree_m = _metrics(y_test, tree.predict_proba(X_test)[:, 1], tree.predict(X_test))

    knn = KNeighborsClassifier(n_neighbors=15).fit(X, y_train)
    knn_m = _metrics(y_test, knn.predict_proba(X_test)[:, 1], knn.predict(X_test))

    return BaselineComparisonResult(
        logistic_regression=own_m, decision_tree=tree_m, knn=knn_m
    )


def sklearn_logreg_metrics(data: ClassificationInput) -> ClassificationMetrics:
    X, X_test, _, _ = _scale(data.train_features, data.test_features)
    y_test = np.asarray(data.test_target).ravel()
    m = LogisticRegression(max_iter=1000).fit(X, np.asarray(data.train_target).ravel())
    return _metrics(y_test, m.predict_proba(X_test)[:, 1], m.predict(X_test))
