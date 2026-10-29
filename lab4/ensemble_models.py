from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def build_model(seed: int = 42) -> VotingClassifier:
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    forest = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, random_state=seed, n_jobs=-1
    )

    boosting = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    )

    return VotingClassifier(
        estimators=[("logreg", logreg), ("forest", forest), ("boosting", boosting)],
        voting="soft",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()

    model = build_model(data.random_seed)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting (LogisticRegression + RandomForest + GradientBoosting)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
