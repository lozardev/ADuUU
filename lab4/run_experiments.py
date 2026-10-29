import json

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_models import build_model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

SEED = 42

df = pd.read_csv("german.csv", sep=";")
X = df.drop(columns="Creditability").to_numpy(float)
y = df["Creditability"].to_numpy(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print("train/test:", len(y_train), len(y_test), "доля класса 1:", y.mean())


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def forest_tuned():
    return RandomForestClassifier(500, min_samples_leaf=3, random_state=SEED, n_jobs=-1)


def boosting_tuned():
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=SEED,
    )


experiments = {
    "LogisticRegression (база из лабы 3)": logreg(),
    "RandomForest (по умолчанию, 100 деревьев)": RandomForestClassifier(
        random_state=SEED, n_jobs=-1
    ),
    "RandomForest (500, min_samples_leaf=3)": forest_tuned(),
    "ExtraTrees (500, min_samples_leaf=3)": ExtraTreesClassifier(
        500, min_samples_leaf=3, random_state=SEED, n_jobs=-1
    ),
    "AdaBoost (200, lr=0.5)": AdaBoostClassifier(
        n_estimators=200, learning_rate=0.5, random_state=SEED
    ),
    "GradientBoosting (по умолчанию)": GradientBoostingClassifier(random_state=SEED),
    "GradientBoosting (300, lr=0.03, depth=3, subsample=0.8)": boosting_tuned(),
    "HistGradientBoosting (по умолчанию)": HistGradientBoostingClassifier(
        random_state=SEED
    ),
    "MLP (32 нейрона, alpha=1.0)": make_pipeline(
        StandardScaler(),
        MLPClassifier((32,), alpha=1.0, max_iter=2000, random_state=SEED),
    ),
    "Stacking (RF + GB + LR -> LR)": StackingClassifier(
        [
            ("forest", forest_tuned()),
            ("boosting", boosting_tuned()),
            ("logreg", logreg()),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
    ),
    "Soft Voting (LR + RF + GB) — выбранная": build_model(SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, model in experiments.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append(
        {
            "model": name,
            "cv_roc_auc": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_roc_auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
    )
    print(
        f"{name:58s} CV={cv_scores.mean():.4f}  test AUC={rows[-1]['test_roc_auc']:.4f}"
    )

table = pd.DataFrame(rows).round(4)
table.to_csv("experiments_table.csv", index=False)

final = run_ensemble_experiment(EnsembleInput(X_train, y_train, X_test, y_test, SEED))
print("\nИтог:", final.model_name)
print(
    f"ROC-AUC={final.roc_auc:.4f} acc={final.accuracy:.4f} prec={final.precision:.4f} "
    f"rec={final.recall:.4f} f1={final.f1:.4f}"
)

stab = {"voting": [], "logreg": []}
for s in range(10):
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
    stab["voting"].append(
        roc_auc_score(d, build_model(SEED).fit(a, c).predict_proba(b)[:, 1])
    )
    stab["logreg"].append(roc_auc_score(d, logreg().fit(a, c).predict_proba(b)[:, 1]))
for k, v in stab.items():
    print(
        f"устойчивость {k}: mean={np.mean(v):.4f} min={np.min(v):.4f} max={np.max(v):.4f}"
    )

results = {
    "model_name": final.model_name,
    "split": {"test_size": 0.2, "random_state": SEED, "stratify": True},
    "selection_criterion": "mean 5-fold CV ROC-AUC on train",
    "test_metrics": {
        "roc_auc": final.roc_auc,
        "accuracy": final.accuracy,
        "precision": final.precision,
        "recall": final.recall,
        "f1": final.f1,
    },
    "cv_roc_auc": float(table.iloc[-1]["cv_roc_auc"]),
    "experiments": table.to_dict(orient="records"),
    "stability_10_splits": {
        k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        for k, v in stab.items()
    },
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
