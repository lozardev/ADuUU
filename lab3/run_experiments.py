import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    sklearn_logreg_metrics,
    train_linear_regression,
    train_logistic_classifier,
)

SEED = 42

# регрессия
df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = (df["Extracurricular Activities"] == "Yes").astype(
    int
)
X = df.drop(columns="Performance Index").to_numpy(float)
y = df["Performance Index"].to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
reg = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte))
test_mse = float(np.mean((reg.predictions - yte) ** 2))
print("reg: duplicates", df.duplicated().sum(), "nan", df.isna().sum().sum())
print(
    "MSE epochs 1,10,100,400,800:",
    [reg.loss_history[i - 1] for i in (1, 10, 100, 400, 800)],
)
print("test MSE", test_mse, "weights", reg.weights, "bias", reg.bias)
# аналитическое решение
A = np.c_[(Xtr - Xtr.mean(0)) / Xtr.std(0), np.ones(len(Xtr))]
sol = np.linalg.lstsq(A, ytr, rcond=None)[0]
At = np.c_[(Xte - Xtr.mean(0)) / Xtr.std(0), np.ones(len(Xte))]
print("lstsq test MSE", float(np.mean((At @ sol - yte) ** 2)))
print("baseline (mean) MSE", float(np.mean((ytr.mean() - yte) ** 2)))

# классификация
g = pd.read_csv("german.csv", sep=";")
Xg = g.drop(columns="Creditability").to_numpy(float)
yg = g["Creditability"].to_numpy(int)
print("german classes", np.bincount(yg), "nan", g.isna().sum().sum())
Xgtr, Xgte, ygtr, ygte = train_test_split(
    Xg, yg, test_size=0.2, random_state=SEED, stratify=yg
)
cdata = ClassificationInput(Xgtr, ygtr, Xgte, ygte)
res = train_logistic_classifier(cdata)
print(
    "BCE epochs 1,10,100,500,1000:",
    [res.loss_history[i - 1] for i in (1, 10, 100, 500, 1000)],
)
cmp = compare_classical_models(cdata)
skl = sklearn_logreg_metrics(cdata)
out = {
    "linear_regression": {"test_mse": test_mse},
    "classification": {
        "own_logistic": asdict(cmp.logistic_regression),
        "logistic_regression": asdict(skl),
        "decision_tree": asdict(cmp.decision_tree),
        "knn": asdict(cmp.knn),
    },
}
for k, v in out["classification"].items():
    print(k, {a: round(b, 4) for a, b in v.items()})
json.dump(out, open("experiment_results.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(reg.loss_history)
ax[0].set_yscale("log")
ax[0].set_title("MSE (train)")
ax[0].set_xlabel("эпоха")
ax[1].plot(res.loss_history)
ax[1].set_title("BCE (train)")
ax[1].set_xlabel("эпоха")
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=100)
