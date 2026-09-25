"""Задачи первой части лабораторной: Pandas и Titanic."""

from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)

    missing = df.isna().sum().to_dict()

    age_over_30 = int((df["Age"] > 30).sum())

    mean_age_pclass = df.groupby("Pclass")["Age"].mean().to_dict()

    survival_rate_pclass = df.groupby("Pclass")["Survived"].mean().to_dict()

    top5_fares = df["Fare"].sort_values(ascending=False).head(5).tolist()

    return TitanicSummary(
        missing, age_over_30, mean_age_pclass, survival_rate_pclass, top5_fares
    )
