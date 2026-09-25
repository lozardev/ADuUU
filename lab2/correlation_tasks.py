"""Задачи второй части лабораторной: корреляционный анализ."""

from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")

    men_df = df[df["Gender"] == "Male"]
    women_df = df[df["Gender"] == "Female"]
    men_count = len(men_df)
    women_count = len(women_df)

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men_corr = {}
    women_corr = {}

    for e in features:
        men_corr[e] = float(men_df[e].corr(men_df["MRI_Count"]))
        women_corr[e] = float(women_df[e].corr(women_df["MRI_Count"]))

    best_feature = ""
    max_abs = -1

    for e in features:
        if abs(men_corr[e]) > max_abs_val:
            max_abs_val = abs(men_corr[e])
            best_feature = e

        if abs(women_corr[e]) > max_abs_val:
            max_abs_val = abs(women_corr[e])
            best_feature = e

    return BrainCorrelationSummary(
        men_count, women_count, women_corr, men_corr, best_feature
    )
