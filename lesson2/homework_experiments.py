"""Сравнение гиперпараметров и групп производных признаков на CPU."""

import csv
from itertools import combinations

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

if __package__:
    from .homework_datasets import QUALITY, load_wine_splits
    from .homework_training import (
        PLOTS, SEED, configure_run, evaluate_loss, fit_model, plt,
        save_figure, save_model,
    )
else:
    from homework_datasets import QUALITY, load_wine_splits
    from homework_training import (
        PLOTS, SEED, configure_run, evaluate_loss, fit_model, plt,
        save_figure, save_model,
    )

# В каждом опыте меняем один параметр относительно первого запуска.
CONFIGURATIONS = (
    (torch.optim.SGD, 0.025, 32),
    (torch.optim.SGD, 0.01, 32),
    (torch.optim.SGD, 0.025, 64),
    (torch.optim.Adam, 0.025, 32),
    (torch.optim.RMSprop, 0.025, 32),
)


def write_results(filename, rows):
    """Сохранить измерения с заголовками, пригодными для повторного анализа."""
    PLOTS.mkdir(exist_ok=True)
    with (PLOTS / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_hyperparameters(rows):
    """Показать ошибку каждого запуска на одной шкале."""
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = [f"{row['optimizer']} · lr {row['learning_rate']} · батч {row['batch_size']}" for row in rows]
    bars = axis.barh(labels, [row["validation_rmse"] for row in rows])
    axis.bar_label(bars, fmt="%.4f", padding=4)
    axis.margins(x=0.18)
    axis.invert_yaxis()
    axis.set_xlabel("RMSE на валидации")
    save_figure(figure, "hyperparameters.png")


def run_hyperparameter_search():
    """Выбрать конфигурацию регрессии по валидации и проверить её на тестовых винах."""
    training, validation, test = load_wine_splits("regression")
    criterion = nn.MSELoss()
    rows = []
    best_model, best_row = None, None
    for optimizer_class, rate, batch in CONFIGURATIONS:
        name = optimizer_class.__name__
        torch.manual_seed(SEED)
        model = nn.Linear(training.features.shape[1], 1)
        optimizer = optimizer_class(model.parameters(), lr=rate)
        losses, best_epoch = fit_model(model, training, validation, criterion, optimizer, epochs=40,
                                      batch_size=batch, label=f"{name}/{rate}/{batch}")
        row = {"optimizer": name, "learning_rate": rate, "batch_size": batch,
               "best_epoch": best_epoch, "validation_rmse": min(losses)**0.5}
        rows.append(row)
        print(f"{name:7} lr={rate:5g} batch={batch:2d} best_epoch={best_epoch:2d} "
              f"val RMSE={row['validation_rmse']:.3f}")
        if best_row is None or row["validation_rmse"] < best_row["validation_rmse"]:
            best_row, best_model = row, model

    test_rmse = evaluate_loss(best_model, test, criterion)**0.5
    print(f"Best by validation: {best_row}; test RMSE={test_rmse:.3f}")
    save_model(best_model, "wine_tuned.pt", task="regression", target=QUALITY,
               preprocessing=training.preprocessor.columns,
               feature_names=training.preprocessor.feature_names,
               selection=best_row, test_rmse=test_rmse)
    write_results("hyperparameters.csv", rows)
    plot_hyperparameters(rows)


def make_features(features, *, polynomial=False, interactions=False, statistics=False):
    """Дополнить исходные столбцы квадратами, парными произведениями или статистиками строк."""
    if features.ndim != 2 or features.shape[1] < 1 or not features.is_floating_point():
        raise ValueError("Признаки должны быть вещественным тензором формы (объекты, признаки)")
    parts = [features]
    if polynomial:
        parts.append(features.square())
    if interactions:
        parts.extend((features[:, i] * features[:, j]).unsqueeze(1)
                     for i, j in combinations(range(features.shape[1]), 2))
    if statistics:
        parts.extend((features.mean(dim=1, keepdim=True), features.var(dim=1, unbiased=False, keepdim=True)))
    return torch.cat(parts, dim=1)


def plot_feature_comparison(rows):
    """Построить столбчатый график RMSE всех заранее заданных вариантов."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(rows))
    axis.bar(positions - 0.19, [row["validation_rmse"] for row in rows], 0.38, label="Валидация")
    axis.bar(positions + 0.19, [row["test_rmse"] for row in rows], 0.38, label="Тест")
    axis.set(xticks=positions, xticklabels=[row["features"] for row in rows], ylabel="RMSE")
    axis.legend()
    save_figure(figure, "feature_engineering.png")


def run_feature_experiment():
    """Сравнить группы новых признаков на тех же выборках CSV красного вина."""
    datasets = load_wine_splits("regression")
    training = datasets[0]
    variants = {"Исходные": {}, "Квадраты": {"polynomial": True}, "Взаимодействия": {"interactions": True},
                "Статистики": {"statistics": True},
                "Все": {"polynomial": True, "interactions": True, "statistics": True}}
    criterion = nn.MSELoss()
    rows = []
    best_row, best_model, best_preprocessing = None, None, None
    for name, flags in variants.items():
        expanded = [make_features(dataset.features, **flags) for dataset in datasets]
        mean = expanded[0].mean(dim=0)
        scale = expanded[0].std(dim=0, unbiased=False)
        scale = torch.where(scale > 0, scale, 1.0)
        train_set, val_set, test_set = [TensorDataset((features - mean) / scale, dataset.targets)
                                       for features, dataset in zip(expanded, datasets)]
        torch.manual_seed(SEED)
        model = nn.Linear(len(mean), 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
        losses, best_epoch = fit_model(model, train_set, val_set, criterion, optimizer, epochs=100,
                                      patience=15, label=name)
        row = {"features": name, "size": len(mean), "best_epoch": best_epoch,
               "validation_rmse": min(losses)**0.5, "test_rmse": evaluate_loss(model, test_set, criterion)**0.5}
        rows.append(row)
        print(f"{name:15} n={row['size']:2d} val RMSE={row['validation_rmse']:.3f} test RMSE={row['test_rmse']:.3f}")
        if best_row is None or row["validation_rmse"] < best_row["validation_rmse"]:
            best_row, best_model = row, model
            best_preprocessing = {"feature_options": flags, "mean": mean.tolist(), "scale": scale.tolist()}

    save_model(best_model, "wine_features.pt", task="regression", target=QUALITY, selection=best_row,
               preprocessing=training.preprocessor.columns, feature_names=training.preprocessor.feature_names,
               **best_preprocessing)
    write_results("feature_engineering.csv", rows)
    plot_feature_comparison(rows)


def main():
    """Запустить перебор гиперпараметров и сравнение производных признаков."""
    configure_run()
    run_hyperparameter_search()
    run_feature_experiment()


if __name__ == "__main__":
    main()
