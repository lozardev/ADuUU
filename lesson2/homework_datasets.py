"""CSV-датасет PyTorch, регрессия и бинарная классификация качества вина."""

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import Dataset

if __package__:
    from .homework_training import (
        LESSON_DIR, SEED, classification_metrics, configure_run,
        fit_model, plot_confusion_matrix, save_model, split_indices,
    )
else:
    from homework_training import (
        LESSON_DIR, SEED, classification_metrics, configure_run,
        fit_model, plot_confusion_matrix, save_model, split_indices,
    )

WINE_CSV = LESSON_DIR / "models" / "winequality-red.csv"
QUALITY = "quality"
BINARY_THRESHOLD = 6
BINARY_LABELS = ("0–5", "6–10")
BINARY_VALUES = {"0": 0.0, "1": 1.0, "false": 0.0, "true": 1.0, "no": 0.0, "yes": 1.0}


def read_csv(path, target, delimiter=","):
    """Прочитать CSV с заголовком и проверить форму строк и наличие цели."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter, strict=True)
        headers = reader.fieldnames or []
        if len(headers) < 2 or len(headers) != len(set(headers)) or any(not name.strip() for name in headers):
            raise ValueError("Нужны уникальные непустые заголовки признаков и цели")
        if target not in headers:
            raise ValueError(f"В CSV нет целевого столбца {target!r}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Строка {reader.line_num}: число значений отличается от заголовка")
            clean = {name: value.strip() for name, value in row.items()}
            if not clean[target]:
                raise ValueError(f"Строка {reader.line_num}: отсутствует целевое значение")
            rows.append(clean)
    if not rows:
        raise ValueError("CSV не содержит объектов")
    return [name for name in headers if name != target], rows


def finite_number(value):
    """Преобразовать текст CSV в конечное число или вернуть None."""
    try:
        number = float(value)
    except ValueError:
        return None
    return number if np.isfinite(number) else None


class CSVPreprocessor:
    """Запомнить статистики числовых и словари категориальных столбцов обучения."""

    def __init__(self, headers, training_rows):
        self.columns = {}
        self.feature_names = []
        for name in headers:
            values = [row[name] for row in training_rows]
            observed = [value for value in values if value]
            numbers = [finite_number(value) for value in observed]
            if observed and all(value.casefold() in BINARY_VALUES for value in observed):
                binary = [BINARY_VALUES[value.casefold()] for value in observed]
                self.columns[name] = {"kind": "binary", "fill": int(sum(binary) / len(binary) >= 0.5)}
                self.feature_names.append(name)
            elif observed and all(number is not None for number in numbers):
                self.columns[name] = {"kind": "numeric", "mean": float(np.mean(numbers)),
                                      "scale": float(np.std(numbers)) or 1.0}
                self.feature_names.append(name)
            else:
                levels = sorted(set(values) | {""})
                self.columns[name] = {"kind": "category", "levels": levels}
                self.feature_names.extend(f"{name}={level!r}" for level in levels)

    def encode(self, row):
        """Преобразовать строку по обученной схеме, кодируя новые категории нулями."""
        features = []
        for name, column in self.columns.items():
            value = row[name]
            if column["kind"] == "numeric":
                number = finite_number(value) if value else column["mean"]
                if number is None:
                    raise ValueError(f"Столбец {name!r}: ожидалось число, получено {value!r}")
                features.append((number - column["mean"]) / column["scale"])
            elif column["kind"] == "binary":
                if value and value.casefold() not in BINARY_VALUES:
                    raise ValueError(f"Столбец {name!r}: ожидалось бинарное значение, получено {value!r}")
                features.append(BINARY_VALUES.get(value.casefold(), column["fill"]))
            else:
                features.extend(float(value == level) for level in column["levels"])
        return features


class CSVDataset(Dataset):
    """Отдать пару (вектор признаков, целевой тензор) из CSV для чтения по батчам."""

    def __init__(self, path, target, task, *, delimiter=",", indices=None, preprocessor=None,
                 classes=None, target_transform=None):
        if task not in {"regression", "binary", "multiclass"}:
            raise ValueError("task должен быть regression, binary или multiclass")
        headers, rows = read_csv(path, target, delimiter)
        if indices is not None:
            rows = [rows[i] for i in indices]
        if not rows:
            raise ValueError("В выбранной части CSV нет объектов")
        self.preprocessor = preprocessor if preprocessor is not None else CSVPreprocessor(headers, rows)
        if set(headers) != self.preprocessor.columns.keys():
            raise ValueError("Схема признаков CSV отличается от обучающей")
        self.features = torch.tensor([self.preprocessor.encode(row) for row in rows], dtype=torch.float32)
        raw_targets = [row[target] for row in rows]
        if target_transform is not None:
            raw_targets = [target_transform(value) for value in raw_targets]
        self.classes = None
        if task in {"binary", "multiclass"}:
            self.classes = list(classes) if classes is not None else sorted(set(raw_targets))
            if len(set(self.classes)) != len(self.classes):
                raise ValueError("Список классов содержит повторяющиеся метки")
            if (task == "binary" and len(self.classes) != 2) or (task == "multiclass" and len(self.classes) < 3):
                raise ValueError("Для binary нужны две метки, для multiclass — минимум три")
            label_map = {label: i for i, label in enumerate(self.classes)}
            if set(raw_targets) - label_map.keys():
                raise ValueError("В CSV найдена неизвестная целевая метка")
            values = [label_map[label] for label in raw_targets]
        else:
            values = [finite_number(value) for value in raw_targets]
            if any(value is None for value in values):
                raise ValueError("Для регрессии целевые значения должны быть конечными числами")
        self.targets = (torch.tensor(values, dtype=torch.long) if task == "multiclass"
                        else torch.tensor(values, dtype=torch.float32).reshape(-1, 1))

    def __len__(self):
        """Вернуть число записей CSV в текущей выборке."""
        return len(self.targets)

    def __getitem__(self, index):
        """Вернуть признаки и целевое значение выбранной строки."""
        return self.features[index], self.targets[index]


def wine_binary_label(value):
    """Преобразовать исходную оценку в низкую или высокую категорию качества."""
    return BINARY_LABELS[int(value) >= BINARY_THRESHOLD]


def load_wine_splits(task, path=WINE_CSV):
    """Исключить точные дубликаты и использовать одно разбиение для всех моделей вина."""
    _, rows = read_csv(path, QUALITY)
    unique = {}
    for index, row in enumerate(rows):
        # Оставляем индекс первого появления каждой строки.
        unique.setdefault(tuple(row.values()), index)
    row_indices = np.array(list(unique.values()))
    labels = [rows[index][QUALITY] for index in row_indices]
    train, validation, test = [row_indices[part] for part in split_indices(len(labels), labels)]
    transform = wine_binary_label if task == "binary" else None
    classes = BINARY_LABELS if task == "binary" else None
    training = CSVDataset(path, QUALITY, task, indices=train, classes=classes, target_transform=transform)
    holdouts = [CSVDataset(path, QUALITY, task, indices=indices, preprocessor=training.preprocessor,
                           classes=training.classes, target_transform=transform) for indices in (validation, test)]
    return training, *holdouts


def fit_wine_model(task):
    """Обучить регрессию или классификатор и напечатать метрики на тестовых строках."""
    training, validation, test = load_wine_splits(task)
    torch.manual_seed(SEED)
    model = nn.Linear(training.features.shape[1], 1)
    if task == "regression":
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
    else:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    _, best_epoch = fit_model(model, training, validation, criterion, optimizer,
                             epochs=160, patience=20, label=f"wine_{task}")
    with torch.no_grad():
        predictions = model(test.features).flatten()
    true_values = test.targets.flatten().numpy()
    if task == "regression":
        mse = float(np.mean((true_values - predictions.numpy()) ** 2))
        metrics = {"mse": mse, "rmse": mse**0.5, "mae": mean_absolute_error(true_values, predictions.numpy()),
                   "r2": r2_score(true_values, predictions.numpy())}
    else:
        positive = predictions.sigmoid().numpy()
        probabilities = np.column_stack((1 - positive, positive))
        metrics = classification_metrics(true_values, probabilities)
        plot_confusion_matrix(true_values, probabilities, training.classes, "confusion_wine_binary.png")
    print(f"wine_{task}: train/val/test={len(training)}/{len(validation)}/{len(test)} best_epoch={best_epoch}",
          {name: round(float(value), 4) for name, value in metrics.items()})
    metadata = {"positive_threshold": BINARY_THRESHOLD} if task == "binary" else {}
    save_model(model, f"wine_{task}.pt", task=task, target=QUALITY, classes=training.classes, **metadata,
               preprocessing=training.preprocessor.columns, feature_names=training.preprocessor.feature_names)


def main():
    """Обучить две модели на одном CSV с разным представлением цели."""
    configure_run()
    fit_wine_model("regression")
    fit_wine_model("binary")


if __name__ == "__main__":
    main()
