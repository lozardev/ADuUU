"""Общий цикл обучения и метрики для трёх заданий."""

import logging
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LESSON_DIR = Path(__file__).resolve().parent
MODELS = LESSON_DIR / "models"
PLOTS = LESSON_DIR / "plots"
SEED = 42
LOGGER = logging.getLogger(__name__)


def split_indices(size, labels=None):
    """Разбить индексы примерно 60/20/20, при наличии меток сохранить доли классов."""
    labels = None if labels is None else np.asarray(labels).reshape(-1)
    train, remaining = train_test_split(
        np.arange(size), test_size=0.4, random_state=SEED, stratify=labels,
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, random_state=SEED,
        stratify=None if labels is None else labels[remaining],
    )
    return train, validation, test


def regularization(weights, l1=0.0, l2=0.0):
    """Посчитать L1 и L2 для весов без штрафа на свободный член."""
    if l1 < 0 or l2 < 0:
        raise ValueError("Коэффициенты регуляризации должны быть неотрицательны")
    return l1 * weights.abs().sum() + l2 * weights.square().sum()


@torch.no_grad()
def evaluate_loss(model, dataset, criterion):
    """Посчитать ошибку на всей небольшой выборке без обновления весов."""
    model.eval()
    features, targets = dataset[:]
    return criterion(model(features), targets).item()


def fit_model(model, training, validation, criterion, optimizer, *, epochs=100,
              batch_size=32, patience=None, l1=0.0, l2=0.0, label="model"):
    """Обучить модель и вернуть список ошибок валидации и номер лучшей эпохи."""
    if min(len(training), len(validation), epochs, batch_size) < 1:
        raise ValueError("Нужны непустые выборки и положительные epochs и batch_size")
    if patience is not None and patience < 1:
        raise ValueError("patience должен быть положительным")
    loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    validation_losses = []
    best_loss, best_epoch, best_weights = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            # Один шаг обучения на одном батче.
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            if l1 or l2:
                loss = loss + regularization(model.weight, l1, l2)
            loss.backward()
            optimizer.step()

        validation_loss = evaluate_loss(model, validation, criterion)
        if not np.isfinite(validation_loss):
            raise FloatingPointError(f"{label}: ошибка валидации не конечна")
        validation_losses.append(validation_loss)
        # Копия нужна, потому что optimizer.step() продолжит менять веса.
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_weights = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            LOGGER.info("%s: эпоха %d, ошибка на валидации %.5f", label, epoch, validation_loss)
        if patience is not None and epoch - best_epoch >= patience:
            LOGGER.info("%s: остановка на эпохе %d, лучшая эпоха %d", label, epoch, best_epoch)
            break

    model.load_state_dict(best_weights)
    return validation_losses, best_epoch


def classification_metrics(targets, probabilities):
    """Вычислить метрики по меткам 0..K-1 и вероятностям порядка (объект, класс)."""
    targets = np.asarray(targets).reshape(-1)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(targets) or probabilities.shape[1] < 2:
        raise ValueError("Нужна матрица вероятностей размера (число объектов, число классов)")
    classes = np.arange(probabilities.shape[1])
    if not np.array_equal(np.unique(targets), classes):
        raise ValueError("Для ROC-AUC в выборке должны быть представлены все классы 0..K-1")
    predicted = probabilities.argmax(axis=1)
    binary = len(classes) == 2
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predicted, labels=classes, average="binary" if binary else "macro", zero_division=0,
    )
    auc = (roc_auc_score(targets, probabilities[:, 1]) if binary else
           roc_auc_score(targets, probabilities, labels=classes, multi_class="ovr", average="macro"))
    return {"accuracy": accuracy_score(targets, predicted), "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": auc}


def save_figure(figure, filename):
    """Сохранить график в папку урока в режиме без графического интерфейса."""
    PLOTS.mkdir(exist_ok=True)
    figure.savefig(PLOTS / filename, dpi=140, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(targets, probabilities, class_names, filename):
    """Сохранить матрицу ошибок, где строки истинные, а столбцы предсказанные."""
    figure, axis = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(targets, probabilities.argmax(axis=1),
                                            labels=range(len(class_names)), display_labels=class_names,
                                            cmap="Blues", ax=axis)
    axis.set(xlabel="Предсказанный класс", ylabel="Истинный класс")
    save_figure(figure, filename)


def save_model(model, filename, **metadata):
    """Сохранить веса, размеры линейного слоя и параметры подготовки входа."""
    MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_features": model.in_features,
                "out_features": model.out_features, **metadata}, MODELS / filename)


def configure_run():
    """Настроить логирование и один поток для небольших CPU-батчей."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.set_num_threads(1)
