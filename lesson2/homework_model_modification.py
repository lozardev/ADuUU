"""Регуляризация и многоклассовая классификация на CSV красного вина."""

import torch
from torch import nn

if __package__:
    from .homework_datasets import QUALITY, load_wine_splits
    from .homework_training import (
        SEED, classification_metrics, configure_run,
        evaluate_loss, fit_model, plot_confusion_matrix, plt, save_figure, save_model,
    )
else:
    from homework_datasets import QUALITY, load_wine_splits
    from homework_training import (
        SEED, classification_metrics, configure_run,
        evaluate_loss, fit_model, plot_confusion_matrix, plt, save_figure, save_model,
    )


def run_regularized_regression():
    """Сравнить линейные модели по ошибке валидации для исходной оценки качества."""
    training, validation, test = load_wine_splits("regression")
    figure, axis = plt.subplots()
    criterion = nn.MSELoss()
    for name, l1, l2 in (("baseline", 0.0, 0.0), ("L1", 0.02, 0.0), ("L2", 0.0, 0.02)):
        torch.manual_seed(SEED)
        model = nn.Linear(training.features.shape[1], 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
        losses, best_epoch = fit_model(model, training, validation, criterion, optimizer,
                                      epochs=160, patience=20, l1=l1, l2=l2, label=name)
        axis.plot(range(1, len(losses) + 1), losses, label=name)
        print(f"{name:8} лучшая эпоха={best_epoch:3} MSE на тесте={evaluate_loss(model, test, criterion):.4f}")
        if l1 or l2:
            save_model(model, f"wine_regularization_{name.lower()}.pt", task="regression",
                       target=QUALITY, preprocessing=training.preprocessor.columns,
                       feature_names=training.preprocessor.feature_names, l1=l1, l2=l2)
    axis.set(xlabel="Эпоха", ylabel="MSE на валидации")
    axis.legend()
    save_figure(figure, "regularization.png")


def run_multiclass_classification():
    """Обучить модель, которая предсказывает одну из наблюдаемых оценок вина."""
    training, validation, test = load_wine_splits("multiclass")
    torch.manual_seed(SEED)
    model = nn.Linear(training.features.shape[1], len(training.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    fit_model(model, training, validation, nn.CrossEntropyLoss(), optimizer,
              epochs=150, patience=20, label="wine_multiclass")
    with torch.no_grad():
        probabilities = model(test.features).softmax(dim=1).numpy()
    true_classes = test.targets.numpy()
    scores = classification_metrics(true_classes, probabilities)
    print("wine_multiclass test (macro/OvR)", {name: round(value, 4) for name, value in scores.items()})
    plot_confusion_matrix(true_classes, probabilities, training.classes, "confusion_wine_multiclass.png")
    save_model(model, "wine_multiclass.pt", task="multiclass", target=QUALITY, classes=training.classes,
               preprocessing=training.preprocessor.columns, feature_names=training.preprocessor.feature_names)


def main():
    """Запустить оба эксперимента первого задания."""
    configure_run()
    run_regularized_regression()
    run_multiclass_classification()


if __name__ == "__main__":
    main()
