"""Модульные тесты критических частей домашнего задания по регрессии."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

if __package__:
    from .homework_datasets import CSVDataset, load_wine_splits
    from .homework_experiments import make_features
    from .homework_training import classification_metrics, fit_model, regularization
else:
    from homework_datasets import CSVDataset, load_wine_splits
    from homework_experiments import make_features
    from homework_training import classification_metrics, fit_model, regularization


class TrainingTests(unittest.TestCase):
    def test_regularization_value_and_gradient(self):
        weights = torch.tensor([-2.0, 1.0], requires_grad=True)
        penalty = regularization(weights, l1=0.3, l2=0.2)
        penalty.backward()
        torch.testing.assert_close(penalty, torch.tensor(1.9))
        torch.testing.assert_close(weights.grad, torch.tensor([-1.1, 0.7]))

    def test_early_stopping_restores_best_weights(self):
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        dataset = TensorDataset(torch.tensor([[1.0]]), torch.tensor([[2.0]]))
        observed_weights = []

        def validation_loss(*args):
            observed_weights.append(model.weight.detach().clone())
            return (2.0, 1.0, 1.5, 2.0)[len(observed_weights) - 1]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with patch(f"{fit_model.__module__}.evaluate_loss", side_effect=validation_loss):
            errors, best_epoch = fit_model(model, dataset, dataset, nn.MSELoss(), optimizer, epochs=10, patience=2)
        self.assertEqual(best_epoch, 2)
        self.assertEqual(len(errors), 4)
        self.assertFalse(torch.equal(observed_weights[1], observed_weights[-1]))
        torch.testing.assert_close(model.weight, observed_weights[1])


class MetricTests(unittest.TestCase):
    def test_binary_auc_uses_scores(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = classification_metrics([0, 0, 1, 1], np.column_stack((1 - scores, scores)))
        self.assertEqual(metrics['roc_auc'], 1.0)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['recall'], 0.0)

    def test_multiclass_macro_metrics(self):
        targets = [0, 0, 1, 1, 2, 2]
        probabilities = np.eye(3)[[0, 1, 1, 1, 2, 0]]
        metrics = classification_metrics(targets, probabilities)
        self.assertAlmostEqual(metrics['precision'], 13 / 18)
        self.assertAlmostEqual(metrics['recall'], 2 / 3)
        self.assertAlmostEqual(metrics['f1'], 59 / 90)
        self.assertAlmostEqual(metrics['roc_auc'], 3 / 4)


class DatasetTests(unittest.TestCase):
    def test_wine_tasks_share_disjoint_splits_and_map_quality_to_labels(self):
        regression = load_wine_splits('regression')
        binary = load_wine_splits('binary')
        multiclass = load_wine_splits('multiclass')
        seen = set()
        self.assertEqual(sum(map(len, regression)), 1359)
        for numeric, thresholded, categorical in zip(regression, binary, multiclass):
            self.assertEqual(numeric.features.shape[1], 11)
            torch.testing.assert_close(numeric.features, thresholded.features)
            torch.testing.assert_close(numeric.features, categorical.features)
            torch.testing.assert_close(thresholded.targets, (numeric.targets >= 6).float())
            scores = torch.tensor([int(label) for label in categorical.classes])
            torch.testing.assert_close(scores[categorical.targets], numeric.targets.flatten().long())
            current = set(map(tuple, numeric.features.tolist()))
            self.assertTrue(seen.isdisjoint(current))
            seen.update(current)

    def test_numeric_binary_and_categories_use_training_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.csv'
            path.write_text('size;flag;color;target\n10;yes;red;low\n20;false;blue;high\n'
                            '100;1;green;low\n;;red;high\n', encoding='utf-8')
            train = CSVDataset(path, 'target', 'binary', delimiter=';', indices=[0, 1])
            held_out = CSVDataset(path, 'target', 'binary', delimiter=';', indices=[2, 3],
                                  preprocessor=train.preprocessor, classes=train.classes)
        self.assertEqual(train.classes, ['high', 'low'])
        torch.testing.assert_close(train[0][0], torch.tensor([-1.0, 1.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(held_out[0][0], torch.tensor([17.0, 1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(held_out[1][0], torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(held_out.targets, torch.tensor([[1.0], [0.0]]))

    def test_malformed_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.csv'
            path.write_text('x,y\n1,2,3\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'число значений'):
                CSVDataset(path, 'y', 'regression')


class FeatureTests(unittest.TestCase):
    def test_feature_values_and_order(self):
        features = torch.tensor([[1.0, 3.0, 5.0]])
        expanded = make_features(features, polynomial=True, interactions=True, statistics=True)
        expected = torch.tensor([[1.0, 3.0, 5.0, 1.0, 9.0, 25.0, 3.0, 5.0, 15.0, 3.0, 8 / 3]])
        torch.testing.assert_close(expanded, expected)


if __name__ == '__main__':
    unittest.main()
