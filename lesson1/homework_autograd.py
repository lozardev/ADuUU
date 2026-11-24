"""Задание 2. Автоматическое дифференцирование и проверка градиентов."""

import math

import torch


def polynomial_gradients():
    """Сверить градиенты многочлена с формулами частных производных."""
    x_value, y_value, z_value = 1.0, 2.0, 3.0
    x = torch.tensor(x_value, requires_grad=True)
    y = torch.tensor(y_value, requires_grad=True)
    z = torch.tensor(z_value, requires_grad=True)
    result = x**2 + y**2 + z**2 + 2 * x * y * z
    result.backward()

    expected = torch.tensor([
        2 * x_value + 2 * y_value * z_value,
        2 * y_value + 2 * x_value * z_value,
        2 * z_value + 2 * x_value * y_value,
    ])
    gradients = torch.stack([x.grad, y.grad, z.grad])
    torch.testing.assert_close(gradients, expected)
    print(f"2.1 f = {result.item():g}, градиенты (dx, dy, dz) = {gradients.tolist()}")


def mean_squared_error(predictions, targets):
    """Вычислить среднее квадратов ошибок между элементами двух тензоров."""
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Предсказания и цели должны быть тензорами PyTorch")
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError("Предсказания и цели должны иметь одинаковую непустую форму")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("Предсказания и цели должны содержать вещественные числа")
    if predictions.device != targets.device:
        raise ValueError("Предсказания и цели должны находиться на одном устройстве")
    return (predictions - targets).square().mean()


def mse_gradients():
    """Найти градиенты MSE по весу и смещению линейной модели."""
    x = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 4.0, 6.0])
    weight = torch.tensor(1.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    loss = mean_squared_error(weight * x + bias, targets)
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(14 / 3))
    torch.testing.assert_close(weight.grad, torch.tensor(-28 / 3))
    torch.testing.assert_close(bias.grad, torch.tensor(-4.0))
    print(f"2.2 MSE = {loss.item():.6f}, dw = {weight.grad.item():.6f}, db = {bias.grad.item():g}")


def chain_rule():
    """Сверить backward и autograd.grad с производной составной функции."""
    x = torch.tensor(1.0, requires_grad=True)
    result = torch.sin(x**2 + 1)
    result.backward()

    # Вычисляем функцию заново для второго способа расчёта градиента.
    gradient = torch.autograd.grad(torch.sin(x**2 + 1), x)[0]
    expected = torch.tensor(2 * math.cos(2))
    torch.testing.assert_close(x.grad, gradient)
    torch.testing.assert_close(gradient, expected)
    print(f"2.3 backward = {x.grad.item():.6f}, autograd.grad = {gradient.item():.6f}")


def main():
    """Запустить три примера и проверить вычисленные градиенты."""
    polynomial_gradients()
    mse_gradients()
    chain_rule()
    print("Проверки градиентов пройдены.")


if __name__ == "__main__":
    main()
