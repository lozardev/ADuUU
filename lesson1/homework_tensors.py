"""Задание 1. Создание тензоров, операции, срезы и изменение формы."""

import torch


def create_tensors():
    """Создать четыре тензора и проверить их формы и значения."""
    random_values = torch.rand(3, 4)
    zeros = torch.zeros(2, 3, 4)
    ones = torch.ones(5, 5)
    sequence = torch.arange(16).reshape(4, 4)

    assert random_values.shape == (3, 4)
    assert bool(((random_values >= 0) & (random_values < 1)).all())
    assert zeros.shape == (2, 3, 4) and torch.count_nonzero(zeros).item() == 0
    assert ones.shape == (5, 5) and bool((ones == 1).all())
    assert sequence.shape == (4, 4)
    torch.testing.assert_close(sequence[-1], torch.tensor([12, 13, 14, 15]))

    for label, tensor in (
        ("Случайные числа", random_values), ("Нули", zeros),
        ("Единицы", ones), ("Числа от 0 до 15", sequence),
    ):
        print(f"{label}, форма {tuple(tensor.shape)}\n{tensor}")


def tensor_operations(a, b):
    """Вернуть транспонированный A, два произведения и сумму элементов A."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A и B должны быть тензорами PyTorch")
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.T.shape:
        raise ValueError("A и B должны иметь формы (m, n) и (n, m)")
    if a.dtype != b.dtype or a.dtype == torch.bool:
        raise TypeError("A и B должны иметь одинаковый числовой dtype")
    if a.device != b.device:
        raise ValueError("A и B должны находиться на одном устройстве")
    return a.T, a @ b, a * b.T, a.sum()


def show_operations():
    """Выполнить операции с матрицами 3×4 и 4×3 и сверить результаты."""
    a = torch.arange(1, 13).reshape(3, 4)
    b = torch.arange(1, 13).reshape(4, 3)
    transposed, product, elementwise, total = tensor_operations(a, b)

    torch.testing.assert_close(transposed, torch.tensor([
        [1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12],
    ]))
    torch.testing.assert_close(product, torch.tensor([
        [70, 80, 90], [158, 184, 210], [246, 288, 330],
    ]))
    torch.testing.assert_close(elementwise, torch.tensor([
        [1, 8, 21, 40], [10, 30, 56, 88], [27, 60, 99, 144],
    ]))
    assert total.item() == 78
    print(f"A.T\n{transposed}\nA @ B\n{product}\nA * B.T\n{elementwise}\nСумма A = {total.item()}")


def show_slices():
    """Показать срезы тензора, оси которого задают слой, строку и столбец."""
    tensor = torch.arange(125).reshape(5, 5, 5)
    first_row = tensor[:, 0, :]
    last_column = tensor[:, :, -1]
    # Берём центральный слой и блок 2×2 с индексом 2 в нижнем правом углу.
    central_block = tensor[2, 1:3, 1:3]
    even_indices = tensor[::2, ::2, ::2]

    assert first_row.shape == last_column.shape == (5, 5)
    torch.testing.assert_close(first_row[0], torch.arange(5))
    torch.testing.assert_close(last_column[0], torch.tensor([4, 9, 14, 19, 24]))
    torch.testing.assert_close(central_block, torch.tensor([[56, 57], [61, 62]]))
    assert even_indices.shape == (3, 3, 3) and even_indices[-1, -1, -1].item() == 124

    for label, values in (
        ("Первая строка каждого слоя", first_row),
        ("Последний столбец каждого слоя", last_column),
        ("Блок 2×2 в центральном слое", central_block),
        ("Чётные индексы по всем осям", even_indices),
    ):
        print(f"{label}\n{values}")


def show_reshapes():
    """Изменить форму последовательности и проверить сохранение порядка."""
    values = torch.arange(24)
    for shape in ((2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)):
        reshaped = values.reshape(shape)
        assert reshaped.shape == shape
        torch.testing.assert_close(reshaped.flatten(), values)
        print(f"Форма {shape}\n{reshaped}")


def main():
    """Запустить примеры и проверки всех четырёх подпунктов."""
    torch.manual_seed(42)
    for title, example in (
        ("1.1 Создание тензоров", create_tensors),
        ("1.2 Операции", show_operations),
        ("1.3 Срезы", show_slices),
        ("1.4 Формы", show_reshapes),
    ):
        print(f"\n{title}")
        example()
    print("\nПроверки тензоров пройдены.")


if __name__ == "__main__":
    main()
