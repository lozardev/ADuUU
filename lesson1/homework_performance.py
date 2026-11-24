"""Задание 3. Измерение времени тензорных операций на CPU и CUDA."""

import time

import torch


SHAPES = ((64, 1024, 1024), (128, 512, 512), (256, 256, 256))


def operations_for(a, b):
    """Подготовить операции над двумя пакетами квадратных матриц float32."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A и B должны быть тензорами PyTorch")
    if a.ndim != 3 or a.shape != b.shape or a.shape[1] != a.shape[2] or a.numel() == 0:
        raise ValueError("Ожидаются одинаковые непустые формы (пакет, n, n)")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("Для сравнения нужны два тензора float32")
    if a.device != b.device:
        raise ValueError("Оба тензора должны находиться на одном устройстве")
    return {
        "Матричное умножение": lambda: torch.matmul(a, b),
        "Сложение": lambda: a + b,
        "Поэлементное умножение": lambda: a * b,
        "Транспонирование": lambda: a.transpose(-2, -1),
        "Сумма всех элементов": lambda: a.sum(),
    }


def measure_ms(operation, device, repetitions):
    """Прогреть операцию и вернуть время одного вызова в миллисекундах."""
    operation()
    if device.type == "cpu":
        start = time.time()
        for _ in range(repetitions):
            operation()
        return (time.time() - start) * 1_000 / repetitions

    stream = torch.cuda.current_stream(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(stream)
    for _ in range(repetitions):
        operation()
    end_event.record(stream)
    # Событие конца должно выполниться на GPU до чтения времени.
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / repetitions


def check_operations(device):
    """Сверить пять операций на маленьких матрицах перед замерами."""
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=device)
    b = torch.tensor([[[2.0, 0.0], [1.0, 2.0]]], device=device)
    expected = {
        "Матричное умножение": [[[4.0, 4.0], [10.0, 8.0]]],
        "Сложение": [[[3.0, 2.0], [4.0, 6.0]]],
        "Поэлементное умножение": [[[2.0, 0.0], [3.0, 8.0]]],
        "Транспонирование": [[[1.0, 3.0], [2.0, 4.0]]],
        "Сумма всех элементов": 10.0,
    }
    for name, operation in operations_for(a, b).items():
        torch.testing.assert_close(operation(), torch.tensor(expected[name], device=device))


def benchmark_shape(shape, cpu_device, cuda_device):
    """Сравнить операции на одном наборе данных и напечатать таблицу."""
    a = torch.rand(shape, device=cpu_device, dtype=torch.float32)
    b = torch.rand(shape, device=cpu_device, dtype=torch.float32)
    cpu_operations = operations_for(a, b)
    cuda_operations = {}
    if cuda_device is not None:
        cuda_operations = operations_for(a.to(cuda_device), b.to(cuda_device))

    print(f"\nФорма тензоров {shape}", flush=True)
    print(f"{'Операция':<24} | {'CPU (мс)':>11} | {'CUDA (мс)':>11} | {'Ускорение':>10}")
    speeds = {}
    for name, operation in cpu_operations.items():
        # Транспонирование создаёт представление и занимает доли миллисекунды.
        repetitions = 1_000 if name == "Транспонирование" else 1
        cpu_ms = measure_ms(operation, cpu_device, repetitions)
        cuda_text = speed_text = "—"
        if name in cuda_operations:
            cuda_ms = measure_ms(cuda_operations[name], cuda_device, repetitions)
            cuda_text = f"{cuda_ms:.6f}"
            if cuda_ms > 0:
                speeds[name] = cpu_ms / cuda_ms
                speed_text = f"{speeds[name]:.2f}x"
        print(f"{name:<24} | {cpu_ms:>11.6f} | {cuda_text:>11} | {speed_text:>10}")

    if speeds:
        fastest = max(speeds, key=speeds.get)
        print(f"Наибольшее отношение CPU/CUDA: {fastest} ({speeds[fastest]:.2f}x).")


def main():
    """Проверить операции и измерить их на трёх размерах входа."""
    torch.manual_seed(42)
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else None
    print(f"Потоки PyTorch на CPU: {torch.get_num_threads()}")
    if cuda_device is None:
        print("CUDA недоступна. Выполняются замеры на CPU.")
    else:
        print(f"Устройство CUDA: {torch.cuda.get_device_name(cuda_device)}")

    check_operations(cpu_device)
    if cuda_device is not None:
        check_operations(cuda_device)
    print("Проверки операций пройдены.")
    for shape in SHAPES:
        benchmark_shape(shape, cpu_device, cuda_device)


if __name__ == "__main__":
    main()
