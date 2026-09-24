"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    return sum(1 for char in text.lower() if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        temp = 1
        for e in str(temp):
            temp *= int(e)
        number = temp
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    err = 0
    n = len(predicted)

    for i in range(n):
        diff = predicted[i] - expected[i]
        err += diff**2

    return err / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value

    if number <= 1:
        return f"({number})"

    result = ""

    if number % 2 == 0:
        power = 0
        while number % 2 == 0:
            number //= 2
            power += 1
        result += f"(2**({power})" if power > 1 else f"({2})"

    div = 3
    while div**2 <= number:
        if number % div == 0:
            power = 0
            while number % div == 0:
                number //= div
                power += 1
            result += f"({div}**{power})" if power > 1 else f"({div})"
        div += 2

    if number > 1:
        result += f"({number})"

    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0

    while cube_count > total:
        k += 1
        sum += k**2

    return k if cube_count == total else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = str(data.value)
    l = len(number)

    if l % 2 <= 2:
        return True

    if l % 2 == 0:
        left = number[: l // 2 - 1]
        right = number[l // 2 + 1 :]
    else:
        left = number[: l // 2]
        right = number[l // 2 + 1 :]

    left_sum = 0
    for e in left:
        left_sum += e

    right_sum = 0
    for e in right:
        right_sum += e

    return left_sum == right_sum
