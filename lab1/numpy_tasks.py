"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.einsum("pij,pjk->ik", matrices, vectors)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = (
        data.rows,
        data.columns,
        data.mean,
        data.std,
        data.seed,
    )

    rng = np.random.default_rng(seed=52)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))

    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    row_vars = matrix.var(axis=1)
    col_vars = matrix.var(axis=0)

    return MatrixStatistics(matrix, row_means, col_means, row_vars, col_vars)


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second

    i = np.arange(rows)[:, None]
    j = np.arange(columns)

    return np.where((i + j) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color)
    start_y = (image_height - height) // 2
    start_x = (image_width - width) // 2
    image[start_y : start_y + height, start_x : start_x + width] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.full((image_height, image_width, 3), background_color)
    y0 = (image_height - 1) / 2
    x0 = (image_width - 1) / 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - x0) ** 2) / (semi_axis_x**2) + ((y - y0) ** 2) / (semi_axis_y**2) <= 1
    image[mask] = shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window

    mean = float(np.mean(values))
    var = float(np.var(values))
    std = float(np.std(values))

    max_test = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    local_max = np.where(max_test)[0] + 1

    min_test = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    local_min = np.where(min_test)[0] + 1

    new_row = np.convolve(values, np.ones(window) / window, mode="valid")

    return TimeSeriesStatistics(mean, var, std, local_max, local_min, new_row)


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count

    if class_count:
        matrix = np.zeros((len(labels), class_count), dtype=int)
    else:
        matrix = np.zeros((len(labels), max(labels) + 1), dtype=int)

    for row, e in enumerate(labels):
        matrix[row, e] = 1

    return matrix
