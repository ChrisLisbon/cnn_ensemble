from copy import deepcopy

import numpy as np


def rotate(image):
    image = np.fliplr(image)
    image = np.rot90(image, 2)
    return image


def binary(matrix):
    matrix_copy = deepcopy(matrix)
    matrix_copy[matrix_copy >= 0.8] = 1
    matrix_copy[matrix_copy < 0.8] = 0
    return matrix_copy


def gradate(raw_array):
    array = deepcopy(raw_array)
    array[np.where((array > 0.9))] = 9
    array[np.where((array > 0.8) & (array <= 0.9))] = 8
    array[np.where((array > 0.7) & (array <= 0.8))] = 7
    array[np.where((array > 0.6) & (array <= 0.7))] = 6
    array[np.where((array > 0.4) & (array <= 0.6))] = 4
    array[np.where((array > 0.3) & (array <= 0.4))] = 3
    array[np.where((array > 0.2) & (array <= 0.3))] = 2
    array[np.where((array > 0.1) & (array <= 0.2))] = 1
    array[array <= 0.1] = 0
    return array