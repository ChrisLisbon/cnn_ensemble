import numpy as np
from skimage.metrics import structural_similarity


def mae(prediction, real):
    return round(float(np.mean(abs(np.array(prediction) - np.array(real)))), 3)


def ssim(prediction, real):
    return round(structural_similarity(prediction, real, data_range=1), 3)


def accuracy(prediction, real):
    f = 125 * 125
    diff = prediction - real
    unique, counts = np.unique(diff, return_counts=True)
    right_calculated = dict(zip(unique, counts))[0]
    return round(right_calculated / f, 3)


def accuracy_without_mask(pred, real, mask):
    pred = pred[mask == 1]
    real = real[mask == 1]
    f = pred.shape[0]
    diff = pred - real
    unique, counts = np.unique(diff, return_counts=True)
    right_calculated = dict(zip(unique, counts))[0]
    return round(right_calculated / f, 3)


def correlation(prediction, real):
    return np.corrcoef(prediction, real)[0, 1]
