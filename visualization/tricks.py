import numpy as np


def rotate(image):
    image = np.fliplr(image)
    image = np.rot90(image, 2)
    return image