import numpy as np


def normalize(points):
    """
    Normalize the points so that the last row is 1.
    :param points:
    :return: normalized points
    """
    for row in points:
        row /= points[-1]
    return points


def make_homogeneous(points):
    """
    Convert points to homogeneous coordinates.
    :param points:
    :return: homogeneous coordinates
    """
    return np.vstack((points, np.ones((1, points.shape[1]))))