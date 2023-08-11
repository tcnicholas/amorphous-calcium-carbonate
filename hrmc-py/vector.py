import numpy as np
from numba import njit


@njit
def deg2rad(x: float) -> float:
    """
    Convert degrees to radians.

    :param x: angle in degrees.
    :return: angle in radians.
    """
    return x * np.pi / 180


@njit
def rad2deg(x: float) -> float:
    """
    Convert radians to degrees.

    :param x: angle in radians.
    :return: angle in degrees.
    """
    return x * 180 / np.pi


@njit
def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Transform 1x3 vector to a unit vector.

    :param v: vector.
    :return: unit vector.
    """
    return v / np.sqrt(np.sum(np.square(v)))


@njit
def random_vector(rand1: float, rand2: float) -> np.ndarray:
    """
    Generate random vector.

    :param rand1: random number between 0 and 1.
    :param rand2: random number between 0 and 1.
    :return: random vector.
    """

    # Generate two random variables.
    phi = rand1 * 2 * np.pi
    z = rand2 * 2 - 1

    # Determine final unit vector.
    z2 = z * z
    x = np.sqrt(1 - z2) * np.sin(phi)
    y = np.sqrt(1 - z2) * np.cos(phi)
    return unit_vector(np.array([x,y,z]))


@njit
def rodrigues(a: np.ndarray, b: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues' rotation formula for rotating vector a about vector b by theta
    radians.

    :param a: vector to rotate.
    :param b: vector to rotate about.
    :param theta: angle to rotate by.
    :return: rotated vector.
    """
    return a * np.cos(theta) + np.cross(b,a) * np.sin(theta) + b * np.dot(b,a) * (1 - np.cos(theta))
