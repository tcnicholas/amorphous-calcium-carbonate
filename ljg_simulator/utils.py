"""
28.06.23
@tcnicholas
Utility functions.
"""

import re
from typing import Tuple, List

import numpy as np
from numba import njit


def round_maxr(max_r: float, bin_width: float) -> float:
    """
    Round the max_r value by flooring to a multiple of the bin_width.

    :param max_r: maximum value of r.
    :param bin_width: width of the bins.
    :return: rounded max_r.
    """

    # get the number of decimal points to round to.
    try:
        rv = len(str(bin_width).split(".")[-1])
    except:
        rv = 0

    top_bin = bin_width * np.round(max_r / bin_width)
    if top_bin > max_r:
        top_bin -= bin_width
    
    return np.round(top_bin, rv)


@njit
def unit_vector(v: np.ndarray(3)) -> np.ndarray(3):
    """
    Transform 1x3 vector to a unit vector.

    :param v: vector.
    :return: unit vector.
    """
    return v / np.sqrt(np.sum(np.square(v)))


@njit
def random_vector(rand1: float, rand2: float) -> np.ndarray(3):
    """
    Generate random vector.
    
    :param rand1: random number 1.
    :param rand2: random number 2.
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


def eps2H(eps: float, sigma: float) -> float:
    """
    The epsilon prefactor for the Gaussian potential in LAMMPS is defined as
    
        \epsilon = \frac{H}{\sigma_h \sqrt{2\pi}}
        
    where H determines--together with the standard deviation \sigma_h--the
    peak height of the Guassian function. As such, given we have fit this to
    epsilon, we need to compute H.

    :param eps: epsilon.
    :param sigma: sigma.
    :return: H.
    """
    # guass params: eps, r0, sigma
    return -eps * np.sqrt(2*np.pi) * sigma


def extract_headers_and_index(line: str) -> Tuple[List[str], int]:
    headers = re.split(r'\s+', re.sub(',', '', line.strip()))
    if '|' not in headers:
        raise ValueError("The log file does not contain a '|' symbol.")
    ix = headers.index('|')
    headers.remove('|')
    return headers, ix


def handle_data_line(line: str, ix: int) -> List[float]:
    try:
        line_values = re.split(r'\s+', line.strip())
        return [np.nan if v == '-' else float(v) for v in (
            line_values[:ix] + line_values[ix+1:]
        )]
    except ValueError:
        return []


def load_mc_ljg_log(filename: str) -> Tuple[List[str], np.ndarray]:
    """
    Parse the log file output from LJG simulator Monte Carlo simulations and
    return simulation headers and the corresponding data.
    
    :param filename: The path to the log file.
    :return: A tuple containing list of headers in the log file and the data.
    """

    headers, data, ix = [], [], None

    try:
        with open(filename, 'r') as file:
            for line in file:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith('Tracking simulation progress:'):
                    headers, ix = extract_headers_and_index(file.readline())
                    file.readline()  # skip one line
                elif ix is not None:
                    data_line = handle_data_line(line, ix)
                    if data_line:
                        data.append(data_line)
                    else:
                        break
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} does not exist.")
    except ValueError as ve:
        raise ValueError(f"Error while reading the file: {ve}")

    return {'headers': headers, 'data': np.array(data, dtype=float)}
