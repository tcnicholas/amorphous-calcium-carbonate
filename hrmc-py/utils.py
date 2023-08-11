"""
24.12.21
@tcnicholas
Utility functions.
"""


import itertools
from typing import List
from ctypes import c_int, c_double

import numpy as np
from numba import njit, prange


def unique_pairs(l: List) -> List:
    """
    Gets unique pairs from 1d list (including self).

    :param l: list of elements.
    """
    pairs = list(itertools.combinations(l, r=2)) + [(x,x) for x in l]
    return sorted((sorted(x) for x in pairs), key=lambda x: x[0])


def round_maxr(max_r: float, bin_width: float) -> float:
    """
    Round the max_r value by flooring to a multiple of the bin_width.

    :param max_r: maximum r value.
    :param bin_width: width of r bins.
    :return: rounded max_r value.
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


def ids2cids(ids: List):
    """
    Convert Python list of ids to C-type array.

    :param ids: list of ids.
    :return: C-type array of ids.
    """
    lenids = len(ids)
    cids = (lenids * c_int)()
    for i in range(lenids):
        cids[i] = ids[i]
    return cids


def cdata2pydata(cdata: list, dim: int) -> np.ndarray:
    """
    Convert C-type array to Python list.

    :param cdata: C-type array.
    :param dim: dimension of data.
    :return: Python array of values.
    """
    cdata = list(cdata)
    if dim > 1:
        return np.array([cdata[x : x + dim] for x in range(0, len(cdata), dim)])
    else:
        return np.array(cdata)


def numpy2c(pyVector: np.ndarray, gtype: int):
    """
    Convert NumPy array to C-type array. 
    
    gtype: int
        0 <=> c_int
        1 <=> c_double
    """
    pyVector = np.array(pyVector).flatten()
    if gtype == 0:
        return (len(pyVector) * c_int)(*pyVector)
    else:
        return (len(pyVector) * c_double)(*pyVector)


@njit(parallel=True)
def mask_distances(d, nmax, tol, rcut):
    """
    Until I find a better way of determining the size of the distance vector
    computed by LAMMPS, I need to over-anticipate the vector size.
    """
    total_v = 0
    for n in prange(nmax):
        if d[n] >= tol:
            v = rcut - d[n]
            if v > 0:
                total_v += v
    return total_v
