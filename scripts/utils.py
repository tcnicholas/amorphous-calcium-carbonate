"""
27.06.23
@tcnicholas
Utility functions.
"""


import pathlib
from typing import List
from IPython import get_ipython
from ctypes import c_int, c_double

import numpy as np
from ase.io import read
from pymatgen.core import Lattice
from ase.build import make_supercell


def mkdir(dirname: str) -> pathlib.Path:
    """
    Generate a directory in one line.

    :param dirname: name of directory to make.
    :return: pathlib.Path object for directory.
    """
    directory = pathlib.Path(dirname)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sorted_directories(parent: pathlib.Path) -> List:
    """
    Get all directories in a parent directory, sorted by name.

    :param parent: parent directory.
    :return: list of directories.
    """
    parent = pathlib.Path(parent)
    directories = [d for d in parent.iterdir() if d.is_dir()]

    def dir_key(x):
        try:
            return float(x.name)
        except ValueError:
            return float('inf')

    return sorted(directories, key=dir_key)


def no_jupyter(func):
    def wrapper(*args, **kwargs):
        if get_ipython() is not None:
            raise RuntimeError(
                "This function cannot be run in a Jupyter notebook environment."
            )
        return func(*args, **kwargs)
    return wrapper


def unit(vector: np.ndarray) -> np.ndarray(1):
    """
    Return the unit vector of a given vector.

    :param vector: vector to normalise.
    :return: unit vector.
    """
    return vector / np.sqrt(np.sum(np.square(vector)))


def angle(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors (in radians).

    :param vector1: first vector.
    :param vector2: second vector.
    :return: angle between vectors.
    """

    d = np.dot(unit(vector1), unit(vector2))
    
    # np.clip() to be included in numba v0.54 so can update then accordingly.
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0 
        
    return np.arccos(d)


def get_lattice(dataFile, supercell=None):
    """
    Extract lattice from LAMMPS data file and store as Pymatgen Lattice object.
    """
    atoms = read(dataFile, format="lammps-data")
    if supercell is not None:
        atoms = make_supercell(atoms, supercell)
    return Lattice(atoms.get_cell()[:])


def ids2cids(ids):
    """ """
    lenids = len(ids)
    cids = (lenids * c_int)()
    for i in range(lenids):
        cids[i] = ids[i]
    return cids


def cdata2pydata(cdata, dim):
    cdata = list(cdata)
    if dim > 1:
        return np.array([cdata[x : x + dim] for x in range(0, len(cdata), dim)])
    else:
        return np.array(cdata)


def numpy2c(pyVector, gtype):
    """ Convert NumPy array to C-type array.
    
    gtype: int
        0 <=> c_int
        1 <=> c_double
    """
    pyVector = np.array(pyVector).flatten()
    if gtype == 0:
        return (len(pyVector) * c_int)(*pyVector)
    else:
        return (len(pyVector) * c_double)(*pyVector)