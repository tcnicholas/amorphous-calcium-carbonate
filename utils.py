"""
30.06.23
@tcnicholas
General utility functions for analysis. These functions have been centralised
because they are used for both the main and supplementary analysis.
"""


import math
import pickle
import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
from ase.io import read
from pymatgen.core.lattice import Lattice

from scripts.plot import *
from scripts.molecules import calcium, carbonate, water


MOLECULES = ['ca', 'co3', 'h2o']
BASE_PATH = pathlib.Path('simulations/hrmc')
PATH_TO_SNAPSHOT = BASE_PATH / 'molecules/1500'


def load_data(filename: pathlib.Path) -> List:
    """
    Load data from a pickle file.
    
    :param filename: Path to the pickle file.
    :return: The loaded data.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def load_lattice() -> Lattice:
    """
    Load the lattice from a pickle file.
    
    :return: The loaded lattice.
    """
    atoms = read(
        BASE_PATH / 'structure' / 'HRMC_150M_proposed_moves.data', 
        format='lammps-data'
    )
    return Lattice(atoms.cell.array)
    

def retrieve_hrmc_data(data_type='all') -> Dict:
    """
    Retrieve the data for the HRMC simulations.

    :param data_type: type of data to retrieve.
    :return: the requested data.
    """

    lattice = load_lattice()

    molecule_data = {molecule: load_data(PATH_TO_SNAPSHOT/f'{molecule}.pickle') 
                     for molecule in MOLECULES}
    
    molecule_dict = {molecule: {obj.id: obj for obj in data} 
                     for molecule, data in molecule_data.items()}
    
    all_data = {
        'lattice': lattice,
        'molecules_data': molecule_data,
        'molecule_dict': molecule_dict
    }

    if data_type.lower() == 'all':
        return all_data

    return all_data[data_type.lower()]


def round_decimals_down(number: float, decimals: int = 2) -> float:
    """
    Returns a value rounded down to a specific number of decimal places.

    :param number: The number to round down.
    :param decimals: The number of decimal places to round down to.
    :return: The rounded down number.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor



