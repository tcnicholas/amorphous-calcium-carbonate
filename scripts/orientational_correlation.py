""" 
08.08.23
@tcnicholas
Calculate the orientational correlation function.
"""


from typing import Callable
from itertools import combinations

import numpy as np
from numba import njit
from pymatgen.core import Lattice

from utils import round_decimals_down


@njit
def compute_correlation(
    correlation,
    counts,
    corr_list_buffer,
    corr_list_counts,
    all_combinations,
    coords,
    bin_width,
    orientations,
    distances,
    max_distance
):
    """ 
    Compute the orientational correlation function.
    """
    for i,j in all_combinations:

        if distances[i,j] < max_distance:

            bin_index = int(distances[i,j] // bin_width)
            corr = abs(np.dot(orientations[i], orientations[j])) * 2 - 1
            
            correlation[bin_index] += corr
            counts[bin_index] += 1
            
            corr_list_buffer[corr_list_counts[bin_index], bin_index] = corr
            corr_list_counts[bin_index] += 1
    
    corr_list = [
        corr_list_buffer[:corr_list_counts[i], i] 
        for i in range(len(corr_list_counts))
    ]
    return correlation, counts, corr_list



def compute_orientational_correlation_function(
    all_molecules: list,
    lattice: Lattice,
    orientation_func: Callable,
    molecule_coordinates_attribute: str = 'carbon',
    bin_width: float = 0.05,
    max_distance: float = 10.0,
):
    """
    Compute the orientational correlation function.

    :all_molecules: list of molecules to compute correlation function for.
    :lattice: lattice to compute correlation function for.
    :orientation_func: function to compute orientation of molecule.
    :molecule_coordinates_attribute: attribute of molecule to use for coordinate
        extraction. 'oxygen_cartesian_position' for H2O, 'carbon' for CO3.
    :bin_width: width of bins.
    :max_distance: maximum distance to compute correlation function to.
    :return: tuple of (rs, correlation, counts, std_dev).
    """

    if max_distance is None:
        max_distance = round_decimals_down(np.amin(lattice.lengths) / 2, 2)

    # prepare arrays.
    rs = np.arange(0, max_distance+bin_width, bin_width)[1:] - (bin_width/2)
    correlation = np.zeros(rs.shape, dtype=np.float64)
    counts = np.zeros(rs.shape, dtype=np.intp)

    # also compute standard error of the mean.
    corr_list_buffer = np.zeros(
        (len(all_molecules), int(max_distance // bin_width) + 1)
    )
    corr_list_counts = np.zeros(int(max_distance//bin_width)+1, dtype=np.int64)
    std_dev = np.zeros(rs.shape, dtype=np.float64)

    # precompute all fractional coordinates, orientations, and distances.
    coords = lattice.get_fractional_coords([
        getattr(molecule, molecule_coordinates_attribute) 
        for molecule in all_molecules
    ])
    orientations = np.array([
        orientation_func(molecule) for molecule in all_molecules
    ])
    distances = lattice.get_all_distances(coords, coords)

    indices = np.arange(distances.shape[0])
    all_combinations = np.array(list(combinations(indices, r=2)))

    correlation, counts, corr_list = compute_correlation(
        correlation,
        counts,
        corr_list_buffer,
        corr_list_counts,
        all_combinations,
        coords,
        bin_width,
        orientations, 
        distances,
        max_distance,
    )

    for i,count in enumerate(counts):
        if count > 0:
            correlation[i] /= count
            std_dev[i] = np.std(corr_list[i]) / np.sqrt(len(corr_list[i]))

    # then filter arrays according to non-zero counts at a given distance.
    use = np.argwhere(counts>0).flatten()

    return rs[use], correlation[use], counts[use], std_dev[use]


def carbonate_orientation(molecule):
    """
    Calculate the orientation of a carbonate molecule.
    """
    return molecule.normal


def water_orientation(molecule):
    """
    Calculate the orientation of a water molecule.
    """
    h = molecule._x[np.argwhere(molecule._atypes=="H").flatten()]
    o = molecule.oxygen_cartesian_position
    v = h-o
    assert np.all(np.round(np.linalg.norm(v,axis=1), 3)==0.957)
    n = np.sum(v, axis=0)
    return n / np.linalg.norm(n)