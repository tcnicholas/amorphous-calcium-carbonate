"""
01.07.23
@tcnicholas
Tasks for the neutron total scattering computes.
"""


import pathlib
from typing import Dict, Any, List

import pymatgen
import numpy as np
from ase.io import read
from pymatgen.core import Structure



def convert_fm_to_angstrom(x):
    """
    Convert femtometers to Angstroms.
    """
    return x * 1E-5


# Neutron coherent scattering lengths taken from: 
# https://www.ncnr.nist.gov/resources/n-lengths/
# Conversion from femtometers to Angstroms
neutron_scattering_lengths = {
    "Ca": convert_fm_to_angstrom(4.70),
    "Zn": convert_fm_to_angstrom(5.680),
    "C": convert_fm_to_angstrom(6.6460),
    "N": convert_fm_to_angstrom(9.36),
    "H1": convert_fm_to_angstrom(-3.7406),
    "H2": convert_fm_to_angstrom(6.671),
    "O": convert_fm_to_angstrom(5.803),
}


def calculate_rsum(q_values, r_values, binwidth):
    """
    Precompute the factor of r^2 * sin(Qr)/Qr * dr for the Fourier Transform.
    This function returns an array where each row corresponds to a different Q
    and each column a different r.
    """
    r2dr = np.square(r_values) * binwidth
    rsum = np.zeros((q_values.shape[0], r_values.shape[0]), dtype=np.float64)
    for q_idx in range(q_values.shape[0]):
        qr = q_values[q_idx] * r_values
        rsum[q_idx, :] = np.multiply(np.divide(np.sin(qr), qr), r2dr)
    return rsum


def calculate_neutron_FQ(q_values, rdf, constants, r_sum):
    """
    Compute the neutron F(Q).
    """
    fq = np.zeros((constants.shape[0], q_values.shape[0]), dtype=np.float64)
    for q_idx in range(q_values.shape[0]):
        for i in range(constants.shape[0]):
            fq[i, q_idx] = constants[i] * np.sum(np.multiply(rdf[i]-1, r_sum[q_idx]))
    neutron_FQ = np.sum(fq, axis=0)
    return neutron_FQ


def compute_structure_factor(
    element_pairs, 
    rdfs, 
    q_values, 
    composition,
    rho0,
    unique_elements,
    r_values, 
    deuteration_fraction
):
    """
    Compute the structure factor S(q).
    """
    # Update the neutron scattering length for Hydrogen based on the deuteration fraction
    neutron_scattering_lengths["H"] = neutron_scattering_lengths["H2"] * deuteration_fraction + neutron_scattering_lengths["H1"] * (1-deuteration_fraction)
    
    atomic_fraction = composition.get_atomic_fraction
    constants = np.zeros((len(element_pairs), 1), dtype=np.float64)

    for idx, (element1, element2) in enumerate(element_pairs):
        constants[idx] = (2 - int(element1 == element2)) * atomic_fraction(element1) * atomic_fraction(element2) * neutron_scattering_lengths[element1] * neutron_scattering_lengths[element2] * rho0 * 4 * np.pi

    rsum = calculate_rsum(q_values, r_values, r_values[1]-r_values[0])
    fq = calculate_neutron_FQ(q_values, rdfs, constants, rsum)

    # Sum over all elements in the structure
    denominator = np.sum([atomic_fraction(element) * neutron_scattering_lengths[element] for element in unique_elements])**2
    return fq / denominator


