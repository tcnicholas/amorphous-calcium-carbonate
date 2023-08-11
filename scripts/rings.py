"""
27.06.23
@tcnicholas
Computing ring statistics of model. This script is used to generate input files
which can be used in combination with the Julia script (scripts/rings.jl) to
compute ring statistics. For usage, see the tasks_si.py script in the main
directory.
"""


from typing import Dict

import ase
import numpy as np
from ase.io import read
from pymatgen.core.lattice import Lattice

from .utils import mkdir


def compute_edges(
    atoms: ase.Atoms, 
    cutoff: float
) -> Dict:
    """
    Compute edges of a graph from an ASE atoms object.

    :param atoms: ASE atoms object.
    :param cutoff: cutoff distance for neighbours.
    :return: dictionary with frac_coords, edges, and cutoff.
    """
    frac = np.array(atoms.get_scaled_positions())
    lattice = Lattice(atoms.cell)
    edges = []
    for ca1 in range(len(atoms)):
        for ca2 in range(len(atoms)):
            if ca2<=ca1:
                continue
            d, i = lattice.get_distance_and_image(frac[ca1], frac[ca2])
            if d <= cutoff:
                edges.append([ca1+1, ca2+1, i.tolist()])
    return {'frac_coords': frac, 'edges': edges, 'cutoff': cutoff}


def save_graph(
    outdir: str,
    data: Dict
) -> None:
    """
    Save edges and images of a graph to (separate) files.

    :param outdir: directory to save files to.
    :param data: dictionary with frac_coords, edges, and cutoff.
    :return: None.
    """
    outdir = mkdir(outdir)
    np.savetxt(
        outdir/f'edges_rcut{data["cutoff"]:.2f}.txt',
        [x[:2] for x in data['edges']],
        fmt='%.0f'
    )
    np.savetxt(
        outdir/f'images_rcut{data["cutoff"]:.2f}.txt',
        [x[-1] for x in data['edges']],
        fmt='%.0f'
    )
    np.savetxt(
        outdir/'frac.txt', 
        data['frac_coords'], 
        fmt='%.8f'
    )