"""
27.07.23
@tcnicholas
Computing Voronoi statistics of model.
"""


from typing import Dict

import ase
import freud


def compute_voronoi(
    atoms: ase.Atoms,
) -> Dict:
    """
    Compute the Voronoi statistics of an ASE atoms object using freud.
    """

    # grab the box and positions.
    atoms.wrap()
    box = freud.box.Box.from_matrix(atoms.get_cell().array)
    pos = atoms.get_positions()

    # compute Voronoi statistics.
    voronoi = freud.locality.Voronoi()
    voronoi.compute(system=(box, pos))
    ns = voronoi.nlist.neighbor_counts
    return {'volumes': voronoi.volumes, 'neighbours': ns}