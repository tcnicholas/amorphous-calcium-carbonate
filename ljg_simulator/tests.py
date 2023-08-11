"""
28.06.23
@tcnicholas
Tests for ljg_simulator.
"""


import unittest

import numpy as np

from .simulation import cg_acc_potential


class Tests(unittest.TestCase):
    """
    A class for testing the ljg_simulator package.
    """


    def test_hard_sphere(self):
        """
        Test the Monte Carlo routine with only a hard sphere potential (i.e. no
        pair potential set). Stick to 1 core because no benefit to having 
        additional cores since no energies are computed by LAMMPS.
        """

        # define a small box.
        box = np.array([15, 15, 15])
    
        # setup the simulation.
        s = cg_acc_potential(
            cores=1, outDir=f"tests/hard_sphere", box=box, nCa=50
        )
        s.initialise_simulation(lj_cut_and_gauss=False)
        s.initialise_hard_sphere(3.2)
        s.run_monte_carlo(nmoves=1e4, max_trans=0.1, T=300, naccept_end=100)
        s.close()



if __name__ == "__main__":
    unittest.main()
