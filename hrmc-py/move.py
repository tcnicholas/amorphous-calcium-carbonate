"""
22.12.21
@tcnicholas
Module for making new RMC moves on atoms and molecules.
"""


import numpy as np

import vector
from utils import ids2cids


rnd = np.random.default_rng()
moveTypes = np.array(["translate", "rotate"])


class Move:
    """
    A note on the image flags: the LAMMPS image flags inidcate which image of
    the simulation box (in each dimension) the atom would be in if its
    coordinates were unwrapped across periodic boundaries.
    """
    def __init__(self, box, lmp, a_ids):
        #  simulation box and LAMMPS instance.
        self._box = box
        self._lmp = lmp

        # atom IDs.
        self._a_ids = a_ids
        self._cids = ids2cids(a_ids)
        self._lencids = len(a_ids)

        # old positions.
        self._xc_o = None
        self._img_o = None
        self._xf_o = None

        # new position.
        self._xc_n = None
        self._img_n = None
        self._gather_position()


    @property
    def xc_o(self) -> np.ndarray:
        """
        Return old cartesian coordinates.
        """
        return self._xc_o


    @property
    def img_o(self) -> np.ndarray:
        """
        Return old image flags.
        """
        return self._img_o


    def propose(self, maxTrans: float, maxRot: float) -> None:
        """
        Propose move.

        :param maxTrans: maximum translation distance (Angstrom).
        :param maxRot: maximum rotation angle (degrees).
        """

        # decide if translation or rotation.
        if self._lencids == 1:
            move = "translate"
        else:
            move = rnd.choice(moveTypes)
        
        # get new coordinates.
        if move == "translate":
            self._translate(maxTrans)
        else:
            self._rotate(maxRot)
            
        # wrap coordinates.
        self._img_n = (self._xc_n // self._box.lengths).astype(np.int32)
        self._xc_n += (-self._img_n) * self._box.lengths

        # send position to LAMMPS.
        self._send_position()


    def reject(self) -> None:
        """
        Reset atom positions if reject move.
        """
        self._lmp.scatter_atoms("x", self._xc_o, ids=self._a_ids)
        self._lmp.scatter_atoms("image", self._img_o, ids=self._a_ids)
    
    
    def _translate(self, maxTrans: float) -> None:
        """
        Translate atom(s).

        :param maxTrans: maximum translation distance (Angstrom).
        """
        self._xc_n += vector.random_vector(rnd.random(), rnd.random()) * maxTrans * rnd.random()


    def _rotate(self, maxRot: float) -> None:
        """
        Rotate atoms.

        :param maxRot: maximum rotation angle (degrees).
        """
        # translate molecule to put the centroid at the origin.
        c = np.mean(self._xc_n, axis=0)
        self._xc_n -= c

        # check the random vector and molecule orientation are not aligned.
        rv = vector.random_vector(rnd.random(), rnd.random())
        theta = vector.deg2rad(rnd.random()* maxRot)
        self._xc_n = np.array([vector.rodrigues(x, rv, theta) for x in self._xc_n]) + c


    def _gather_position(self) -> None:
        """
        Extract current position and image of atoms from LAMMPS. 
        """

        # extract old cartesian coordinates.
        self._xc_o = self._lmp.gather_atoms("x", ids=self._a_ids)
        self._img_o = self._lmp.gather_atoms("image", ids=self._a_ids)

        # compute starting point for new cartesian coordinates by adding correct
        # image.
        self._xc_n = (self._img_o * self._box.lengths) + self._xc_o


    def _send_position(self) -> None:
        """
        Send new coordinate(s) and image(s) to LAMMPS.
        """
        self._lmp.scatter_atoms("x", self._xc_n, ids=self._a_ids)
        self._lmp.scatter_atoms("image", self._img_n, ids=self._a_ids)


