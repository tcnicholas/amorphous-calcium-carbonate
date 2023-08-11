"""
07.02.23
@tcnicholas
Identify all molecules in ACC configurations.
"""

import sys
import pickle
import pathlib
import itertools
from typing import Dict, List

import numpy as np
from lammps import lammps

from .utils import angle, ids2cids, cdata2pydata


hrmc_lattice = '../hrmc/structure/components/lattice.pickle'
traj_data = '../hrmc/trajectory/data-files'
outdir = '../hrmc/molecules'


atom_types = {1:"Ca", 2:"C", 3:"O", 4:"Ow", 5:"H"}


def main():
    """ Search configuration for all molecules. """
    
    # load lattice object.
    with open(hrmc_lattice, 'rb') as f:
        lattice = pickle.load(f)
        
    # iterate through all molecules and save to file.
    this_data_file = f'{traj_data}/{int(sys.argv[1])}.data'
    lmp = start_lammps(this_data_file)
    ca, co3, h2o = get_species(lmp, lattice, atom_types)
    
    # save them.
    out = pathlib.Path(f'{outdir}/{int(sys.argv[1])}')
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'ca.pickle', 'wb') as f:
        pickle.dump(ca, f)
    with open(out/'co3.pickle', 'wb') as f:
        pickle.dump(co3, f)
    with open(out/'h2o.pickle', 'wb') as f:
        pickle.dump(h2o, f)
    
    
def get_species(lmp, lattice, atom_types, ca_oc=2.8, ca_ow=2.8, ow_ow=3.75,
    oc_hw=2.3):
    """
    Extract all Ca, CO3, and H2O species from the structure.
    """
    
    # get all molecule IDs and identify CO3 as those with 4 occurences (i.e.
    # there are 4 atoms with that mol-ID).
    atomIDs = lmp.numpy.extract_atom("id")
    molID_atoms = lmp.numpy.extract_atom("molecule")
    molID, counts = np.unique(molID_atoms, return_counts=True)

    # extract molecule IDs for each species according to the number of atoms in
    # the molecule (e.g. Ca:1, CO3: 4, H2O: 3)
    ca_mol = [mol for mol,count in zip(molID, counts) if count==1]
    co3_mol = [mol for mol,count in zip(molID, counts) if count==4]
    h2o_mol = [mol for mol,count in zip(molID, counts) if count==3]
    
    # extract coordinates for each species.
    all_calciums = []
    for ca in ca_mol:
        aIDs = atomIDs[np.argwhere(molID_atoms==ca).flatten()]
        all_calciums.append(calcium(lmp, lattice, atom_types, aIDs))

    all_carbonates = []
    for co3 in co3_mol:
        aIDs = atomIDs[np.argwhere(molID_atoms==co3).flatten()]
        all_carbonates.append(carbonate(lmp, lattice, atom_types, aIDs))

    all_waters = []
    for h2o in h2o_mol:
        aIDs = atomIDs[np.argwhere(molID_atoms==h2o).flatten()]
        all_waters.append(water(lmp, lattice, atom_types, aIDs))

    # then determine neighbours for each species. first consider Ca-O
    # coordination.
    print(">>> Determining Ca connectivity...")
    for ca in all_calciums:
        ca_frac = ca.get_frac_coords(lattice)
        
        # find bound waters.
        for h2o in all_waters:
            o = h2o.get_oxygen_frac_coords(lattice)
            d = lattice.get_all_distances(ca_frac, o)
            if np.any(d<=ca_ow):

                # add the IDs to both water and calcium.
                ca.add_water(h2o.id)
                h2o.add_calcium(ca.id)

                # need to add the number of bound oxygens.
                num_o = np.count_nonzero(d<=ca_ow)
                ca.add_water_oxygen(num_o)
        
        # find bound carbonates.
        for co3 in all_carbonates:
            o = co3.get_oxygen_frac_coords(lattice)
            d = lattice.get_all_distances(ca_frac, o)
            if np.any(d<=ca_oc):

                # add the molecule IDs to both carbonate and calcium.
                ca.add_carbonate(co3.id)
                co3.add_calcium(ca.id)

                # need to add the number of bound oxygens.
                num_o = np.count_nonzero(d<=ca_oc)
                ca.add_carbonate_oxygen(num_o)

                # also record the dentisity of the interaction.
                co3.add_denticity(num_o)
                ca.add_carbonate_dentisity(num_o)
    
    # next consider carbonate interactions with water. By inspection, the
    # observation that there exists CaCO3 rich channels percolated by H2O should
    # be reflected in the analysis.
    print(">>> Determining CO3 connectivity...")
    for co3 in all_carbonates:

        o = co3.get_oxygen_frac_coords(lattice)
        for h2o in all_waters:

            h = h2o.get_proton_frac_coords(lattice)
            #ow = h2o.get_oxygen_frac_coords(lattice)
            d = lattice.get_all_distances(o, h)
            if np.any(d<=oc_hw):

                # add the molecule IDs to both carbonate and water.
                co3.add_water(h2o.id)
                h2o.add_carbonate(co3.id)
    
    print(">>> Determining H2O connectivity...")
    for i,h2o_i in enumerate(all_waters):

        o_i = h2o_i.get_oxygen_frac_coords(lattice)
        #h_i = h2o_i.get_proton_frac_coords(lattice)

        for j,h2o_j in enumerate(all_waters):

            # don't repeat pairs (or consider self-interaction).
            if i > j:
                
                o_j = h2o_j.get_oxygen_frac_coords(lattice)
                #h_j = h2o_j.get_proton_frac_coords(lattice)

                #d_oh = lattice.get_all_distances(o_i, h_j)
                #d_ho = lattice.get_all_distances(h_i, o_j)
                d_oo = lattice.get_all_distances(o_i, o_j)

                if np.any(d_oo<=ow_ow): # or np.any(d_ho<=ow_h):

                    # add molecule IDs to both water molecules.
                    h2o_i.add_water(h2o_j.id)
                    h2o_j.add_water(h2o_i.id)

    # now reset the molecule class variable counters.
    print(">>> Resetting molecule IDs...")
    calcium.reset()
    carbonate.reset()
    water.reset()

    return all_calciums, all_carbonates, all_waters
    

def start_lammps(path, special_bonds=False):
    """
    Initialise a LAMMPS instance for a given data file. Assumes metal units (as
    used during the HRMC simulations for ACC). Set a large comm_mofidy cutoff to
    allow for Voronoi tesselations to be computed for very sparsely distributed
    atoms (e.g. only considering the Ca atoms).

    Args
    ----
    path: str
        Path to LAMMPS data file.
    """
    lmp = lammps(cmdargs=["-screen", "none"])
    cmd_str = f"""clear
    units metal
    boundary p p p
    atom_style full
    box tilt large
    read_data {path}
    comm_modify mode single cutoff 30.0
    """
    if special_bonds:
        cmd_str += "special_bonds lj/coul 1.0e-50 1.0e-50 1.0e-50 angle no dihedral no\n"

    lmp.commands_string(cmd_str)
    return lmp


class molecule:
    """ Base class for storing molecule information. """

    def __init__(self, lmp, lattice, atom_types, aIDs):
        """
        Args
        ----
        lmp: lammps.lammps object.
            The LAMMPS instance.

        lattice: pymatgen.core.Lattice
            Pymatgen Lattice object for computing distance properties and
            properly unwrapping coordinates.

        aIDs: list(int)
            Atom IDs for this CO3 molecule.
        """
        self.aIDs = aIDs
        self._c_aIDs = ids2cids(aIDs)
        self._x = self.unwrap_coords(lmp, lattice)
        self._atypes = self.sort_atom_types(lmp, atom_types)
        self._c_aIDs = None # in order to make pickle-able.

    
    def unwrap_coords(self, lmp, lattice):
        """
        Unwrap the coordinates using the LAMMPS coordinates and image for a
        given set of aIDs.
        """

        # gather coordinates and images.
        xc = cdata2pydata(
            # name, type, count, ndata, ids
            lmp.gather_atoms_subset("x", 1, 3, len(self._c_aIDs), self._c_aIDs),
            3)

        img = cdata2pydata(
            # name, type, count, ndata, ids
            lmp.gather_atoms_subset("image", 0, 3, len(self._c_aIDs),
            self._c_aIDs), 3)

        # get correct periodic image.
        return lattice.get_cartesian_coords(lattice.get_fractional_coords(xc)+img)

    
    def sort_atom_types(self, lmp, atom_types):
        """
        Determine which atom is the carbon atom (LAMMPS will not necessarily
        preserve the atom ID order).
        """
        atypes = cdata2pydata(
            # name, type, count, ndata, ids
            lmp.gather_atoms_subset("type", 0, 1, len(self._c_aIDs),
            self._c_aIDs), 1)
        return np.array([atom_types[a] for a in atypes],dtype=str)

    
    @property
    def x(self):
        return self._x
        

class carbonate(molecule):
    """ Class for storing co3 molecule coordinates. """

    num = 1

    def __init__(self, lmp, lattice, atom_types, aIDs):
        """
        Args
        ----
        lmp: lammps.lammps object.
            The LAMMPS instance.

        lattice: pymatgen.core.Lattice
            Pymatgen Lattice object for computing distance properties and
            properly unwrapping coordinates.

        aIDs: list(int)
            Atom IDs for this CO3 molecule.
        """
        super().__init__(lmp, lattice, atom_types, aIDs)
        self._id = self.num
        self.update()
        self._bound_ca = []
        self._denticity = []
        self._bound_h2o = []

    
    @classmethod
    def update(cls):
        cls.num += 1

    
    @classmethod
    def reset(cls):
        cls.num = 1

    
    @property
    def id(self):
        return self._id

    
    @property
    def normal(self):
        """
        Compute a vector normal to the plane that all three oxygens lie in.
        """
        o_ix = np.argwhere(self._atypes=="O").flatten()
        o_pos = self._x[o_ix]
        n = np.cross(o_pos[1,:]-o_pos[0,:], o_pos[2,:]-o_pos[0,:])
        return n / np.linalg.norm(n)

    
    @property
    def carbon(self):
        """
        Return coordinates of the carbon atom to be used as the "center" of the
        molecule when computing the distance between carbonates.
        """
        return self._x[np.argwhere(self._atypes=="C").flatten()[0]]


    @property
    def oxygen_cartesian_positions(self):
        """
        Return coordinates of the carbon atom to be used as the "center" of the
        molecule when computing the distance between carbonates.
        """
        return self._x[np.argwhere(self._atypes=="O").flatten()]

    
    def get_oxygen_frac_coords(self, lattice):
        """
        Return the fractional coordinates of the oxygen atom for this carbonate
        molecule.

        Args
        ----
        lattice: Pymatgen.core.Lattice
        """
        return lattice.get_fractional_coords(self.oxygen_cartesian_positions)

    
    def get_carbon_frac_coords(self, lattice):
        """
        Return the fractional coordinates of the carbon atom for this carbonate
        molecule.

        Args
        ----
        lattice: Pymatgen.core.Lattice
        """
        return lattice.get_fractional_coords(self.carbon)


    def add_calcium(self, ca_id):
        self._bound_ca.append(ca_id)

    
    def add_water(self, h2o_id):
        self._bound_h2o.append(h2o_id)

    
    def add_denticity(self, val):
        self._denticity.append(val)
        

class water(molecule):

    num = 1

    def __init__(self, lmp, lattice, atom_types, aIDs):
        """
        Args
        ----
        lmp: lammps.lammps object.
            The LAMMPS instance.

        lattice: pymatgen.core.Lattice
            Pymatgen Lattice object for computing distance properties and
            properly unwrapping coordinates.

        aIDs: list(int)
            Atom IDs for this CO3 molecule.
        """
        super().__init__(lmp, lattice, atom_types, aIDs)
        self._id = self.num
        self.update()
        self._bound_ca = []
        self._bound_co3 = []
        self._bound_h2o = []

    
    @classmethod
    def update(cls):
        cls.num += 1

    
    @classmethod
    def reset(cls):
        cls.num = 1


    @property
    def id(self):
        return self._id

    
    @property
    def oxygen_cartesian_position(self):
        """
        Return coordinates of the carbon atom to be used as the "center" of the
        molecule when computing the distance between carbonates.
        """
        return self._x[np.argwhere(self._atypes=="Ow").flatten()[0]]

    
    @property
    def protons_cartesian_positions(self):
        """
        Return coordinates of the carbon atom to be used as the "center" of the
        molecule when computing the distance between carbonates.
        """
        return self._x[np.argwhere(self._atypes=="H").flatten()]


    def get_oxygen_frac_coords(self, lattice):
        """
        Return the fractional coordinates of the oxygen atom for this water
        molecule.

        Args
        ----
        lattice: Pymatgen.core.Lattice
        """
        return lattice.get_fractional_coords(self.oxygen_cartesian_position)
    

    def get_proton_frac_coords(self, lattice):
        """
        Return the fractional coordinates of the hydrogen atoms for this water
        molecule.

        Args
        ----
        lattice: Pymatgen.core.Lattice
        """
        return lattice.get_fractional_coords(self.protons_cartesian_positions)

    
    def add_calcium(self, ca_id):
        self._bound_ca.append(ca_id)

    
    def add_carbonate(self, co3_id):
        self._bound_co3.append(co3_id)


    def add_water(self, h2o_id):
        self._bound_h2o.append(h2o_id)



class calcium(molecule):
    """
    A boring class which is perhaps overboard using a molecule base class but
    useful to make use of the same functionality.
    """

    num = 1

    def __init__(self, lmp, lattice, atom_types, aIDs):
        """
        Args
        ----
        lmp: lammps.lammps object.
            The LAMMPS instance.

        lattice: pymatgen.core.Lattice
            Pymatgen Lattice object for computing distance properties and
            properly unwrapping coordinates.

        aIDs: list(int)
            Atom IDs for this CO3 molecule.
        """
        super().__init__(lmp, lattice, atom_types, aIDs)
        self._id = self.num
        self.update()
        self._bound_waters = []
        self._bound_carbonates = []
        self._bound_carbonates_dentisity = []
        self._bound_water_o = 0
        self._bound_carbonate_o = 0


    @classmethod
    def update(cls):
        cls.num += 1

    
    @classmethod
    def reset(cls):
        cls.num = 1

    
    @property
    def id(self):
        return self._id

    
    @property
    def waters(self):
        return self._bound_waters
    

    @property
    def carbonates(self):
        return self._bound_carbonates


    @property
    def total_mol_cn(self):
        """ The total coordination number. """
        return len(self._bound_waters) + len(self._bound_carbonates)

    
    @property
    def total_o_cn(self):
        return self._bound_water_o + self._bound_carbonate_o

    
    @property
    def water_mol_cn(self):
        """ The number of bound water molecules. """
        return len(self._bound_waters)

    
    @property
    def carbonate_mol_cn(self):
        """ The number of bound carbonate molecules. """
        return len(self._bound_carbonates)

    
    @property
    def water_o_cn(self):
        return self._bound_water_o


    @property
    def carbonate_o_cn(self):
        return self._bound_carbonate_o

    
    def add_water(self, water_id):
        self._bound_waters.append(water_id)

    
    def add_carbonate(self, carbonate_id):
        self._bound_carbonates.append(carbonate_id)

    
    def add_carbonate_dentisity(self, val):
        self._bound_carbonates_dentisity.append(val)

    
    def add_carbonate_oxygen(self, num):
        self._bound_carbonate_o += num


    def add_water_oxygen(self, num):
        self._bound_water_o += num

    
    def get_frac_coords(self, lattice):
        """
        Return the fractional coordinates of this calcium.

        Args
        ----
        lattice: Pymatgen.core.Lattice
        """
        return lattice.get_fractional_coords(self._x)[0]
    

class MoleculeProcessor:
    """
    Class to process the coordination environments of each molecule in the HRMC 
    simulations.
    """
    def __init__(self, 
        hrmc_data_all: Dict,
        ca_oc: float = 2.8,
        ca_ow: float = 2.8,
    ) -> None:
        """
        Initialise the MoleculeProcessor class.

        :param hrmc_data_all: Custom dictionary of hrmc data containing the
            lattice, molecule data and molecule dictionary. The molecule data is 
            a dictionary of lists of each molecule type (e.g. calcium, 
            carbonates, and waters). The molecule dictionary is a dictionary of
            dictionaries of each molecule type (e.g. calcium, carbonates, and
            waters). The keys for the molecule dictionary are the molecule IDs.
        :param ca_oc: cut-off distance for Ca–O$_{\rm{C}}$.
        :param ca_ow: cut-off distance for Ca–O$_{\rm{W}}$.
        :return: None.
        """
        self.ca_oc = ca_oc
        self.ca_ow = ca_ow
        self.lattice = hrmc_data_all['lattice']
        self.molecule_data = hrmc_data_all['molecules_data']
        self.molecule_dict = hrmc_data_all['molecule_dict']


    def get_distances(self, 
        ca_list: List, 
        molecule_type: str, 
        molecule_key: str, 
        cut_off_distance: float,
    ) -> np.ndarray(1):
        """
        Calculate the distances between a list of Ca atoms and a list of a
        particular type of molecule.

        :param ca_list: list of Ca atoms.
        :param molecule_type: type of molecule (e.g. carbonates, waters).
        :param molecule_key: key for molecule dictionary.
        :param cut_off_distance: cut-off distance for the distance calculation.
        :return: array of distances.
        """
        distances = []
        for this_ca in ca_list:
            for molecule in getattr(this_ca, '_bound_{}'.format(molecule_type)):
                this_molecule = self.molecule_dict[molecule_key][molecule]
                ds = self.lattice.get_all_distances(
                    this_ca.get_frac_coords(self.lattice),
                    this_molecule.get_oxygen_frac_coords(self.lattice)
                )
                distances.append(ds[ds<=cut_off_distance].tolist())
        return np.concatenate(distances)
    

    def get_common_oxygen_count(self,
        ca1_frac: np.ndarray, 
        ca2_frac: np.ndarray, 
        o_frac: np.ndarray,
    ):
        """
        Determine the number of oxygen atoms that are within the cut-off
        distance of both Ca atoms.

        :param ca1_frac: fractional coordinates of first Ca atom.
        :param ca2_frac: fractional coordinates of second Ca atom.
        :param o_frac: fractional coordinates of oxygen atoms.
        :return: number of oxygen atoms that are within the cut-off distance of
        """
        d1 = self.lattice.get_all_distances(ca1_frac, o_frac) <= self.ca_oc
        d2 = self.lattice.get_all_distances(ca2_frac, o_frac) <= self.ca_oc
        return sum(v1 and v2 for v1, v2 in zip(d1.flatten(), d2.flatten()))
    

    def process_carbonates(self, co3_list):
        ca_ca_ds = {i: [] for i in range(4)}
        for this_co3 in co3_list:
            for ca1, ca2 in itertools.combinations(this_co3._bound_ca, r=2):
                ca1_frac = self.molecule_dict['ca'][ca1].get_frac_coords(self.lattice)
                ca2_frac = self.molecule_dict['ca'][ca2].get_frac_coords(self.lattice)
                d = self.lattice.get_all_distances(ca1_frac, ca2_frac).flatten()[0]
                common_o = self.get_common_oxygen_count(ca1_frac, ca2_frac, this_co3.get_oxygen_frac_coords(self.lattice))
                ca_ca_ds[common_o].append(d)
        return ca_ca_ds


    def get_ca_pairs(self):
        ca_pairs = set()
        for this_co3 in self.molecule_data['co3']:
            ca_pairs |= set(tuple(sorted(x)) for x in itertools.combinations(this_co3._bound_ca,r=2))
        return ca_pairs


    def refine_distances(self, ca_pairs):
        d_ca_ca = {i: [] for i in range(4)}
        for ix1, ix2 in ca_pairs:
            ca1 = self.molecule_dict['ca'][ix1]
            ca2 = self.molecule_dict['ca'][ix2]
            ca1_frac = ca1.get_frac_coords(self.lattice)
            ca2_frac = ca2.get_frac_coords(self.lattice)
            d = self.lattice.get_all_distances(ca1_frac, ca2_frac).flatten()[0]

            all_common = []
            shared_co3 = set(ca1._bound_carbonates) & set(ca2._bound_carbonates)
            for this_co3_ix in shared_co3:
                this_co3 = self.molecule_dict['co3'][this_co3_ix]
                common_o = self.get_common_oxygen_count(ca1_frac, ca2_frac, this_co3.get_oxygen_frac_coords(self.lattice))
                all_common.append(common_o)
            
            shortest_path = np.amax(all_common)
            d_ca_ca[shortest_path].append(d)
        return d_ca_ca
    

    def get_bound_oxygen(self, this_ca):
        ca_frac = this_ca.get_frac_coords(self.lattice) % 1.0
        ca_cart = self.lattice.get_cartesian_coords(ca_frac)

        bound_oxygen = []
        for this_co3 in this_ca._bound_carbonates:
            o_frac = self.molecule_dict['co3'][this_co3].get_oxygen_frac_coords(self.lattice)
            ds = self.lattice.get_all_distances(ca_frac, o_frac).ravel()
            ix = np.argwhere(ds <= self.ca_oc).ravel()

            for i in ix:
                d, img = self.lattice.get_distance_and_image(ca_frac, o_frac[i])
                o_cart = self.lattice.get_cartesian_coords(o_frac[i] + img)
                bound_oxygen.append([this_co3, o_cart - ca_cart])

        return bound_oxygen


    def calculate_angles(self, bound_oxygen):
        oc_ca_oc = []
        oc_ca_oc_same = []
        for v1, v2 in itertools.combinations(bound_oxygen, r=2):
            a = angle(v1[1], v2[1]) * 180 / np.pi
            if v1[0] == v2[0]:
                oc_ca_oc_same.append(a)
            else:
                oc_ca_oc.append(a)

        return oc_ca_oc, oc_ca_oc_same


    def get_bound_oxygen_and_co3(self, this_ca):
        ca_frac = this_ca.get_frac_coords(self.lattice) % 1.0
        ca_cart = self.lattice.get_cartesian_coords(ca_frac)

        bound_oxygen = []
        for this_h2o in this_ca._bound_waters:
            o_frac = self.molecule_dict['h2o'][this_h2o].get_oxygen_frac_coords(self.lattice)
            d, img = self.lattice.get_distance_and_image(ca_frac, o_frac)
            o_cart = self.lattice.get_cartesian_coords(o_frac + img)
            bound_oxygen.append([this_h2o, o_cart - ca_cart])

        bound_co3_oxygen = []
        for this_co3 in this_ca._bound_carbonates:
            o_frac = self.molecule_dict['co3'][this_co3].get_oxygen_frac_coords(self.lattice)
            ds = self.lattice.get_all_distances(ca_frac, o_frac).ravel()
            ix = np.argwhere(ds <= self.ca_oc).ravel()

            for i in ix:
                d, img = self.lattice.get_distance_and_image(ca_frac, o_frac[i])
                o_cart = self.lattice.get_cartesian_coords(o_frac[i] + img)
                bound_co3_oxygen.append([this_co3, o_cart - ca_cart])

        return bound_oxygen, bound_co3_oxygen
    

    def calculate_oxygen_angles(self, bound_oxygen, bound_co3_oxygen):
        ow_ca_ow = []
        ow_ca_oc = []

        for v1, v2 in itertools.combinations(bound_oxygen, r=2):
            a = angle(v1[1], v2[1]) * 180 / np.pi
            ow_ca_ow.append(a)

        for v1, v2 in itertools.product(bound_oxygen, bound_co3_oxygen):
            a = angle(v1[1], v2[1]) * 180 / np.pi
            ow_ca_oc.append(a)

        return ow_ca_ow, ow_ca_oc
    

    def process_molecules(self):
        d_ca_oc = self.get_distances(self.molecule_data['ca'], 'carbonates', 'co3', self.ca_oc)
        d_ca_ow = self.get_distances(self.molecule_data['ca'], 'waters', 'h2o', self.ca_ow)

        ca_pairs = self.get_ca_pairs()
        d_ca_ca = self.refine_distances(ca_pairs)

        oc_ca_oc = []
        oc_ca_oc_same = []
        ow_ca_ow = []
        ow_ca_oc = []

        for this_ca in self.molecule_data['ca']:
            bound_oxygen = self.get_bound_oxygen(this_ca)
            oc_ca_oc_, oc_ca_oc_same_ = self.calculate_angles(bound_oxygen)
            oc_ca_oc.extend(oc_ca_oc_)
            oc_ca_oc_same.extend(oc_ca_oc_same_)

            bound_oxygen, bound_co3_oxygen = self.get_bound_oxygen_and_co3(this_ca)
            ow_ca_ow_, ow_ca_oc_ = self.calculate_oxygen_angles(bound_oxygen, bound_co3_oxygen)
            ow_ca_ow.extend(ow_ca_ow_)
            ow_ca_oc.extend(ow_ca_oc_)


        results = {
            'distances': {
                'd_ca_oc': d_ca_oc,
                'd_ca_ow': d_ca_ow,
                'd_ca_ca': d_ca_ca,
            },

            'angles': {
                'oc_ca_o_different': oc_ca_oc,
                'oc_ca_oc_same': oc_ca_oc_same,
                'ow_ca_ow': ow_ca_ow,
                'ow_ca_oc': ow_ca_oc
            }
        }

        return results


if __name__ == "__main__":
    main()
