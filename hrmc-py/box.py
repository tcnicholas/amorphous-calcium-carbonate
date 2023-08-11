import numpy as np
import utils
import pathlib
from monty.fractions import gcd, gcd_float
from numba import njit


class Composition:
    """
    A stripped down Composition class (c.f. Pymatgen's fancy one).
    """

    def __init__(self, sp_c):
        """
        sp_c: dict
            Chemical symbol : count.
        """
        self.species_counts = sp_c
        self._natoms = sum(sp_c.values())
    
    @property
    def natoms(self) -> int:
        return self._natoms
    
    @property
    def elements(self) -> list:
        return list(self.species_counts.keys())

    @property
    def formula(self) -> str:
        return " ".join(f"{e}{c}" for e,c in self.species_counts.items())

    def atomic_fraction(self, el):
        """ Get the atomic fraction of a given element. """
        return abs(self.species_counts[el]) / self._natoms
    
    @property
    def reduced_formula_and_factor(self):
        """ Compute reduced formula and return the factor. """
        factor = abs(gcd(*(int(i) for i in self.species_counts.values())))
        formula = (f"{s}{c/factor:.0f}" for s, c in self.species_counts.items())
        formula = " ".join(formula)
        return formula, factor


class Box:
    def __init__(self, lmp, atomTypes, name, dataFile):
        """ Store details about simulation box. """
        self.lmp = lmp
        self._name = name
        self._natoms = lmp.extract_global("natoms") # number of atoms.
        self._ntypes = lmp.extract_global("ntypes") # number of atom types.
        self._lengths = self._get_lengths(lmp)
        self._matrix = np.array([[self._lengths[0],0,0], [0,self._lengths[1],0], [0,0,self._lengths[2]]])
        self._inv = np.linalg.inv(self._matrix)
        self._elements = None
        self._elementPairs = None
        self._element_pair_ix = None
        self._atomTypes = atomTypes
        self._type_counts = {}
        self._typeGroups = self._sort_atom_types(atomTypes)
        self._compositon = self._get_atomic_fractions(lmp, atomTypes)
        self._molecules = self._extract_molecules(lmp)
        self._topologyData = self._extract_topology(dataFile)
    
    def get_fractional_coords(self, coords):
        return np.dot(coords, self._inv)

    def get_cartesian_coords(self, coords):
        return np.dot(coords, self._matrix)

    def write_cif(self, file_path: str):
        """
        Output current configuration in CIF format.
        """

        formula, factor = self.composition.reduced_formula_and_factor

        # create file string.
        file_str = f"data_{self._name}\n" \
        f"_chemical_formula_structural\t\t'{self.composition.formula}'\n"
        
        # gather cell data.
        cell_params = list(self.lengths) + [90.0, 90.0, 90.0]
        cell_volume = self.volume

        file_str += f"_cell_length_a\t\t\t{cell_params[0]:.5f}\n" \
        f"_cell_length_b\t\t\t{cell_params[1]:.5f}\n" \
        f"_cell_length_c\t\t\t{cell_params[2]:.5f}\n" \
        f"_cell_angle_alpha\t\t{cell_params[3]:.5f}\n" \
        f"_cell_angle_beta\t\t{cell_params[4]:.5f}\n" \
        f"_cell_angle_gamma\t\t{cell_params[5]:.5f}\n" \
        f"_cell_volume\t\t\t{cell_volume:.5f}\n" \
        f"_cell_formula_units_Z\t\t{factor}\n" \
        f"_symmetry_space_group_name_H-M\t'P 1'\n" \
        f"_symmetry_Int_Tables_number\t1\n" \
        "loop_\n" \
        "_symmetry_equiv_pos_site_id\n" \
        "_symmetry_equiv_pos_as_xyz\n" \
        "1 x,y,z\n"

        # gather all atom types (convert to atom labels) and coordinates.
        a_t = self.lmp.gather_atoms("type")
        symbols = [self._atomTypes[i] for i in a_t]
        xc = self.lmp.gather_atoms("x")
        x = [self.get_fractional_coords(i) for i in xc]

        atom_list = []
        for i, (s,c) in enumerate( zip(symbols, x) ):
            atom_list.append(f"{i+1:<5.0f} {s:>4} 1 "  + \
                            f"{c[0]:<8.5f} {c[1]:<8.5f} {c[2]:<8.5f} 1")

        # add to file string.
        file_str += "loop_\n" \
        "_atom_site_label\n" \
        "_atom_site_type_symbol\n" \
        "_atom_site_symmetry_multiplicity\n" \
        "_atom_site_fract_x\n" \
        "_atom_site_fract_y\n" \
        "_atom_site_fract_z\n" \
        "_atom_site_occupancy\n"

        file_str += "\n".join(atom_list) + "\n"

        # footer.
        file_str += f"#End of data_{self._name}\n\n"

        # write to file.
        with open(pathlib.Path(file_path), "w+") as f:
            f.write(file_str)
            
        del formula, factor, file_str, cell_params, cell_volume, a_t, symbols, xc, x, atom_list

    @property
    def natoms(self):
        return self._natoms
    
    @property
    def ntypes(self):
        return self._ntypes
    
    @property
    def typeGroups(self):
        return self._typeGroups

    @property
    def lengths(self):
        return self._lengths

    @property
    def elements(self):
        return self._elements

    @property
    def elementPairs(self):
        return self._elementPairs

    @property
    def element_pair_ix(self):
        return self._element_pair_ix

    @property
    def composition(self):
        return self._compositon

    @property
    def volume(self):
        return np.prod(self._lengths)

    @property
    def rho0(self):
        return self._natoms / np.prod(self._lengths)

    @property
    def molecules(self):
        return self._molecules

    @property
    def matrix(self):
        return self._matrix

    @property
    def inv(self):
        return self._inv
        
    @property
    def type_counts(self):
        return self._type_counts

    def _sort_atom_types(self, atomTypes):
        """ Create an {element_symbol : atom-type} dictionary. """

        assert self.ntypes == len(atomTypes), \
            f"Must specify the element for all ({self.ntypes}) atom types."

        # group by unique elements
        typeGroups = {e:[[],""] for e in sorted(set(atomTypes.values()))}

        # elements should be given in alphabetical order to allow internal
        # consistency. This allows the  use of the asterix range in LAMMPS.
        for t,e in atomTypes.items():
            typeGroups[e][0].append(t)

        # generate asterix-style strings too for lammmps rdf compute.
        for e in typeGroups.keys():
            if len(typeGroups[e][0]) > 1:
                typeGroups[e][1] += f"{typeGroups[e][0][0]}" + \
                    f"*{typeGroups[e][0][-1]}"
            else:
                typeGroups[e][1] += f"{typeGroups[e][0][0]}"

        # sort elements alphabetically.
        self._elements = tuple(sorted(typeGroups.keys()))
        self._elementPairs = utils.unique_pairs(self._elements)
        self._element_pair_ix = [ [typeGroups[x[0]][1],typeGroups[x[1]][1]] for x in self._elementPairs]
        return typeGroups


    def _get_lengths(self, lmp):
        """ Determine simulation box lengths. """
        min, max = np.array(lmp.extract_box()[0:2])
        return max - min


    def _get_atomic_fractions(self, lmp, atomTypes):
        """ Determine atomic fractions. """
        #oldLAMMPS u, c = np.unique(lmp.extract_atom("type"), return_counts=True)
        u, c = np.unique(lmp.gather_atoms("type"), return_counts=True)
        sp_c = {}
        for ix, count in zip(u,c):
            if atomTypes[ix] in sp_c:
                sp_c[atomTypes[ix]] += count
            else:
                sp_c[atomTypes[ix]] = count
            self._type_counts[ix] = count
        return Composition(sp_c)
    

    def _extract_molecules(self, lmp):
        """
        Extract atom IDs for each molecule ID.
        
        These are stored in a list of lists.
        """

        # extract molecule ids and atom ids.
        ids_m = lmp.gather_atoms("molecule")
        ids_a = lmp.gather_atoms("id")

        # gather atom ids for each unique molecule id.
        u_mol = np.unique(ids_m)
        return np.array(
            [ids_a[np.argwhere(mol==ids_m).flatten()] for mol in  u_mol],
            dtype=object)

        
    def _extract_topology(self, dataFile):
        """
        Extract topology information from datafile (assume this doesn't change
        during the HRMC run).
        """

        with open(dataFile, "r") as f:
            lines = f.readlines()

        headers = ["Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]

        curr_header = None
        header = []
        data = {}
        while lines:

            l = lines.pop(0)
            if curr_header is not None: l = l.strip()
            if not l: continue

            if l.strip() in headers: 
                curr_header = l
                if l != "Atoms\n":
                    data[l] = []

            if curr_header is None:
                header.append(l)

            elif curr_header is not None and curr_header != "Atoms\n" and l not in headers:
                data[curr_header].append(l)
        
        return header, data

    
    def write_data_file(self, fileName):
        """
        Custom-code to write the LAMMPS data file (because this sometimes did
        not work correctly using PyLAMMPSMPI), perhaps due to a weird connection
        between different cores.
        """

        # format header of data file.
        f = "".join(self._topologyData[0])
        
        # extract Atoms section for data-file.
        id_ = self.lmp.gather_atoms("id")
        mol_ = self.lmp.gather_atoms("molecule")
        ty_ = self.lmp.gather_atoms("type")
        q_ = self.lmp.gather_atoms("q")
        x_ = self.lmp.gather_atoms("x")
        img_ = self.lmp.gather_atoms("image")

        # add Atoms section.
        f += "Atoms\n\n"
        for i,m,ty,q,x,img in zip(id_, mol_, ty_, q_, x_, img_):
            f += "{:>8.0f} {:>8.0f} {:>8.0f} {:>12.6f} {:>20.12f} {:>20.12f} {:>20.12f} {:>3.0f} {:>3.0f} {:>3.0f}\n".format(i,m,ty,q,*list(x), *list(img))

        # add topology sections.
        for t,val in self._topologyData[1].items():
            f += f"\n{t}\n\n" + "\n".join(val) + "\n"

        with open(fileName, "w") as file:
            file.write(f)








