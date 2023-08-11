"""
10.08.23
@tcnicholas
Scripts used to modify the RMC ACC structure to "fix" the carbonate and water
molecules to the rigid bodies.
"""

import pathlib

import numpy as np
from ase import Atoms
from ase.io import read
from pymatgen.io.cif import CifParser
from ase.calculators.lammps import Prism, convert



def read_structure(filePath):
    """
    Read CIF using Pymatgen.
    """
    parsed = CifParser( filePath,
                        occupancy_tolerance=100,
                        site_tolerance=0)
    s = parsed.get_structures(primitive=False)[0]
    s.merge_sites(tol=0.1, mode="delete")
    return s


def get_ix(structure, element):
    """
    Get indices of element in structure.
    """
    return np.array([i for i,z in enumerate(structure.species) if z.symbol == element])


def get_ca(structure, atomID, molID):
    """
    Get Ca positions and format in LAMMPS atom style.
    """
    atoms_lammps = []
    for ca in get_ix(structure, "Ca"):
        atoms_lammps.append([
            atomID,         # atom ID.
            molID,          # molecule ID.
            1,              # atom type.
            charges["Ca"],  # atom charge.
            *list(structure[ca].coords),
        ])

        atomID +=1
        molID += 1
    return atoms_lammps, atomID, molID


def get_co3(structure, atomID, molID, bondID, angleID, improperID, cutoff=1.6):
    """
    Get carbonate atom positions, bonds, and angles and format in LAMMPS style.
    """
    atoms_lammps = []
    bonds_lammps = []
    angles_lammps = []
    impropers_lammps = []

    c_ix = get_ix(structure, "C")
    c_frac = structure.frac_coords[c_ix]
    c_cart = structure.lattice.get_cartesian_coords(c_frac)

    o_ix = get_ix(structure, "O")
    o_frac = structure.frac_coords[o_ix]
    used_o = []

    ds = structure.lattice.get_all_distances(   structure.frac_coords[c_ix],
                                                structure.frac_coords[o_ix] )
    for i,c in enumerate(c_ix):
        
        # grab shortest three C-O bonds.
        bound_o = np.argpartition(ds[i], 3)[:3]

        # when getting the positions, need to make sure correct periodic image
        # is found for each. select carbon at default cartesian position and
        # then determine relevant image of bonded oxygens.
        c_pos = c_frac[i]
        o_pos = []
        for o in bound_o:
            _, img = structure.lattice.get_distance_and_image(c_pos, o_frac[o])
            o_pos.append(o_frac[o]+img)

        used_o += list(o_ix[bound_o])

        # convert to cartesian coordinates.
        o_pos = structure.lattice.get_cartesian_coords(o_pos)
        
        # ---------------------------------------------------------------------#
        # make carbonate molecule planar with r(C-O) = 1.284 Å and
        # theta(O-C-O) = 120*.
        temp_co3 = Atoms(   "CO3", cell=structure.lattice.matrix,
                            positions=np.vstack([c_cart[i], o_pos]))
        for j in [1,2]:
            temp_co3.set_distance(0, j, 1.284000000000000, fix=0) # fix the carbon in place.
            
        temp_co3.set_angle(1, 0, 2, 120.000000000000)
        v1 = temp_co3.get_distance(0,1,vector=True)
        v2 = temp_co3.get_distance(0,2,vector=True)
        v3 = -(v1+v2)
        v3 *= 1.28400000000000 / np.linalg.norm(v3)
        temp_co3.positions[3] = temp_co3.positions[0] + v3
        o_pos = temp_co3[1:].positions
        # ---------------------------------------------------------------------#

        atoms_lammps.append([
            atomID,             # atom ID.
            molID,              # molecule ID.
            2,                  # atom type.
            charges["C"],       # atom charge.
            *list(c_cart[i]),   # coordinates
        ])

        for ii,o in enumerate(o_pos, 1):
            atoms_lammps.append([
                atomID+ii,      # atom ID.
                molID,          # molecule ID.
                3,              # atom type.
                charges["O"],   # atom charge.
                *list(o),       # coordinates.
            ])

        molID += 1

        # also add topology information.
        bonds_lammps += [[bondID+iii, 1, atomID, atomID+iii+1] for iii in range(0,3)]
        bondID += 3

        angles_lammps += [
            [angleID, 1, atomID+1, atomID, atomID+2],
            [angleID+1, 1, atomID+1, atomID, atomID+3],
            [angleID+2, 1, atomID+2, atomID, atomID+3]]

        angleID += 3
        
        # temporarily add impropers to ensure the carbonate is flat.
        impropers_lammps += [
            [improperID, 1, atomID+1, atomID, atomID+2, atomID+3]]
            
        improperID += 1
        atomID += 4

    # then determine which oxygens have been left (assume they are water oxygen)
    remaining_o = list(set(o_ix) - set(used_o))

    return atoms_lammps, bonds_lammps, angles_lammps, impropers_lammps, atomID, molID, bondID, angleID, improperID, remaining_o


def get_h2o(structure, atomID, molID, bondID, angleID, remainingOxygenIX):
    """
    """

    atoms_lammps = []
    bonds_lammps = []
    angles_lammps = []
    
    o_frac = structure.frac_coords[remainingOxygenIX]
    o_cart = structure.lattice.get_cartesian_coords(o_frac)

    h_ix = get_ix(structure, "H")
    h_frac = structure.frac_coords[h_ix]

    ds = structure.lattice.get_all_distances(o_frac, h_frac)

    for i,o in enumerate(remainingOxygenIX):

        # grab shortest two O-H bonds.
        bound_h = np.argpartition(ds[i], 2)[:2]

        o_pos = o_frac[i]
        h_pos = []
        for h in bound_h:
            _, img = structure.lattice.get_distance_and_image(o_pos, h_frac[h])
            h_pos.append(h_frac[h]+img)
 
        # convert to cartesian coordinates.
        h_pos = structure.lattice.get_cartesian_coords(h_pos)
        
        # ---------------------------------------------------------------------#
        # sort out water molecule to have r(O-H) = 0.9572, theta(H-O-H) = 104.52.
        temp_h2o = Atoms(   "OH2", cell=structure.lattice.matrix,
                            positions=np.vstack([o_cart[i], h_pos]))
                            
        # fix the carbon in place.
        for j in [1,2]:
            temp_h2o.set_distance(0, j, 0.9572000000000, fix=0)
            
        temp_h2o.set_angle(1, 0, 2, 104.52000000000000)
        
        for j in [1,2]:
            temp_h2o.set_distance(0, j, 0.9572000000000, fix=0)
            
        h_pos = temp_h2o[1:].positions
        # ---------------------------------------------------------------------#

        atoms_lammps.append([
            atomID,             # atom ID.
            molID,              # molecule ID.
            4,                  # atom type.
            charges["Ow"],       # atom charge.
            *list(o_cart[i]),   # coordinates
        ])

        for ii,h in enumerate(h_pos, 1):
            atoms_lammps.append([
                atomID+ii,         # atom ID.
                molID,          # molecule ID.
                5,              # atom type.
                charges["H"],   # atom charge.
                *list(h),       # coordinates.
            ])

        molID += 1

        # also add topology information.
        bonds_lammps += [   [bondID, 2, atomID, atomID+1],
                            [bondID+1, 2, atomID, atomID+2]]
        bondID += 2

        angles_lammps += [[angleID, 2, atomID+1, atomID, atomID+2]]
        angleID += 1

        atomID += 3
    
    return atoms_lammps, bonds_lammps, angles_lammps, atomID, molID, bondID, angleID


def write_lammps_data(fileName, structure, atoms, bonds, angles,
            impropers, protons, units="metal"):
    """
    Gather LAMMPS data and write to file.
    """

    file_str = f"LAMMPS data file for RMC ACC Config.\n\n"
    
    file_str += f"{len(atoms)} atoms\n"
    file_str += f"{len(bonds)} bonds\n"
    file_str += f"{len(angles)} angles\n"
    file_str += "0 dihedrals\n"
    file_str += f"{len(impropers)} impropers\n\n"
    
    if protons:
        file_str += f"5 atom types\n"
        file_str += f"2 bond types\n"
        file_str += f"2 angle types\n"
    else:
        file_str += f"3 atom types\n"
        file_str += f"1 bond types\n"
        file_str += f"1 angle types\n\n"
        
    if impropers:
        file_str += f"1 improper types\n"
    file_str += "\n"

    # use ASE converter to get LAMMPS simulation box parameters.
    p = Prism(structure.lattice.matrix)
    xhi, yhi, zhi, xy, xz, yz = np.round(convert(p.get_lammps_prism(),
                                        "distance", "ASE", units),4)
    
    # simulation box size/parameters.
    # assume origin centred at (0, 0, 0).
    file_str += \
        f"0.0 {xhi:<15.10f} xlo xhi\n" + \
        f"0.0 {yhi:<15.10f} ylo yhi\n" + \
        f"0.0 {zhi:<15.10f} zlo zhi\n" + \
        f"{xy+0.0:<15.10f} {xz+0.0:<15.10f} {yz+0.0:<15.10f} xy xz yz\n\n"
    
    # also convert positions to appropriate LAMMPS coordinates.
    decor = np.array(atoms)
    pos = p.vector_to_lammps(decor[:,-3:], wrap=False)
    decor[:,-3:] = pos
    atoms = decor
    
    # Add atomic masses for each atom.
    file_str += "Masses\n\n"

    types = {1:"Ca", 2:"C", 3:"O", 4:"Ow", 5:"H"}
    if not protons:
        del types[4]
        del types[5]
    
    for n,atom in types.items():
        file_str += f"{n}\t{masses[atom]} # {atom}\n"

    file_str += "\nAtoms\n\n"
    for a in atoms:
        file_str += "\t{:>5.0f}\t{:>5.0f}\t{:>5.0f}\t{:>15.6f}\t{:>25.20f}\t{:>25.20f}\t{:>25.20f}\n".format(*a)
        
    file_str += "\nBonds\n\n"
    for b in bonds:
        file_str += "\t"+"\t".join((str(x) for x in b)) + "\n"
        
    file_str += "\nAngles\n\n"
    for ang in angles:
        file_str += "\t"+"\t".join((str(x) for x in ang)) + "\n"
    
    if impropers:
        file_str += "\nImpropers\n\n"
        for ang in impropers:
            file_str += "\t"+"\t".join((str(x) for x in ang)) + "\n"
        
    # write files.
    with open(pathlib.Path(fileName), "w") as f:
        f.write(file_str)
    
    
def extract_topology(dataFile):
    """
    Extract topology information from datafile (assume this doesn't change
    during the HRMC run).
    """

    with open(dataFile, "r") as f:
        lines = f.readlines()

    headers = ["Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]

    curr_header = None
    header = []
    data = {}
    while lines:

        l = lines.pop(0)
        if curr_header is not None: l = l.strip()
        if not l.strip(): continue
        
        if l.strip().split("#")[0].strip() in headers:
            curr_header = l.strip().split("#")[0].strip()
            data[curr_header] = []

        elif curr_header is None:
            header.append(l)

        elif l not in headers:
            data[curr_header].append(l)
    
    data["Atoms"] = [x.split() for x in data["Atoms"]]
    for x in data["Atoms"]:
        x[0] = int(x[0])
    data["Atoms"] = sorted(data["Atoms"], key=lambda x: x[0])
    for x in data["Atoms"]:
        x[0] = str(x[0])
    data["Atoms"] = [" ".join(x) for x in data["Atoms"]]
    
    return header, data


masses = {
    "C" : 12.011,
    "H" : 1.008,
    "Ca" : 40.078,
    "O" : 15.9994,
    "Ow" : 15.9994,
}


charges = {
    "C" : 1.123282, # a.u.
    "Ca" : 2.0, # a.u.
    "O" : -1.041094, # a.u.
    "Ow" : -1.04844, # a.u.
    "H" : 0.52422, # a.u.
}


def main():
    
    inname = 'rmc_acc_config.cif'
    outname = 'rmc_acc_config_fixed'
    
    acc_rmc = read_structure(inname)
    atoms_lammps, atomID, molID = get_ca(acc_rmc, 1, 1)
    atoms_lammps2, bonds_lammps, angles_lammps, impropers_lammps, atomID, molID, bondID, angleID, improperID, remaining_o = get_co3(acc_rmc, atomID, molID, 1, 1, 1, cutoff=1.6)
    atoms_lammps += atoms_lammps2
    
    try:
        atoms_lammps2, bonds_lammps2, angles_lammps2, atomID, molID, bondID, angleID = get_h2o(acc_rmc, atomID, molID, bondID, angleID, remaining_o)
        atoms_lammps += atoms_lammps2
        bonds_lammps += bonds_lammps2
        angles_lammps += angles_lammps2
        protons = True
    except:
        protons = False

    write_lammps_data(  f"{outname}.data", acc_rmc, atoms_lammps, bonds_lammps,
                        angles_lammps, impropers_lammps, protons, units="metal")
                        
    atoms = read(f"{outname}.data", format="lammps-data")
    atoms.write(f"{outname}.cif")
    
    # re-process to remove dihedrals etc.
    h, d = extract_topology(f"{outname}.data")
    
    fstr = ""
    for x in h:
        if "atoms" in x:
            fstr += f"\n{x}"
        elif "impropers" in x:
            fstr += f"0 impropers\n"
        elif "improper types" in x:
            fstr += f"0 improper types\n\n"
        else:
            fstr += x
    
    fstr += "\nMasses\n\n"
    fstr += "\n".join(d["Masses"])
    fstr += "\n\nAtoms\n\n"
    fstr += "\n".join(d["Atoms"])
    fstr += "\n\nBonds\n\n"
    fstr += "\n".join(d["Bonds"])
    fstr += "\n\nAngles\n\n"
    fstr += "\n".join(d["Angles"])
    
    with open(f"{outname}.data", "w") as f:
        f.write(fstr)
        

if __name__ == "__main__":
    main()
