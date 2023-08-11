# note (24.10.22)
The mass of the water oxygen (OW) was manually post-edited to the mass of lithium to ease the identification of water molecules vs carbonate molecules.

# note (24.10.22)
split_components used to store pymatgen Lattice object for the HRMC configuration. Also create CIFs with each component  (Ca, carbonate, water etc.) In order for this to work, a modification to ASE io LAMMPS readers was made to automatically detect element identities based on the mass.