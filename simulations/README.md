# Simulation data

This directory contains structure files and simulation trajectory data for the
HRMC, RMC, and LJG simulations.

### HRMC

- \<structure>

    - The final HRMC ACC structure CIF `HRMC_150M_proposed_moves.cif`.
    - The final HRMC ACC structure in LAMMPS data format `HRMC_150M_proposed_moves.data`
    - `<components>` contains separate CIFs/LAMMPS data files for different
    molecular components (e.g. a Ca only structure, a CO<sub>3</sub> only structure etc.).

- \<trajectory>

    - `log_tot.txt` is the HRMC log file containing information on the number
    and type of moves accepted and the energy and data costs.
    - `hrmc_trajectory.csv` contains the energies of each HRMC snapshot computed
    using the force field in LAMMPS.

- \<rdfs>

    The partial radial distribution functions for the final 12 snapshots
of the HRMC trajectory. These are divided into 'intra' and 'nointra', where 
intramolecular bonds are included and excluded in the calculations, respectively.

- \<molecules>

    Snapshots were captured every 100,000 proposed moves from the HRMC trajectory, 
    and the final 12 of these snapshots were selected for decomposition into 
    their molecular components (see `scripts/molecules.py` for details). Each 
    molecular component stores information on its local coordination environment.

    The computed ring statistics are also included for each structure snapshot.

- \<molecular_dynamics>
    
    - The final snapshot of the 1 ns MD trajectory starting from the HRMC 
    structure is given in CIF and LAMMPS data format (`hrmc_nvt_1ns.cif` and 
    `hrmc_md_nvt_1ns.data`, respectively).

    - The total scattering data for the final snapshot are included in the `<rdfs>`
    directory.

### LJG

For each LJG Monte Carlo and molecular dynamics simulation, the following files 
are included:

- `README.txt` trajectory data (moves performed, energies, etc.).
- The structures files (in both CIF and LAMMPS data format).
- The LAMMPS computed radial distribution function.
- Ring statistics data.
- g3 correlation function input (coordinates file).


### RMC

- \<structure> 
    - The original RMC ACC structure in CIF and LAMMPS data format.
    - A modified version of the RMC ACC structure with the molecular components
    "fixed" to the rigid body geometries.

- \<rdfs>

    The scattering data for the RMC model.

