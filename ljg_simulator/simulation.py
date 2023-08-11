"""
28.06.22
@tcnicholas
LAMMPS input file for coarse-grained Ca-Ca potential in ACC.
"""

import copy
import pathlib
import argparse
import datetime
import numpy as np

from ase.io import read
from pylammpsmpi import LammpsLibrary
from lammps import LMP_STYLE_GLOBAL, LAMMPS_INT

from .utils import round_maxr, random_vector, eps2H


boltzmann = 8.617333262E-5 # eV K-1


def kT2eV(kT, T=300):
    return kT * 1.380649e-23 * T / 1.602176634e-19


lj_fit = {
    'lj_eps' : kT2eV(0.20557592812243825),
    'lj_sigma' : 3.84317734638228,
}


ljg_fit = {
    'lj_eps' : kT2eV(0.09759071),
    'lj_sigma' : 3.7584971993521488,
    
    'g_eps' : kT2eV(1.5350319437883675),
    'g_r0' : 5.991246456844763,
    'g_sigma' : 0.601172550524922,
}


# Lennard-Jones Gaussian fit with additional repulsive feature.
ljg2_fit = {
    'lj_eps' : kT2eV(0.3677540583801819),
    'lj_sigma' : 3.7584971993521488,
    
    'g_eps' : kT2eV(1.5350319437883675),
    'g_r0' : 5.991246456844763,
    'g_sigma' : 0.601172550524922,
    
    'g2_eps' : kT2eV(-1.2040527392972389),
    'g2_r0' : 5.919659952808235,
    'g2_sigma' : 0.9307156951487607,
}


class cg_acc_potential:
    
    
    def __init__(
        self,
        box=np.array([52.794504272175, 54.794296100667, 45.195295323908]),
        nCa=1620,
        cores=1,
        outDir="output"
    ):
    
        self._box = box
        self._nCa = nCa
        self._lmp = None
        self._cores = cores

        # admin.
        self._out = outDir
        pathlib.Path(self._out).mkdir(parents=True, exist_ok=True)
        self._log = pathlib.Path(self._out) / "README.txt"
        self._start_time = datetime.datetime.now()
        self._write2traj = None
        self._trajfile = None
        self._num_mc = 1
        
        header = f"Simulation began at: {self._start_time}\n\n"
        with open(self._log, "w") as f:
            f.write(header)
        print(header)

        self._log_box()
        
        # potential parameters.
        self._lj_param = None
        self._guass_param = None
        self._guass2_param = None
        
        # Monte Carlo.
        self._natoms = 0
        self._rnd = None
        
        # hard sphere.
        self._rcut = None
        self._lattice = None
        self._coords = None
        self._v = None
        self._violations = None
        
        # compression progress.
        self._ncompressed = 0
        self._compress_outdir = None
        self._compress_struct = None
        self._compress_rdf = None
    
    
    @property
    def lj_param(self):
        return self._lj_param
    
    
    @lj_param.setter
    def lj_param(self, param: list):
        """
        Set the Lennard-Jones parameters.
        
        Note in LAMMPS, the LJ sigma is defined as the zero-crossing distance
        for the potential, not the energy minimum at 2^(1/6)sigma.
        """
        assert len(param)==2, "Set [epsilon, sigma] for LJ parameters."
        self._lj_param = list(param)
        fstr = "Lennard-Jones parameters set:\n"
        fstr += f"\teps = {self._lj_param[0]:.8f} eV\n"
        fstr += f"\tsigma = {self._lj_param[1]:.8f} Å\n\n"
        with open(self._log, "a+") as f:
            f.write(fstr)
        print(fstr)
        
    
    @property
    def gauss_param(self):
        return self._gauss_param
    
    
    @property
    def gauss2_param(self):
        return self._gauss2_param
    
    
    @gauss_param.setter
    def gauss_param(self, param: list):
        """
        Set the Gaussian parameters [epsilon (eV), r0 (Å), sigma (Å)].
        """
        assert len(param)==3, "Set [epsilon, r0, sigma] for Gaussian parameters."
        self._gauss_param = list(param)
        eps_val = copy.copy(param[0])
        self._gauss_param[0] = eps2H(param[0], param[2])
        fstr = "Gaussian parameters set:\n"
        fstr += f"\teps = {eps_val:.8f} eV\n"
        fstr += f"\tH = {self._gauss_param[0]:.8f} eV # See LAMMPS documentation\n"
        fstr += f"\tr0 = {self._gauss_param[1]:.8f} Å\n"
        fstr += f"\tsigma = {self._gauss_param[2]:.8f} Å\n\n"
        with open(self._log, "a+") as f:
            f.write(fstr)
        print(fstr)
        
    
    @gauss2_param.setter
    def gauss2_param(self, param: list) -> None:
        """
        Set the Gaussian parameters [epsilon (eV), r0 (Å), sigma (Å)] for the
        second Gaussian.

        :param param: list of parameters [epsilon, r0, sigma].
        :return: None.
        """
        assert len(param)==3, "Set [epsilon, r0, sigma] for Gaussian parameters."
        self._gauss2_param = list(param)
        eps_val = copy.copy(param[0])
        self._gauss2_param[0] = eps2H(param[0], param[2])
        fstr = "Gaussian (II) parameters set:\n"
        fstr += f"\teps = {eps_val:.8f} eV\n"
        fstr += f"\tH = {self._gauss2_param[0]:.8f} eV # See LAMMPS documentation\n"
        fstr += f"\tr0 = {self._gauss2_param[1]:.8f} Å\n"
        fstr += f"\tsigma = {self._gauss2_param[2]:.8f} Å\n\n"
        with open(self._log, "a+") as f:
            f.write(fstr)
        print(fstr)

    
    def initialise_simulation(self,
        cutoff: float = 25,
        lj_cut_and_gauss: bool = False,
        lj: bool = False,
        ljg2: bool = False,
        positions: str = None,
        random_number: int = 42
    ) -> None:
        """
        Prepare the simulation box with the specified parameters.

        :param cutoff: cutoff distance for the potential.
        :param lj_cut_and_gauss: use a hybrid/overlay potential.
        :param lj: use a pure Lennard-Jones potential.
        :param ljg2: use a hybrid/overlay potential with two Gaussians.
        :param positions: path to file containing positions to add to the box.
        :param random_number: random number seed.

        Notes
        -----
        Cutoff is the value to taper the interaction to zero. Default value of 
        10 is the same value used in the aqueous calcium carbonate rigid 
        potential.
        """
        with open(self._log, "a+") as f:
            f.write(f'\n\n Initialising simulation with {self._cores} cores.')
    
        # create lammps instance.
        self._lmp = LammpsLibrary(cores=self._cores)
        self._lmp.command(f'log {self._out}/log.txt')
        
        with open(self._log, "a+") as f:
            f.write('\n\n LAMMPS started.')
        
        # build a box.
        self._lmp.command("region box block 0 {} 0 {} 0 {}".format(*list(self._box)))
        self._lmp.command("units metal")
        self._lmp.command("dimension 3")
        self._lmp.command("boundary p p p")
        self._lmp.command("atom_style full")
        self._lmp.command("create_box 1 box")

        # populate the box randomly with the specified number of calcium atoms.
        if positions is not None:
            coords = np.loadtxt(positions)
            print(f'Adding {coords.shape[0]} atoms from positions.')
            for coord in coords:
                self._lmp.command(
                    'create_atoms 1 single {} {} {}'.format(*coord.tolist())
                )
        else:
            self._lmp.command(f"create_atoms 1 random {self._nCa} {random_number} box")
        
        self._lmp.command(f"mass 1 40.078")

        # setup the potential.
        if lj_cut_and_gauss:
            assert self._lj_param is not None and self._gauss_param is not None, \
                "Must set LJ and Gaussian parameters."
            self._lmp.command(f"pair_style hybrid/overlay lj/cut {cutoff} gauss/cut {cutoff}")
            self._lmp.command("pair_coeff 1 1 lj/cut {} {}".format(*self._lj_param))
            self._lmp.command("pair_coeff 1 1 gauss/cut {} {} {}".format(*self._gauss_param)) # H, r0, sigma
        elif lj:
            assert self._lj_param is not None, 'must set LJ parameters.'
            self._lmp.command(f'pair_style lj/cut {cutoff}')
            self._lmp.command('pair_coeff 1 1 {} {}'.format(*self._lj_param))
        elif ljg2:
            assert self._lj_param is not None and self._gauss_param is not None \
             and self._gauss2_param is not None, \
                'Must set LJ, Gauss, and Gauss2 parameters.'
            self._lmp.command(f"pair_style hybrid/overlay lj/cut {cutoff} gauss/cut {cutoff} gauss/cut {cutoff}")
            self._lmp.command("pair_coeff 1 1 lj/cut {} {}".format(*self._lj_param))
            self._lmp.command("pair_coeff 1 1 gauss/cut 1 {} {} {}".format(*self._gauss_param)) # H, r0, sigma
            self._lmp.command("pair_coeff 1 1 gauss/cut 2 {} {} {}".format(*self._gauss2_param)) # H, r0, sigma
            
        self._lmp.command("run 1")
        
    
    def initialise_hard_sphere(self, rcut: float = 3.0) -> None:
        """
        Add a hard-sphere potential to the Monte Carlo routine.

        :param rcut: cutoff distance for the hard-sphere potential.
        :return: None.
        """
        
        # only import this locally because had compatability issues on ARC with
        # numpy/numba/mpi4py.
        from pymatgen.core import Lattice
        
        # store cutoff value.
        self._rcut = rcut
        
        # create a Pymatgen Lattice object.
        self._lattice = Lattice(np.diag(self._box))
        
        # extract all current coordinates and convert to fractional coordinates.
        # the 'gather_atoms()' function orders atoms by atom ID (important!).
        self._coords = self._lattice.get_fractional_coords(
                                            self._lmp.gather_atoms('x').copy())
        
        # setup a violations score to allow moves to 'get less bad'.
        d = self._lattice.get_all_distances(self._coords, self._coords)
        np.fill_diagonal(d, 100) # ignore self-distances.
        self._v = d - self._rcut
        self._v[self._v>0.0] = 0.0 # ignore all non-violations.
        self._violations = np.sum(self._v)
        
    
    def check_hard_sphere(self, atom_ix: int, atom_cart: np.ndarray(3)) -> bool:
        """
        Compute new hard sphere violations.

        :param atom_ix: index of atom to move.
        :param atom_cart: new cartesian coordinates of atom.
        :return: True if move is accepted, False otherwise.
        """
        
        # store old coordinates and then update coordinates array.
        old_coords = self._coords[atom_ix,:].copy()
        old_violations = self._v[atom_ix,:].copy()
        
        # compute new distances to atom.
        frac_coords = self._lattice.get_fractional_coords(atom_cart)
        d = self._lattice.get_all_distances(frac_coords, self._coords)[0]
        d[atom_ix] = 100
            
        # compute new violations to atom.
        self._v[atom_ix,:] = self._v[:,atom_ix] = d - self._rcut
        self._v[self._v>0.0] = 0.0
        new_v = np.sum(self._v)
        
        # decide if the move was good.
        if new_v >= self._violations:
            accept = True
            self._coords[atom_ix,:] = frac_coords
            self._violations = new_v
        else:
            accept = False
            self._v[atom_ix,:] = self._v[:,atom_ix] = old_violations
            
        return accept

    
    def add_vacuum_and_center(self, shell: float) -> None:
        """
        Possibility to add a vacuum around the simulation box of a given
        thickness. This removes the periodicity of the cell, which allows the 
        particles to settle at whichever density minimises the energy.

        :param shell: thickness of vacuum to add (Å).
        :return: None.

        Notes
        -----
        A fix is also applied to continually center the C.O.M. in the centre of
        the simulation box.
        """
        
        assert shell <= 20.0, 'LAMMPS loses atoms if increase in box size too '\
            'sudden. Build up in steps (i.e. in a "for" loop with shell <= 20).'
        
        _, (hx, hy, hz), xy, xz, yz, _, _ = self._lmp.extract_box()
        self._lmp.command(f'change_box all x final 0 {hx+shell} y final 0 {hy+shell} z final 0 {hz+shell} boundary f f f')
        self._lmp.command('run 1')
        self._lmp.command('fix 1 all recenter 0.5 0.5 0.5 units fraction')
        self._lmp.command('run 1')

        fstr = 'Added vacuum to simulation box\n'
        fstr += f'\tvacuum thickness = {shell:.3f} Å\n\n'
        self._append_to_log(fstr)
        
    
    def run_nvt(self, 
        steps: int = 10000000,
        timestep: float = 0.0005, 
        binwidth: float = 0.02
    ) -> None:
        """
        Perform an NVT-ensemble simulation with current box setup.

        :param steps: number of steps to run.
        :param timestep: timestep (ps).
        :param binwidth: binwidth for RDF (Å).
        :return: None.
        """
        
        # print out statistics.
        fstr = "Performing NVT-ensemble simulation.\n"
        fstr += f"\ttimestep = {timestep:.8f} ps\n"
        fstr += f"\tsteps = {steps:.0f}\n"
        fstr += f"\tlength = {steps*timestep*0.001:.2f} ns\n\n"
        self._append_to_log(fstr)
        
        self._lmp.command("thermo 1000")
        self._lmp.command("neighbor 2.0 bin")
        self._lmp.command("neigh_modify every 1 delay 0 check yes one 200000 page 2000000000")
        self._lmp.command("minimize 0.0 1.0e-8 100 1000")
        
        self._lmp.command(f"timestep {timestep}") # picoseconds.
        self._lmp.command("fix 1 all nvt temp 300.0 300.0 $(100.0*dt)")
        self._lmp.command("fix removeMomentum all momentum 1 linear 1 1 1")
        self._lmp.command(f"dump 1 all custom 1000 {self._out+'/ca.traj'} id mol type mass x y z")
        self._lmp.command("dump_modify 1 sort id")
        self._lmp.command(f"run {int(steps)}")
        
        # output RDF.
        rmax = round_maxr(np.amin(self._box)/2, binwidth)
        nbins = int(rmax // binwidth) + 1
        self._lmp.command(f"comm_modify cutoff {rmax+5}")
        self._lmp.command(f"compute myRDF all rdf {nbins} cutoff {rmax}")
        self._lmp.command(f"fix 2 all ave/time 1 1 1 c_myRDF[*] file {self._out+'/rdf.rdf'} mode vector")
        self._lmp.command("run 0 pre yes post no")
        self._lmp.command('uncompute myRDF')
        self._lmp.command('unfix 2')
        
        # also convert the lammps-data file into CIF.
        self._lmp.command(f"write_data {self._out+'/ca_md.data'} nocoeff nofix")
        a = read(self._out+'/ca_md.data', format='lammps-data')
        a.write(self._out+'/ca_md.cif')
        
    
    def track_mc(self):
        """
        Begin logging the Monte Carlo simulation progress.
        """
        fstr = f'Tracking simulation progress:\n'
        fstr += 'nmoves, naccept, nboltzmann, nhardspherereject, nreject | curr_e, hs_violations\n'
        fstr += len('nmoves, naccept, nboltzmann, nhardspherereject, nreject | curr_e, hs_violations') * '-' + '\n'
        self._append_to_log(fstr)
    
    
    def append_trajectory(self, ts):
        """
        Mimick the LAMMPS style trajectory files to watch the simulation
        progress.
        """

        # extract box parameters.
        (lx, ly, lz), (hx, hy, hz), xy, xz, yz, (px, py, pz), _ = self._lmp.extract_box()
        natoms = self._lmp.extract_global("natoms")
        x = self._lmp.gather_atoms("x")

        # write details to string.
        fstr = 'ITEM: TIMESTEP\n'
        fstr += f'{ts}\n'
        fstr += 'ITEM: NUMBER OF ATOMS\n'
        fstr += f'{natoms}\n'
        fstr += 'ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
        fstr += '{:>20.10f} {:>20.10f} {:>20.10f}\n'.format(lx, hx, xy)
        fstr += '{:>20.10f} {:>20.10f} {:>20.10f}\n'.format(ly, hy, xz)
        fstr += '{:>20.10f} {:>20.10f} {:>20.10f}\n'.format(lz, hz, yz)
        fstr += 'ITEM: ATOMS id type mass x y z\n'

        # add all atom positions.
        for i in range(x.shape[0]):
            fstr += '{:>6.0f} {:>6.0f} {:>6.3f} {:>10.5f} {:>10.5f} {:>10.5f}\n'.format(i+1, 1, 40.078, *x[i,:].tolist())
        
        # append to trajectory file.
        with open(self._trajfile, 'a+') as f:
            f.write(fstr)

        
    def compress_box(self, new_box: list):
        """
        Isotropically compress the box during a Monte Carlo routine.
        """
        
        # write the previous simulation progress values to the log to start a
        # new dataframe for the new box size. also write out data.
        filename = self._compress_struct/f'compress_{self._ncompressed}.data'
        filename_cif = self._compress_struct/f'compress_{self._ncompressed}.cif'
        
        self._lmp.command(f'write_data {filename} nocoeff nofix')
        a = read(filename, format='lammps-data')
        a.write(filename_cif)
        
        # create new log header.
        if self._ncompressed > 0:
            fstr = 'Simulation box change:\n'
            fstr += '\tPrevious box = {:.5f} {:.5f} {:.5f}\n'.format(*self._box.tolist())
            fstr += '\tNew box = {:.5f} {:.5f} {:.5f}\n\n'.format(*list(new_box))
            self._box = np.array(new_box)
            self._append_to_log(fstr)
        
        self._lmp.command(f'change_box all ' \
            f'x final 0 {new_box[0]} ' \
            f'y final 0 {new_box[1]} ' \
            f'z final 0 {new_box[2]} ' \
            'remap')
            
        # start new log section.
        self._ncompressed += 1

    
    def write_data_file(self, outdir, filename):
        """
        Write the current configuration to folder specified.
        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        datafile = outdir/(filename+'.data')
        cif = outdir/(filename+'.cif')
        self._lmp.command(f'write_data {datafile} nocoeff nofix')
        a = read(datafile, format='lammps-data')
        a.write(cif)
        
    
    def run_monte_carlo(self,
        nmoves: int = 1e6, 
        max_trans: float = 0.1, 
        T: float = 300, 
        naccept_end: int = 1e5,
        write_to_log: int = 1e4,
        binwidth: float = 0.02, 
        write_compress: bool = False, 
        write_temp: bool = False,
        write_traj: int = None
    ):
        """
        Perform a Monte Carlo simulation on the current configuration.

        :param nmoves: number of moves to perform.
        :
        """

        # define the simulation temperature.
        kT = boltzmann * T
        
        # setup random number generator and run an initial energy calculation.
        self._rnd = np.random.default_rng()
        self._lmp.command("neighbor 2.0 bin")
        self._lmp.command(
            "neigh_modify every 1 delay 0 check yes one 200000 page 200000000"
        )
        self._lmp.command("run 1")
        self._lmp.command("variable e equal pe")
        self._lmp.command("variable elast equal $e")
        self._lmp.command("thermo_style custom step v_elast pe")
        self._natoms = self._lmp.extract_global("natoms")

        # compute intial configuration energy.
        estart = self._lmp.extract_compute(
            "thermo_pe", LMP_STYLE_GLOBAL,LAMMPS_INT) / self._natoms
        elast = estart
        naccept = 0
        nboltzmann = 0
        nreject = 0
        nhardsphere_reject = 0
        
        # write Monte Carlo stats to log file.
        fstr = 'Performing Monte-Carlo simulation.\n'
        fstr += f'\tMax moves = {nmoves:.0f}\n'
        fstr += f'\tMax accepted moves = {naccept_end}\n'
        fstr += f'\tTemperature = {T} K\n'
        fstr += f'\tMax translation = {max_trans} Å\n'
        fstr += f'\tEnergy(start) = {estart:.5f} eV\n'
        if self._rcut is not None:
            fstr += f'\tHard-sphere cutoff = {self._rcut} Å\n'
            fstr += f'\tHard-sphere quality = {self._violations:.4f} (ideal = 0)\n'
        fstr += f'\tNumber of cores = {self._cores}\n'
        fstr += '\n\n'
        self._append_to_log(fstr)
        
        if write_compress:
            self._append_to_log(f'#Compression {self._ncompressed}#\n')
        elif write_temp:
            pathlib.Path(self._out+'/temp_step/rdfs').mkdir(
                parents=True, exist_ok=True)
            pathlib.Path(self._out+'/temp_step/atoms').mkdir(
                parents=True, exist_ok=True)

        if write_traj is not None:
            self._write2traj = int(write_traj)
            self._trajfile = pathlib.Path(self._out) / f"mc_{self._num_mc}.traj"
            self.append_trajectory(0)

        # then run specified number of moves.
        self.track_mc()
        self._log_mc(0,naccept,nboltzmann,nhardsphere_reject,nreject,elast)
        for m in range(int(nmoves)):
            
            # choose a random atom.
            iatom = self._rnd.integers(1,self._natoms+1)
            xi = self._lmp.gather_atoms("x", ids=[iatom])[0]
            xi_new = xi + max_trans * random_vector(
                 self._rnd.random(), self._rnd.random())
            
            # first check it passes the hard-sphere conditions (if requested).
            if self._rcut is not None and not self.check_hard_sphere(iatom-1,xi_new):
                nreject += 1
                nhardsphere_reject += 1
                if (m+1) % write_to_log == 0:
                    self._log_mc(m+1,naccept,nboltzmann,nhardsphere_reject,nreject,elast)
                continue

            # then determine quality of the move.
            self._lmp.scatter_atoms("x", xi_new, ids=[iatom])
            self._lmp.command("run 1")
            e = self._lmp.extract_compute("thermo_pe", LMP_STYLE_GLOBAL, LAMMPS_INT) / self._natoms

            if e <= elast:
                elast = e
                self._lmp.command("variable elast equal $e")
                naccept += 1
            elif self._rnd.random() <= np.exp(self._natoms*(elast-e)/kT):
                elast = e
                self._lmp.command("variable elast equal $e")
                naccept += 1
                nboltzmann += 1
            else:
                self._lmp.scatter_atoms("x", xi, ids=[iatom])
                nreject += 1
            
            # break loop if desired number of accepted moves met.
            if naccept==naccept_end: 
                break
            if (m+1) % write_to_log == 0: 
                self._log_mc(
                    m+1, naccept, nboltzmann, nhardsphere_reject, nreject, elast
                )
            if write_traj is not None and (m+1) % write_traj == 0: 
                self.append_trajectory(m+1)
        
        # write the final values to the log.
        self._log_mc(m+1,naccept,nboltzmann,nhardsphere_reject,nreject,elast)
        
        # hence output configuration and write Monte Carlo finishing stats to
        # log file. need a way of determining convergence.
        fstr = '\n\nMonte Carlo simulation results:\n'
        fstr += f'\tTotal accepted moves = {naccept:.0f}\n'
        fstr += f'\tE_start, E_last, E_diff = {estart:.5f}, {elast:.5f}, {elast-estart:.5f} eV\n'
        if self._rcut is not None:
            fstr += f'\tHard-sphere quality = {self._violations:.4f}\n'
        fstr += '\n'
        self._append_to_log(fstr)
        
        # output RDF.
        rmax = round_maxr(np.amin(self._box)/2, binwidth)
        nbins = int(rmax // binwidth) + 1
        
        # determine where to put the file.
        if write_compress:
            rdf_out = self._compress_rdf/f'rdf_{self._ncompressed}.rdf'
            data_out = self._out+'/ca_mc.data'
            cif_out = self._out+'/ca_mc.cif'
        elif write_temp:
            rdf_out = self._out+f'/temp_step/rdfs/rdf_{T:.1f}K.rdf'
            data_out = self._out+f'/temp_step/atoms/ca_mc_{T:.1f}K.data'
            cif_out = self._out+f'/temp_step/atoms/ca_mc_{T:.1f}K.cif'
        else:
            rdf_out = self._out+'/rdf.rdf'
            data_out = self._out+'/ca_mc.data'
            cif_out = self._out+'/ca_mc.cif'

        self._lmp.command(f"comm_modify cutoff {rmax+5}")
        self._lmp.command(f"compute myRDF all rdf {nbins} cutoff {rmax}")
        self._lmp.command(f"fix 2 all ave/time 1 1 1 c_myRDF[*] file {rdf_out} mode vector")
        self._lmp.command("run 0 pre yes post no")
        self._lmp.command('uncompute myRDF')
        self._lmp.command('unfix 2')
        
        # also convert the lammps-data file into CIF.
        self._lmp.command(f"write_data {data_out} nocoeff nofix")
        a = read(data_out, format='lammps-data')
        a.write(cif_out)
        
        # update the counter for number of simulations run.
        self._num_mc += 1


    def lammps2cif(self):
        """
        Use ASE file parser to convert the LAMMPS data output to CIF.
        """
        atoms = read(self._out+'/ca.data', format='lammps-data')
        atoms.write(f'{self._out}/ca.cif')
        
    
    def close(self):
        """
        Finalise simulation.
        """
        try:
            self._lmp.close()
            finish_time = datetime.datetime.now()
            footer = f"Simulation ended at: {finish_time}\n"
            footer += f'Elapsed time = {finish_time-self._start_time}\n\n'
            self._append_to_log(footer)
        except:
            pass

    
    def _log_box(self):
        """
        Write box details to log file, lest I forgot what I am doing.
        """
        fstr = "Box details:\n"
        fstr += '\tcell parameters = {:.5f} {:.5f} {:.5f}\n'.format(*self._box.tolist())
        fstr += f'\tcell volume = {np.product(self._box):.5f} Å^3\n'
        fstr += f'\tCa number density = {self._nCa/np.product(self._box):.8f}\n\n'
        self._append_to_log(fstr)
        
    
    def _log_mc(self, nmoves, naccept, nboltzmann, nhardsphere_reject, nreject, curr_e):
        """
        Write current Monte Carlo stats to log file.
        """
        if self._rcut is None:
            hs = '-'
        else:
            hs = f'{self._violations:.4f}'
            
        fstr = '{:>10.0f} {:>10.0f} {:>10.0f} {:>10.0f} {:>10.0f} | {:>12.8f} {:>12}\n'.format(
                nmoves, naccept, nboltzmann, nhardsphere_reject, nreject, curr_e, hs)
        self._append_to_log(fstr)
        
    
    def _append_to_log(self, fstr):
        """
        Append a given string (fstr) to the log file.
        """
        with open(self._log, "a") as f:
            f.write(fstr)
        print(fstr)
        
    
    def prepare_compression_directories(self):
        """
        Create subdirectories to keep compression runs tidier.
        """
        self._compress_outdir = pathlib.Path(self._out+f'/compress')
        self._compress_outdir.mkdir(parents=True, exist_ok=True)
        self._compress_struct = pathlib.Path(self._compress_outdir/'structures')
        self._compress_struct.mkdir(parents=True, exist_ok=True)
        self._compress_rdf = pathlib.Path(self._compress_outdir/'rdf')
        self._compress_rdf.mkdir(parents=True, exist_ok=True)
        


def ljg2_mc(args):
    """
    Test the Monte Carlo routine, parameterised with the Lennard-Jones and
    Guassian values extracted from the inverted PDF->Potential. This time use
    the Lennard-Jones Gaussian(squared) potential to fit the high-R values.
    """
    
    if not args.outdir: args.outdir = 'monte_carlo/ljg2'
    if args.trajectory is not None: args.trajectory = int(float(args.trajectory))
    
    # ACC box details
    box = np.array([52.794504272175, 54.794296100667, 45.195295323908])
    
    s = cg_acc_potential(cores=args.ncores, outDir=args.outdir, box=box)
    s.lj_param = [ljg2_fit['lj_eps'], ljg2_fit['lj_sigma']] # eps (eV); sigma (Å).
    s.gauss_param = [ljg2_fit['g_eps'], ljg2_fit['g_r0'], ljg2_fit['g_sigma']] # eps (eV); r0 (Å); sigma (Å).
    s.gauss2_param = [ljg2_fit['g2_eps'], ljg2_fit['g2_r0'], ljg2_fit['g2_sigma']] # eps (eV); r0 (Å); sigma (Å).
    s.initialise_simulation(ljg2=True, positions=args.positions)
    s.run_monte_carlo(nmoves=int(args.nmoves), max_trans=0.1, T=300,
                    naccept_end=int(args.naccept), write_traj=args.trajectory)
    s.close()
    
    
def ljg2_nvt(args):
    """
    Test the Monte Carlo routine, parameterised with the Lennard-Jones and
    Guassian values extracted from the inverted PDF->Potential. This time use
    the Lennard-Jones Gaussian(squared) potential to fit the high-R values.
    """
    
    if not args.outdir: args.outdir = 'monte_carlo/ljg2'
    if args.trajectory is not None: args.trajectory = int(float(args.trajectory))
    
    # ACC box details
    box = np.array([52.794504272175, 54.794296100667, 45.195295323908])
    
    s = cg_acc_potential(cores=args.ncores, outDir=args.outdir, box=box)
    s.lj_param = [ljg2_fit['lj_eps'], ljg2_fit['lj_sigma']] # eps (eV); sigma (Å).
    s.gauss_param = [ljg2_fit['g_eps'], ljg2_fit['g_r0'], ljg2_fit['g_sigma']] # eps (eV); r0 (Å); sigma (Å).
    s.gauss2_param = [ljg2_fit['g2_eps'], ljg2_fit['g2_r0'], ljg2_fit['g2_sigma']] # eps (eV); r0 (Å); sigma (Å).
    s.initialise_simulation(ljg2=True, positions=args.positions, random_number=args.random)
    print('Initialisesd')
    s.run_nvt(steps=int(2e6)) # 1ns run with 0.0005 ps timestep.
    s.close()


def ljg2_mc_slowcool(args):
    """
    Test the Monte Carlo routine, parameterised with the Lennard-Jones and
    Guassian values extracted from the inverted PDF->Potential. This time use
    the Lennard-Jones Gaussian(squared) potential to fit the high-R values.
    """
    
    if not args.outdir: args.outdir = 'monte_carlo/ljg2'
    if args.trajectory is not None: args.trajectory = int(float(args.trajectory))
    
    # ACC box details
    box = np.array([52.794504272175, 54.794296100667, 45.195295323908])
    
    s = cg_acc_potential(cores=args.ncores, outDir=args.outdir, box=box)
    s.lj_param = [ljg2_fit['lj_eps'], ljg2_fit['lj_sigma']] # eps (eV); sigma (Å).
    s.gauss_param = [-ljg2_fit['g_eps'], ljg2_fit['g_r0'], ljg2_fit['g_sigma']] # eps (eV); r0 (Å); sigma (Å).
    s.gauss2_param = [-ljg2_fit['g2_eps'], ljg2_fit['g2_r0'], ljg2_fit['g2_sigma']] # eps (eV); r0 (Å); sigma (Å).
    s.initialise_simulation(ljg2=True, positions=args.positions)

    # expand the box to create a vacuum around the current configuration. this
    # doesn't seem to work if I do it all in one go (leads to lost atoms if the
    # change in box size is too big too fast).
    for _ in range(5):
        s.add_vacuum_and_center(20) # Å.

    # then run simulation by slowly cooling down.
    for T in np.flip(np.arange(25,325,25)).tolist() + [20, 15, 10, 5]:
        s.run_monte_carlo(nmoves=int(args.nmoves), max_trans=0.1, T=T, write_temp=True,
                        naccept_end=int(args.naccept), write_traj=args.trajectory)
    s.close()
    

def hard_sphere_mc(args):
    """
    Test the Monte Carlo routine with only a hard sphere potential (i.e. no
    pair potential set). Stick to 1 core because no benefit to having additional
    cores since no energies are computed by LAMMPS.
    """
    
    # set hard cutoff value.
    rcut = 3.8 # Å
    
    # ACC box details
    box = np.array([52.794504272175, 54.794296100667, 45.195295323908])
    
    # run simulation.
    s = cg_acc_potential( cores=1, outDir=f"monte_carlo/hard_sphere/{rcut}",
            box=box, nCa=1620)
    s.initialise_simulation(lj_cut_and_gauss=False)
    s.initialise_hard_sphere(rcut)
    s.run_monte_carlo(nmoves=5e6, max_trans=0.1, T=300, naccept_end=2e6)
    s.close()
    

def main():
    """
    Decide which type of potential to run.
    """
    
    description = '''Perform MD or MC simulations on boxes of Ca atoms using
coarse-grained (effective) potentials.'''
    
    parser = argparse.ArgumentParser(
                        prog = 'CG_CA_Ponteials',
                        description = description)
    
    # give option for simulation types and additional associated arguements.
    parser.add_argument('-p', '--potential', choices=['lj','ljg','hs','ljg+c', 
                        'ljg2', 'ljg2+sc', 'ljg2_nvt'],
                        required=True, help='Choose type of potential. The ' \
                        'addition of "+c" requests a compression run.')
    parser.add_argument('-rcut', '--rcut', type=float, default=3.2)
    parser.add_argument('-o', '--outdir', required=True,
                        help='Give path to output directory.')
    parser.add_argument('-n', '--nmoves', default=int(5e6), type=float,
                        help='Total number of moves.')
    parser.add_argument('-a', '--naccept', default=int(2e6), type=float,
                        help='Number of accepted moves to declare convergence.')
    parser.add_argument('-c', '--ncores', type=int, default=10,
                        help='Number of cores to use.')
    parser.add_argument('-t', '--trajectory', default=None,
                        help='Number of steps after which to write trajectory step.')
    parser.add_argument('-pos', '--positions', default=None,
                        help='Path to starting cartesian coordinates.')
    parser.add_argument('-temp', '--temperature', default=300, type=float,
                        help='Simulation temperature.')
    parser.add_argument('-r', '--random', default=42, type=int,
                    help='Random number seed.')
    
    
    # hence pass any arguments passed to the script.
    args = parser.parse_args()
    
    # determine which kind of potential to run and send to the appropriate f(x).
    if args.potential=='hs':
        hard_sphere_mc(args)
    elif args.potential=='ljg2':
        ljg2_mc(args)
    elif args.potential=='ljg2+sc':
        ljg2_mc_slowcool(args)
    elif args.potential=='ljg2_nvt':
        ljg2_nvt(args)
    else:
        print("No simulation type specified!")
    
    
if __name__ == "__main__":
    main()
