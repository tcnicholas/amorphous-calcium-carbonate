"""
27.06.23
@tcnicholas
All tasks for the project.
"""


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import pathlib
import subprocess
from typing import List, Dict, Tuple, Any

import pymatgen
import numpy as np
from ase import Atoms
from ase.io import read
import matplotlib.pyplot as plt

from scripts.g3 import G3InputFileGenerator, G3InputFileParameters, call_gnuplot
from scripts.molecules import calcium, carbonate, water, MoleculeProcessor
from scripts.plot import plot_rdf, _update_rdf_axes, RDF_SCATTER_KWARGS
from scripts.lammps import LammpsLogFile, make_lammps_dump_string
from scripts.neutron import compute_structure_factor
from scripts.rings import compute_edges, save_graph
from scripts.utils import sorted_directories, mkdir
from scripts.voronoi import compute_voronoi

from utils import retrieve_hrmc_data

from ljg_simulator.utils import load_mc_ljg_log


cif_names = {
    'monte_carlo': 'ca_mc.cif',
    'molecular_dynamics': 'ca_md.cif'
}


def gather_logs(method: str = 'monte_carlo') -> Dict:
    """
    This function loads log files from specified directory.

    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :return: dictionary of log files.
    """
    logs = {}
    for simulation in sorted_directories(f'simulations/ljg/{method}'):
        if simulation.stem == 'g3':
            continue
        if method == 'monte_carlo':
            logs[int(simulation.stem)] = load_mc_ljg_log(simulation/'README.txt')
        elif method == 'molecular_dynamics':
            logs[int(simulation.stem)] = LammpsLogFile(simulation/'log.txt')
    return logs


def plot_mc_energy(
    ax: plt.Axes,
    data: Dict[int, Tuple[List[str], np.ndarray]],
    colour: str = 'k',
    energy_baseline = None
):
    """
    Plot the energy from the Monte Carlo simulations.

    :param ax: The axis to plot on.
    :param data: The data to plot.
    :param colour: The colour to plot with.
    :return: None.
    """
    ix = data['headers'].index('curr_e')
    total_energy = data['data'][:,ix]
    energy_baseline = total_energy.min() if energy_baseline is None else energy_baseline
    ax.plot(
        data['data'][:,1]/1e6,
        (total_energy - energy_baseline),
        c = colour,
        lw=1
    )


def plot_md_energy(
    ax: plt.Axes,
    data: LammpsLogFile,
    colour: str = 'k',
    skipblocks: int = 0,
    energy_baseline = None
):
    """
    Plot the energy from the Molecular Dynamics simulations.

    :param ax: The axis to plot on.
    :param data: The data to plot.
    :param colour: The colour to plot with.
    :param skipblocks: The number of blocks to skip.
    :return: None.
    """
    steps = data.gather_property('Step', data_start=skipblocks)
    total_energy = data.gather_property('TotEng', data_start=skipblocks)
    energy_baseline = total_energy.min() if energy_baseline is None else energy_baseline
    ax.plot(
        steps * float(data.info['timestep']) / 1e3,
        (total_energy-energy_baseline)/1620,
        c = colour,
        lw=1
    )


def plot_experiment_structure_factor(
    ax: plt.Axes, 
    data: np.ndarray, 
    x_offset: float = 0.0, 
    y_offset: float = 0.0, 
    scale_factor: float = 1.0, 
    **kwargs: Dict[str, Any]
) -> None:
    """
    Plots the data on the given axis with provided offsets and scaling.

    :param ax: The axis to plot on.
    :param data: The data to plot.
    :param x_offset: The x-offset to apply to the data.
    :param y_offset: The y-offset to apply to the data.
    :param scale_factor: The scale factor to apply to the data.
    :param kwargs: Contains scatter_args.
    :return: None.
    """

    x_values = data[:, 0] + x_offset
    y_values = (data[:, 1] + y_offset) * scale_factor

    scatter_args = {
        'facecolors': 'none',
        'edgecolors': 'k', 
        'lw': 1, 
        'alpha': 1.0, 
        's': 50
    }
    scatter_args.update(kwargs)

    ax.scatter(x_values, y_values, **scatter_args)


def plot_structure_factor(
    axis: plt.Axes, 
    expt_data: np.ndarray, 
    composition: pymatgen.core.composition.Composition,
    rs: np.ndarray,
    element_pairs: List,
    rdfs: np.ndarray,
    rho0: float,
    unique_elements: List,
    **kwargs: Dict[str, Any]
) -> None:
    """
    Plot experimental data and computed structure factor on the given axis.

    :param axis: The axis to plot on.
    :param expt_data: The experimental data to plot.
    :param composition: The composition of the structure.
    :param rs: The r-values used to compute the RDF.
    :param kwargs: Contains q_range, deuteration_fraction, expt_args, line_args.
    :return: None.
    """
    q_range = kwargs.get('q_range', (0, 1, 0.1))
    deuteration_fraction = kwargs.get('deuteration_fraction', 0.0)
    expt_args = kwargs.get('expt_args', {})
    line_args = kwargs.get('line_args', {'c': 'r', 'lw': 1})

    plot_experiment_structure_factor(axis, expt_data, **expt_args)
    qs = np.arange(*q_range)
    sq = compute_structure_factor(element_pairs, rdfs, qs, composition, rho0,
        unique_elements, rs, deuteration_fraction)
    axis.plot(qs, sq, **line_args)


def compute_graphs(
    model: str = 'ljg',
    method: str = 'monte_carlo',
    cutoff = 5.2
):
    """
    In order to run the graph analysis on the coarse-grained models, we need to
    define the edges (with the correct periodic images) for a periodic graph.

    :param model: name of model (e.g. ljg, hrmc).
    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :param cutoff: cutoff distance for neighbours.
    :return: None.
    """

    if model == 'hrmc':

        # gather the lattice object.
        lattice_path = f'simulations/hrmc/structure/components/lattice.pickle'
        with open(lattice_path, 'rb') as f:
            lattice = pickle.load(f)

        # then gather the voronoi statistics for each frame.
        for simulation in sorted_directories(f'simulations/{model}/molecules'):

            with open(simulation/'ca.pickle', 'rb') as f:
                ca = pickle.load(f)

            pos_ca = np.array([c.get_frac_coords(lattice) for c in ca])
            atoms = Atoms(f'Ca{len(ca)}', cell=lattice.matrix)
            atoms.set_scaled_positions(pos_ca)
            edges = compute_edges(atoms, cutoff)
            save_graph(simulation/'rings', edges)

    else:

        for simulation in sorted_directories(f'simulations/{model}/{method}'):
            atoms = read(simulation/cif_names[method])
            edges = compute_edges(atoms, cutoff)
            save_graph(simulation/'rings', edges)


def compute_ring_statistics(
    model: str = 'ljg',
    method: str = 'monte_carlo',
):
    """
    Compute the ring statistics for the simulations.

    :param model: name of model (e.g. ljg, hrmc).
    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :return: None.
    """
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("scripts/rings.jl");

    if model == 'hrmc':
        for simulation in sorted_directories(f'simulations/hrmc/molecules'):
            Main.compute_rings(str(simulation/'rings'))

    else:
        for simulation in sorted_directories(f'simulations/{model}/{method}'):
            Main.compute_rings(str(simulation/'rings'))


def compute_voronoi_statistics(
    model: str = 'ljg',
    method: str = 'monte_carlo',
):
    """
    Gather the Voronoi statistics from the simulations.

    :param model: name of model (e.g. ljg, hrmc).
    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :return: dictionary of Voronoi statistics.
    """
    voronoi_stats = {}

    if model == 'hrmc':

        # gather the lattice object.
        lattice_path = f'simulations/hrmc/structure/components/lattice.pickle'
        with open(lattice_path, 'rb') as f:
            lattice = pickle.load(f)

        # then gather the voronoi statistics for each frame.
        for simulation in sorted_directories(f'simulations/{model}/molecules'):

            if simulation.stem == 'g3':
                continue

            with open(simulation/'ca.pickle', 'rb') as f:
                ca = pickle.load(f)

            pos_ca = np.array([c.get_frac_coords(lattice) for c in ca])
            atoms = Atoms(f'Ca{len(ca)}', cell=lattice.matrix)
            atoms.set_scaled_positions(pos_ca)
            voronoi_stats[int(simulation.stem)] = compute_voronoi(atoms)

        return voronoi_stats

    for simulation in sorted_directories(f'simulations/{model}/{method}'):

        if simulation.stem == 'g3':
            continue

        atoms = read(simulation/cif_names[method])
        voronoi_stats[int(simulation.stem)] = compute_voronoi(atoms)

    return voronoi_stats


def gather_radial_distribution_functions(
    model: str = 'ljg',
    method: str = 'monte_carlo',
) -> Dict:
    """
    Gather the radial distribution functions from the simulations.

    :param model: name of model (e.g. ljg, hrmc).
    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :return: array of radial distribution functions.
    """
    if model == 'hrmc':
        rdf = np.loadtxt('simulations/hrmc/rdfs/intra/Ca_Ca.txt')[:,[0,-1]]
        return {'r_values': rdf[:,0], 'rdfs': rdf[:,1]}

    rs = None
    rdfs = []
    for simulation in sorted_directories(f'simulations/{model}/{method}'):

        if simulation.stem == 'g3':
            continue

        this_rdf = np.loadtxt(simulation/'rdf.rdf', skiprows=4)

        if rs is None:
            rs = this_rdf[:,1]

        # check the same r values are used.
        assert np.allclose(rs, this_rdf[:,1])
        rdfs.append(this_rdf[:,2])

    return {'r_values': rs, 'rdfs': np.vstack(rdfs)}


def gather_ring_statistics(
    model: str = 'ljg',
    method: str = 'monte_carlo',
    cutoff = 5.2
) -> Dict:
    """
    Gather the ring statistics from the simulations.

    :param model: name of model (e.g. ljg, hrmc).
    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :return: array of ring statistics.
    """

    ring_statistics = {}

    if model == 'hrmc':
        
        for simulation in sorted_directories(f'simulations/hrmc/molecules'):

            if simulation.stem == 'g3':
                continue

            rings = np.loadtxt(
                simulation/f'rings/rings_statistics_rcut{cutoff:.2f}.txt', 
                skiprows=1, dtype=int
            )

            strong_rings = np.loadtxt(
                simulation/f'rings/strongrings_statistics_rcut{cutoff:.2f}.txt', 
                skiprows=1, dtype=int
            )

            ring_statistics[int(simulation.stem)] = {
                'rings': rings,
                'strong_rings': strong_rings
            }

        return ring_statistics

    for simulation in sorted_directories(f'simulations/{model}/{method}'):

        if simulation.stem == 'g3':
            continue

        rings = np.loadtxt(
            simulation/f'rings/rings_statistics_rcut{cutoff:.2f}.txt', 
            skiprows=1, dtype=int
        )

        strong_rings = np.loadtxt(
            simulation/f'rings/strongrings_statistics_rcut{cutoff:.2f}.txt', 
            skiprows=1, dtype=int
        )

        ring_statistics[int(simulation.stem)] = {
            'rings': rings,
            'strong_rings': strong_rings
        }

    return ring_statistics


def plot_averaged_hrmc_rdfs(
    axes: np.ndarray,
    plot_order: List[str],
    include_intramolecular: List[str] = [],
    rmin: float = 0.0,
    rmax: float = 10.0,
    data_path: str = 'simulations/hrmc/rdfs',
    **kwargs
):
    """
    Gather the average RDFs from the final 12 frames of the HRMC refinement.

    :param axes: axes to plot on.
    :param plot_order: order of the plots.
    :param include_intramolecular: pairs to include as intra-molecular.
    :param rmin: minimum distance to plot (Å).
    :param rmax: maximum distance to plot (Å).
    :param data_path: path to data.
    :param kwargs: keyword arguments for scatter plot.
    :return: None.
    """

    # file admin.
    path_to_data = pathlib.Path(data_path)

    for ax, pair in zip(axes.flatten(), plot_order):

        data = np.loadtxt(path_to_data / 'nointra' / f'{pair}.txt')
        x, y = data[:, 0], data[:, -1]
        ix = np.logical_and(x > rmin, x < rmax)

        scatter_kwargs = RDF_SCATTER_KWARGS.copy()
        scatter_kwargs.update(kwargs)
        ax.scatter(x[ix], y[ix], **scatter_kwargs)

        if pair in include_intramolecular:

            data = np.loadtxt(path_to_data / 'intra' / f'{pair}.txt')
            x, y = data[:, 0], data[:, -1]
            ix = np.logical_and(x > rmin, x < rmax)
            ax.plot(x[ix], y[ix], lw=1.5, zorder=-1, alpha=0.2, c='k')

        pair_label = "–".join(pair.split('_'))
        ax.text(1.0, 1.0, pair_label, ha='right', va='top', 
                transform=ax.transAxes, fontsize=7)

    _update_rdf_axes(axes)


def prepare_g3_files(
    dumpfilename: str = 'all_frames.dump',
    inputfilename: str = 'g3.in',
    resultfilename: str = 'g3_result.dat',
    model: str = 'ljg',
    method: str = 'monte_carlo',
    **kwargs
) -> None:
    """
    Gather one configuration from each method and generate the coordinates
    file for g3 computation. They take the LAMMPS dump format so we can average
    over all ensembles as with the other statistics.

    :param dumpfilename: name of dump file.
    :param inputfilename: name of input file.
    :param model: name of model (e.g. ljg, hrmc).
    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :return: None.
    """

    if model == 'hrmc':

        path2sim = f'simulations/hrmc/molecules'

        # gather the lattice object.
        lattice_path = f'simulations/hrmc/structure/components/lattice.pickle'
        with open(lattice_path, 'rb') as f:
            lattice = pickle.load(f)

        nconfigs = 0
        dump_string = ''
        for timestep, simulation in enumerate(sorted_directories(path2sim)):
            
            if simulation.stem == 'g3':
                continue

            with open(simulation/'ca.pickle', 'rb') as f:
                ca = pickle.load(f)

            pos_ca = np.array([c.get_frac_coords(lattice) for c in ca])
            atoms = Atoms(f'Ca{len(ca)}', cell=lattice.matrix)
            atoms.set_scaled_positions(pos_ca)

            dump_string += make_lammps_dump_string(atoms, timestep)
            nconfigs += 1

        outdir = mkdir(f'simulations/hrmc/molecules/g3')
        with open(outdir/dumpfilename, 'w') as f:
            f.write(dump_string)

        # generate input file.
        params = G3InputFileParameters(
            nConf=nconfigs,
            input = os.path.abspath(outdir/dumpfilename),
            output = os.path.abspath(outdir/resultfilename),
            **kwargs
        )
        generator = G3InputFileGenerator(params)
        generator.generate_file(outdir/inputfilename)
        return

    path2sim = f'simulations/{model}/{method}'

    nconfigs = 0
    dump_string = ''
    for timestep, simulation in enumerate(sorted_directories(path2sim)):

        if simulation.stem == 'g3':
            continue

        atoms = read(simulation/cif_names[method])
        dump_string += make_lammps_dump_string(atoms, timestep)
        nconfigs += 1

    outdir = mkdir(f'simulations/{model}/{method}/g3')
    with open(outdir/dumpfilename, 'w') as f:
        f.write(dump_string)

    # generate input file.
    params = G3InputFileParameters(
        nConf=nconfigs,
        input = os.path.abspath(outdir/dumpfilename),
        output = os.path.abspath(outdir/resultfilename),
        **kwargs
    )
    generator = G3InputFileGenerator(params)
    generator.generate_file(outdir/inputfilename)


def run_g3(
    method: str = 'monte_carlo',
    model: str = 'ljg',
) -> None:
    """
    Run the g3 code on the configurations generated by the simulations.

    :param method: name of method (e.g. monte_carlo, molecular_dynamics).
    :param model: name of model (e.g. ljg, hrmc).
    :return: None.

    Notes
    -----
    The g3 code is compiled from the source code in the scripts/g3 directory.
    This ensures the code is compiled with the correct compilers valid for the 
    current environment.

    Notes from original authors.
    ----------------------------
    The g3 code is written in C++ and is compiled with the following command:

        g++ -std=gnu++11 -O3 g3.cpp -o g3

    If not specified, the default values are used (see g3.h for values).

    Central atom is atom B and end atoms are A and C (see fig.1 in the main 
    article):

        "A mixed radial, angular, three-body distribution function as a tool for 
        local structure characterization: Application to single-component 
        structures"; Sergey V. Sukhomlinov and Martin H. Müser.

    There is a classical way of implementing calculation of any distribution 
    function. Break the distance into bins, then calculate how many 
    pairs/triplets etc. fall into this bin, then proceed to normalize, such that 
    the ideal gas gives one.

    Here we proceeded in a different way. Initially we do not make bins, but 
    rather place nrGrid (or naGrid) points onto an axis from 0 to LRcut. 
    Whenever we have a pair of atoms, the distance is calculated and the two 
    nearst grid points are identified (the same would hold for the cosine axis 
    when triplets are considered). This distance gives a weight to each of these 
    two grid points, which is proportional to negative distance to them. For 
    example, if the distance falls exactly in the middle between two nearest 
    grid points, it contributes 1/2 to each of the points. For triplets one
    would have not just a 1D segment, but rather a cuboid.

    The total g3 is calculated. At the end, integration over one of the
    distances from Rmin to Rmax is performed.

    The output format is compatible with gnuplot.
    """

    # compile the g3 code.
    path2g3 = pathlib.Path('scripts/g3')
    subprocess.run([
        'g++', '-std=gnu++11', '-O3', path2g3/'g3.cpp', '-o', path2g3/'g3'
    ])
    print('Program compiled successfully.')

    if model == 'hrmc':
        input_file = os.path.abspath(f'simulations/hrmc/molecules/g3/g3.in')
    else:
        input_file = os.path.abspath(f'simulations/{model}/{method}/g3/g3.in')

    # run the g3 code.
    g3_exec = os.path.abspath('scripts/g3/g3')
    subprocess.run([g3_exec, input_file])

    # remove the executable.
    os.remove(g3_exec)

    # then call gnuplot to plot the results.
    parent_dir = pathlib.Path(input_file).parent
    call_gnuplot(parent_dir/'g3_result.dat', parent_dir/'g3_result.png')
    print('Plot generated successfully.')


def retrieve_and_process_hrmc_data():
    """
    Retrieve and process the molecule data.
    """
    hrmc_data = retrieve_hrmc_data()
    processor = MoleculeProcessor(hrmc_data)
    results = processor.process_molecules()
    return results


if __name__ == '__main__':
    compute_ring_statistics(method='molecular_dynamics')
    compute_ring_statistics(model='hrmc')

    compute_graphs(model='hrmc')
    compute_graphs(method='monte_carlo')
    compute_graphs(method='molecular_dynamics')

    prepare_g3_files(model='hrmc')
    prepare_g3_files(method='monte_carlo')
    prepare_g3_files(method='molecular_dynamics')

    run_g3(model='hrmc')
    run_g3(method='monte_carlo')
    run_g3(method='molecular_dynamics')