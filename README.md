# Research data for "Geometrically-frustrated interactions drive structural complexity in amorphous calcium carbonate"

<div align="center">

> **[Geometrically-frustrated interactions drive structural complexity in amorphous calcium carbonate](https://arxiv.org/abs/2303.06178)**\
> _[Thomas C. Nicholas](https://twitter.com/thomascnicholas), Adam E. Stones, Adam Patel, F. Marc Michel, Richard J. Reeder, Dirk G. A. L. Aarts, [Volker L. Deringer](http://deringer.chem.ox.ac.uk), and [Andrew L. Goodwin](https://goodwingroupox.uk/)_

</div>

---

## Repository overview

- Simulation scripts

    - **HRMC simulation scripts**. The custom scripts used to run the hybrid 
    reverse Monte Carlo (HRMC) refinements. The program is under continual 
    development, but the program version used for the work is archived 
    [here](hrmc-py).

    - **LJG simulator scripts**. The custom scripts used to run the LJG 
    simulations are included [here](ljg_simulator).

- Simulation data

    - **Hybrid reverse Monte Carlo model**. The final model is included in both 
    LAMMPS and CIF data formats [here](simulations/hrmc/structure/).

    - **LJG coarse-grained models**. We ran both 
    [Monte Carlo](simulations/ljg/monte_carlo/) (MC) and [molecular
    dynamics](simulations/ljg/molecular_dynamics) (MD) simulations using our 
    LJG-parameterised effective Ca–Ca interaction potential. We ran 12 
    independent simulations for each method. Final configurations are provided
    in LAMMPS and CIF data formats. The radial distribution functions computed
    in LAMMPS are also provided as `"rdf.rdf"` for each configuration.

- Analysis scripts


- Plotting scripts

    - **`main.ipynb`** is a Jupyter Notebook containing the plotting scripts for
    generating the raw graphic data for the main text. Specifically, this 
    includes:

        1. Comparing computed X-ray total scattering functions with experimental
        value for the RMC, HRMC, and MD methods (Fig. 1a).

        2. Comparing the relative energies and quality of fit-to-data metrics
        for the RMC, HRMC, and MD methods (Fig. 1b).

        3. Coordination number histograms for calcium and carbonate environments
        in the HRMC configuration (Fig. 2a and b).

        4. Comparison of the Ca–Ca partial radial distribution functions for the
        HRMC, RMC, and LJG with the Fourier transform of the X-ray total
        scattering function (Fig. 2e).

        5. The extracted effective Ca–Ca interaction potential from the
        test-particle insertion algorithm for the ACC configuration, together 
        with the LJG parameterisation (Fig. 3a).

        6. Orientational correlation function for CO<sub>3</sub> and 
        H<sub>2</sub>O species (Fig. 3a).

    - **`si.ipynb`** is a Jupyter Notebook containing the plotting scripts for
    generating the raw graphic data for the supporting information. 
    Specifically, this includes:

        1. Evolution of system properties during HRMC refinement and MD 
        simulation (Fig. S1).

        2. Evolution of coarse-grained simulation ensemble energies (Fig. S2).

        3. HRMC partial radial distribution functions, averaged over the final
        12 frames of the refinement trajectory (Fig. S3). 

        4. Key bond-length and bond-angle distributions from the final HRMC
        structure (Fig. S4).

        5. Computed neutron total scattering measurements, as compared with three
        independent neutron total scattering measurements of ACC (Fig. S5).

        6. RDFs and particle cluster sizes pertinent to investigating the role of
        water in ACC (Fig. S6).

        7. Comparing properties of the Ca-only (coarse-grained) configurations
        from HRMC with those produced by LJG Monte Carlo and molecular dyanmics 
        simulations. We compare distributions of Voronoi cell volumes, Voronoi
        cell face counts (neighbours), (strong) ring sizes, and a Ca-triplet
        three-body correlation function (Fig. S7–9).

---

## Citing this work

You can cite this work using the following BibTeX reference.

```bibtex
@misc{Nicholas_2023,
      title={Geometrically-frustrated interactions drive structural complexity in amorphous calcium carbonate}, 
      author={Thomas C. Nicholas and Adam E. Stones and Adam Patel and F. Marc Michel and Richard J. Reeder and Dirk G. A. L. Aarts and Volker L. Deringer and Andrew L. Goodwin},
      year={2023},
      eprint={2303.06178},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

---

## License <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a>

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.


