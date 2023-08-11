"""
18.12.21
@tcnicholas
Module for calculating total scattering.
"""


import copy
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from ase.data import atomic_numbers
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage.filters import gaussian_filter1d

import utils
from graphs import *


empCoeff = {
    "H" : (0.489918,20.6593,0.262003,7.74039,0.196767,49.5519,0.049879,2.20159,
    0.001305),

    "C" : (2.31,20.8439,1.02,10.2075,1.5886,0.5687,0.865,51.6512,0.2156),

    "N" : (12.2126,0.0057,3.1322,9.8933,2.0125,28.9975,1.1663,0.5826,-11.529),

    "O" : (3.0485,13.2771,2.2868,5.7011,1.5463,0.3239,0.867,32.9089,0.2508),

    "Ca" :(8.6266,10.4421,7.3873,0.6599,1.5899,85.7484,1.0211,178.437,1.3751),

    "Zn" :(14.0743,3.2655,7.0318,0.2333,5.1652,10.3163,2.41,58.7097,1.3041),
}


# Neutron coherent scattering lengths taken from: 
# https://www.ncnr.nist.gov/resources/n-lengths/
neutScatLength = {
    "Zn" : 5.680, # fm
    "C" : 6.6460, # fm
    "N" : 9.36, # fm
    "H1" : -3.7406, # fm
    "H2" : 6.671 # fm
}


@dataclass
class ScatteringData:
    """
    Class for storing experimental total-scattering data.

    :param data: experimental data.
    :param weight: weight of scattering.
    :param singleWeight: whether the weight is a single value or an array.
    :param mulQ: whether to multiply the Q-values by the scattering.
    :param ones: array of ones for weighting.
    """
    data: np.ndarray
    weight: np.ndarray
    singleWeight: bool
    mulQ: bool
    ones: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return self.data[:,0]
    
    @property
    def y(self) -> np.ndarray:
        return self.data[:,1]


class TotalScattering:
    """
    Class for calculating total scattering from a radial distribution function.
    """
    def __init__(self, 
        boxLengths: np.ndarray, 
        rMin: float = 0, 
        rMax: float = None, 
        binWidth: float = 0.02, 
        **kwargs
    ) -> None:
        """
        :param boxLengths: box lengths.
        :param rMin: minimum r value.
        :param rMax: maximum r value.
        :param binWidth: width of r bins.
        :param kwargs: keyword arguments.
        """
        self.rMin = rMin
        self.rMax = self._rMax(rMax, binWidth, boxLengths)
        self.binWidth = binWidth
        self.rs = None
        self.rdf_arr_size = None

        # store data.
        self.get_lims = True
        self.scat_lims = []
        self.numCalcs = 0
        self.xray = None
        self.neutron = None

        # store old and new rdfs.
        self.rdf = None
        self.rdf_smoothing = None
        self.smooth = {}

        # store constants for x-ray calculations.
        self.fourier_sum_x = None
        self.ppff = None
        self.fq = None
        
        # store constants for neutron calculations.
        self.fourier_sum_n = None
        self.neutron_constant = None
        self.sq = None
    
    
    def set_neutron(self, 
        neutron: np.ndarray, 
        weight: float, 
        calculator: str
    ) -> None:
        """
        The Q-values to use for total scattering calculations. If fitting to
        total scattering data, use same Q-values so that evaulating the cost
        function is easier.
        """
        num_qs = neutron[:,0].shape[0]
        self.neutron = ScatteringData(neutron, *self._weight_array(num_qs, weight), calculator, np.ones(num_qs))
    
    
    def set_xray(self, xray, weight, calculator):
        """
        The Q-values to use for total scattering calculations. If fitting to
        total scattering data, use same Q-values so that evaulating the cost
        function is easier.
        """
        num_qs = xray[:,0].shape[0]
        self.xray = ScatteringData(xray, *self._weight_array(num_qs,weight), calculator, np.ones(num_qs))


    def neutron_cost(self, weight: float):
        """
        Caclculate the neutron scattering chi2 value.
        """
        return chi2_sum(self.sq, self.neutron.y, weight)
    
    
    def xray_cost(self, weight: float):
        """
        Caclculate the x-ray scattering chi2 value.
        """
        return chi2_sum(self.fq, self.xray.y, weight)


    def compute_scattering(self, rdf: np.ndarray) -> None:
        """
        Compute new patterns.

        :param rdf: radial distribution function.
        """
        
        if self.neutron is not None:
            self.sq = neutron(self.neutron.x, self.neutron_constant, rdf,
                        self.fourier_sum_n)
                        
        if self.xray is not None:
            self.fq = xray(self.xray.x, self.xray.y, self.ppff, rdf,
                        self.fourier_sum_x, self.xray.mulQ)

    
    def plot_real_space(self, sigma):
        """
        Plot the radial distriubtion function.
        """

        norm = 0
        for sym in self.box.elements:
            norm += self.box.composition.atomic_fraction(sym) * atomic_numbers[sym]

        # oxygen, carbon, calcium
        cols = ["#003f5c", "#bc5090", "#ffa600"]

        cols = ["#ffa600", "#bc5090", "#bc5090", "#bc5090", "#ffa600", "#ffa600", "#ffa600", "#003f5c", "#003f5c", "#003f5c"]
        ls = ["dotted", "", "dashed", "dotted", "", "dashed", "solid", "", "", "solid"]
        skip = [1, 4, 7, 8]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        for i,pair in enumerate(self.box.elementPairs):
            if i not in skip:
                gij = gaussian_filter1d(self.rdf[i,:], sigma/self.binWidth)
                gij *= self.box.composition.atomic_fraction(pair[0]) * self.box.composition.atomic_fraction(pair[1])
                gij *= atomic_numbers[pair[0]] * atomic_numbers[pair[1]]
                gij /= norm * norm
                ax.plot(self.rs, gij, label="—".join(pair), c=cols[i], linestyle=ls[i])
                ax.text(np.amax(self.rs), gij[-1]+0.01, "—".join(pair), fontsize=12, ha="right", c=cols[i])

        ax.set_ylabel(r"$c_{i}c_{j}\frac{Z_{i}Z_{j}}{\left(\sum_{i=1} c_{i}Z_{i}\right)} g_{ij}(r)$", fontsize=15)
        ax.set_xlabel(r'$r$ (Å)', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.out/"real_space.png", dpi=600)
    
    
    def plot_scattering(self):
        """ """
        c = 0
        fig, ax = plt.subplots(2, self.numCalcs, figsize=(8*self.numCalcs,8),)
            #gridspec_kw={'height_ratios': [2, 1]})
        if self.numCalcs > 1:
            if self.neutron is not None:
                ax[0,c].plot(self.neutron.x, self.neutron.y, c="k", label="experiment")
                ax[0,c].plot(self.neutron.x, self.sq, c="r", label="simulation")
                ax[0,c].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[0,c].set_ylabel("$S^N(Q)$")

                if self.get_lims:
                    self.scat_lims.append([ax[0,c].get_xlim(), ax[0,c].get_ylim()])
                
                #ax[0,c].set_xlim(self.scat_lims[0,c][0])
                #ax[0,c].set_ylim(self.scat_lims[0,c][1])

                # plot difference function.
                diff = chi2_diff(self.sq, self.neutron.y, self.neutron.weight)
                ax[1,c].plot(self.neutron.x, diff, c="purple")

                ax[1,c].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[1,c].set_ylabel("$\Delta$")
                c+=1

            if self.xray is not None:
                ax[0,c].plot(self.xray.x, self.xray.y, c="k", label="experiment")
                ax[0,c].plot(self.xray.x, self.fq, c="r", label="simulation")
                ax[0,c].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[0,c].set_ylabel("$F^X(Q)$")
                
                if self.get_lims:
                    self.scat_lims.append([ax[0,c].get_xlim(), ax[0,c].get_ylim()])

                # plot difference function.
                diff = chi2_diff(self.fq, self.xray.y, self.xray.weight)
                ax[1,c].plot(self.xray.x, diff, c="purple")

                ax[1,c].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[1,c].set_ylabel("$\Delta$")

                #ax[c].set_xlim(self.scat_lims[0,c][0])
                #ax[c].set_ylim(self.scat_lims[0,c][1])

                c+=1
        else:
            if self.neutron is not None:
                ax[0].plot(self.neutron.x, self.neutron.y, c="k", label="experiment")
                ax[0].plot(self.neutron.x, self.sq, c="r", label="simulation")
                ax[0].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[0].set_ylabel("$S^N(Q)$")

                if self.get_lims:
                    self.scat_lims.append([ax[0].get_xlim(), ax[0].get_ylim()])
                
                #ax[0,c].set_xlim(self.scat_lims[0,c][0])
                #ax[0,c].set_ylim(self.scat_lims[0,c][1])

                # plot difference function.
                diff = chi2_diff(self.sq, self.neutron.y, self.neutron.weight)
                ax[1].plot(self.neutron.x, diff, c="purple")

                ax[1].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[1].set_ylabel("$\Delta$")
                
            if self.xray is not None:
                ax[0].plot(self.xray.x, self.xray.y, c="k", label="experiment")
                ax[0].plot(self.xray.x, self.fq, c="r", label="simulation")
                
                if self.xray.mulQ:
                    ylabel = "$QF^X(Q)$"
                else:
                    ylabel = "$F^X(Q)$"
                    
                ax[0].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[0].set_ylabel(ylabel)
                
                if self.get_lims:
                    self.scat_lims.append([ax[0].get_xlim(), ax[0].get_ylim()])

                # plot difference function.
                diff = chi2_diff(self.fq, self.xray.y, self.xray.weight)
                ax[1].plot(self.xray.x, diff, c="purple")

                ax[1].set_xlabel("$Q$ (Å$^{-1}$)")
                ax[1].set_ylabel("$\Delta$")
            

        #plt.legend(loc="upper right")
        fig.tight_layout()
        plt.savefig(self.out/"scattering.png", dpi=600)
        
        plt.clf()
        plt.close("all")
        del fig, ax, diff
        
        self.get_lims = False

    
    def _rMax(self, rMax, binWidth, boxLengths):
        """ Compute the maximum r-value for distance histogramming. """
        if rMax is None:
            rMax = np.amin(boxLengths) / 2
        return utils.round_maxr(rMax,binWidth)
    

    def _weight_array(self, numPoints, weight):
        """ """
        singleWeight = False
        if type(weight) in [int, float]:
            weight = np.full(numPoints, weight)
            singleWeight = True
        return weight, singleWeight
        

    def initialise_neutron(self, box, HD_ratio):
        """ """
        self.numCalcs += 1

        # compute fourier constant r-sum.
        self.fourier_sum_n = fourier_rsum(self.neutron.x, self.rs, self.binWidth)

        # determine level of deuteration.
        neutScatLength["H"] = (neutScatLength["H1"] * HD_ratio) + \
            (neutScatLength["H2"] * (1-HD_ratio))

        # create array of concentration * scattering length.
        self.neutron_constant = np.zeros((len(box.elementPairs),))
        for i, (a1, a2) in enumerate(box.elementPairs):
            self.neutron_constant[i] = box.composition.atomic_fraction(a1) * \
                box.composition.atomic_fraction(a2) * \
                fm2m(neutScatLength[a1]) * fm2m(neutScatLength[a2]) * (2-bool(a1==a2))
        
        # compute initial scattering.
        self.sq = neutron(self.neutron.x, self.neutron_constant, self.rdf, self.fourier_sum_n)
        
    
    def initialise_xray(self, box):
        """ """

        self.numCalcs += 1

        # compute fourier constant r-sum.
        self.fourier_sum_x = fourier_rsum(self.xray.x, self.rs, self.binWidth)

        # compute the partial pair form factors.
        self.ppff = form_factors(   box.elements, box.elementPairs, self.xray.x,   
                                    box.composition)

        # compute initial scattering and minimise difference between experiment
        # and calculated (X-ray scattering intensity is complicated).
        self.fq = xray( self.xray.x, self.xray.y, self.ppff, self.rdf,
                        self.fourier_sum_x, self.xray.mulQ)

    
    def smooth_intramolecular(self, refine_pairs, smooth_parameters,
            rdf_neutron, min_r, max_r, plot_lim=[0,6], plot=False):
        """
        By holding molecular units as rigid bodies we negate the effects of
        thermal motion on the form of the radial distribution functions which
        can lead to spikiness that propogates in the Fourier transform. Here is 
        a function for applying a constant smoothing for intramolecular bonds in
        the unsubsituted ZIF system.

        Args
        ----
        refine_pairs: list
            A list of tuples with the pair of elements for which the
            corresponding pdf should be refined.

        smooth_parameters: np.arange
            A range of gaussian broadnesses to pass to
            scipy.filter.guassian_filter1d to smoothen the rdf.

        rdf_neutron: np.array
            Experimental total radial distribution function calculated by
            Fourier transfrom of the neutron S(Q). First column should be the
            r-values, second column should be the intensity values.
        """

        # the smoothing will be given as the difference between the smoothed and
        # raw radial distribution function such that it can be added on to each
        # rdf at each move.
        self.rdf_smoothing = np.zeros(self.rdf.shape)

        for pair in refine_pairs:
            self.smooth[pair] = 0.0

        # get index of min/max r-values to take when computing the difference
        # between the experimental and computed rdfs.
        rmin_ix = np.where(rdf_neutron[:,0]==min_r)[0][0]
        rmax_ix = np.where(rdf_neutron[:,0]==max_r)[0][0]

        # for every trialled gaussian broadness, compute the difference betweeen
        # the experiment total radial distriubtion function and the computed.
        diffvals = np.zeros(smooth_parameters.shape)
        for i, val in enumerate(smooth_parameters):
            
            # need to make sure we don't write to the RDF stored internally in 
            # LAMMPS (i.e. make a deep copy of the array).
            rdf = copy.deepcopy(self.rdf)

            # update the current value for this correlation.
            for pair in refine_pairs:
                self.smooth[pair] = val
            
            smooth_val = np.zeros(len(self.box.elementPairs))
            for j,(a1,a2) in enumerate(self.box.elementPairs):
                if (a1,a2) in self.smooth.keys():
                    smooth_val[j] = self.smooth[(a1,a2)]

            # apply guassian filter to the relevant pairs.
            for j,(a1,a2) in enumerate(self.box.elementPairs):
                if smooth_val[j] != 0:
                    gaussian_filter1d(rdf[j,:], smooth_val[j], output=rdf[j,:])

            # compute neutron real space.
            gr = np.zeros(self.rs.shape[0])
            for j in range(len(self.box.elementPairs)):
                gr[:] += self.neutron_constant[j] * (rdf[j,:]-1)
            gr *= 1e28

            # compute the difference between the computed and experimental fx().
            diffvals[i] = chi2_sum( gr[rmin_ix:rmax_ix], 
                                    rdf_neutron[rmin_ix:rmax_ix,1],
                                    sigma=1)

        # compute the optimal parameter.
        minix = np.argmin(diffvals)
        for pair in refine_pairs:
            self.smooth[pair] =  smooth_parameters[minix]

        # hence compute "difference" in the radial distributions pre- and post-
        # gaussian smoothing so that it can be added after each move.
        old_rdf = copy.deepcopy(self.rdf)
        new_rdf = copy.deepcopy(self.rdf)
        smooth_val = np.zeros(len(self.box.elementPairs))
        for j,(a1,a2) in enumerate(self.box.elementPairs):
            if (a1,a2) in self.smooth.keys():
                smooth_val[j] = self.smooth[(a1,a2)]

        # apply guassian filter to the relevant pairs.
        for j,(a1,a2) in enumerate(self.box.elementPairs):
            if smooth_val[j] != 0:
                gaussian_filter1d(new_rdf[j,:],smooth_val[j],output=new_rdf[j,:])

        # compute the difference.
        self.rdf_smoothing = new_rdf - old_rdf

        # write smoothing parameters to file.
        fstr = "Smoothing summary:\n"
        fstr += "-"*len("Smoothing summary:") + "\n"
        for k,v in self.smooth.items():
            fstr += "-".join(k)+f" {v:.2f}" + "\n"
        fstr += "\n"
        with open(self.out/"smoothing.dat", "w") as f:
            f.write(fstr)

        # plot the smoothed total radial distribution function.
        if plot:
            gr = np.zeros(self.rs.shape[0])
            for j in range(len(self.box.elementPairs)):
                gr[:] += self.neutron_constant[j] * (new_rdf[j,:]-1)
            gr *= 1e28

            gr_raw = np.zeros(self.rs.shape[0])
            for j in range(len(self.box.elementPairs)):
                gr_raw[:] += self.neutron_constant[j] * (old_rdf[j,:]-1)
            gr_raw *= 1e28

            doubleColumnFigSize = (3.375*2,3.375*(5**0.5-1)/2)
            fig, ax = plt.subplots(1,2, figsize=doubleColumnFigSize)

            ax[0].plot(rdf_neutron[:,0], rdf_neutron[:,1], c="k")
            ax[0].plot(self.rs, gr_raw, c="r", linewidth=1.5)

            ax[1].plot(rdf_neutron[:,0], rdf_neutron[:,1], c="k")
            ax[1].plot(self.rs, gr, c="r", linewidth=1.5)
            
            for i in [0,1]:
                ax[i].xaxis.set_minor_locator(AutoMinorLocator(2))
                ax[i].yaxis.set_minor_locator(AutoMinorLocator(2))
                ax[i].set_xlabel(r"$r ({\rm{\AA{}}})$")
                ax[i].set_ylabel(r"$G(r)$")
                ax[i].set_xlim(plot_lim)

            plt.tight_layout()
            plt.savefig(self.out/"G(r)_smoothing.png", dpi=800)
            plt.close()
    
    
    def update_smoothing(self):
        """
        Update the difference rdf for the smoothed-unsmoothed functions.
        """
        
        # make deep copies of the rdf.
        old_rdf = copy.deepcopy(self.rdf)
        new_rdf = copy.deepcopy(self.rdf)
        
        # make sure all smooth values are used.
        smooth_val = np.zeros(len(self.box.elementPairs))
        for j,(a1,a2) in enumerate(self.box.elementPairs):
            if (a1,a2) in self.smooth.keys():
                smooth_val[j] = self.smooth[(a1,a2)]

        # apply guassian filter to the relevant pairs.
        for j,(a1,a2) in enumerate(self.box.elementPairs):
            if smooth_val[j] != 0:
                gaussian_filter1d(new_rdf[j,:],smooth_val[j],output=new_rdf[j,:])
        
        # then take the difference.
        self.rdf_smoothing = new_rdf - old_rdf
        

@njit(parallel=True, fastmath=True)
def neutron(qs, const, rdf, rsum):
    sq = np.zeros((rdf.shape[0],qs.shape[0]))
    for q in prange(qs.shape[0]):
        for i in range(rdf.shape[0]):
            sq[i][q] = np.sum(np.multiply(rdf[i]-1,rsum[q])) * const[i]
    return np.sum(sq, axis=0) * 1e28 # to get into units of Barns.


@njit(parallel=True, fastmath=True)
def neutron_real_space(gr, neutron_constant, elementPairs, rdf):
    """
    Compute the neutron total radial distribution function.
    """
    for i in prange(len(elementPairs)):
        gr[:] += neutron_constant[i] * (rdf[i,:]-1)
    gr *= 1e28

    

@njit(parallel=True, fastmath=True)
def xray(qs, fq_expt, ppff, rdf, rsum, mulQ):
    """
    Calculate total X-ray scattering function.

    Args
    ----
    qs      :  all Q-values to iterate over;
    ppff    :  pairwise partial form factors;
    rdf     :  radial distribution functions;
    rSum    :  constant pre-calculatd rTerm.

    Returns
    -------
    fq      :   X-ray total scattering.
    """
    fq = np.zeros((ppff.shape[0],qs.shape[0]))
    for q in prange(qs.shape[0]):
        for i in range(ppff.shape[0]):
            fq[i][q] = ppff[i][q] * np.sum(np.multiply(rdf[i]-1, rsum[q]))
    if mulQ:
        fq = np.multiply(qs,fq)
    return minimise_difference(np.sum(fq, axis=0),fq_expt)


@njit
def minimise_difference(a, b):
    """
    Minimise goodness-of-fit metric between the calculated and experimental
    X-ray total scattering functions.
    
    \chi^2 = \sum_i[sF_calc(Q_i) - F_exp(Q_i)]^2 by solving for stationary
    point, and therefore evaluating:

    s = \frac{\sum_i F_calc(Q_i)F_exp(Q_i)}{\sum_i[F_calc(Q_i)]^2}.

    Args
    ----
    a:
        calculated values (to rescale).

    b:
        measured values.
    """
    return a * np.sum( np.multiply(a,b) ) / np.sum( np.square(a) )


@njit
def chi2_sum(calc, expt, sigma):
    """
    Calculate scattering function cost. Note I multiply the value by half here.

    Args
    ----
    calc
        Calculated scattering function.
    expt
        Experimental scattering function.
    sigma
        weight per-point.
    """
    return np.sum(np.divide( np.square( calc - expt ), np.square(sigma) )) * 0.5


@njit
def chi2_diff(calc, expt, sigma):
    """
    Calculate scattering function cost.

    Args
    ----
    calc
        Calculated scattering function.
    expt
        Experimental scattering function.
    sigma
        weight per-point.
    """
    return np.divide(calc-expt, sigma)


@njit
def fm2m(x):
    """
    Convert fm to m.
    """
    return x*1E-15


def fourier_rsum(qs, rs, binWidth):
    """
    Calculate the constant r term in the X-ray total scattering to avoid 
    unecessary re-calcuations during the simulation. For every value of Q,
    calculate:

        r^2 * dr * sin(Q*r) / (Q*r)
    """

    def rsum_q(q, rs, r2dr):
        """
        Define the rsum value for a given Q value.
        """
        qr = q * rs
        return np.multiply( np.divide( np.sin(qr),qr ), r2dr )

    # r^2 * dr
    r2dr = np.square(rs) * binWidth

    # then calculate for all Q values.
    rsum = np.zeros((qs.shape[0], rs.shape[0]), dtype=np.float64)
    for q in range(qs.shape[0]):
        rsum[q, :] = rsum_q(qs[q], rs, r2dr)
    return rsum


def form_factors(elements: tuple, atom_pairs: list, qs: np.ndarray, composition):
    """
    Calculate form factors and pairwise partial form factors from emprical
    coefficients for a given list of elements and an array of Q values. The
    X-ray form factors are given by:

        f(s) = A\exp{-as^2} + B\exp{-bs^2} + C\exp(-cs^2) + D\exp{-ds^2} + E,

    where s = Q / 4\pi. These are then combined into pairwise partial
    form-factor terms:

    p_ij(Q) = \frac{\alpha_ij c_i c_j f_i(Q) f_j(Q)}
                {\right[ \sum_{i=1}^n c_i f_i(Q) \left] ^ 2}

    Args
    ----
    elements: tuple
    """
    # --------------------------------------- #
    # Calculate form factors for each element.
    # --------------------------------------- #
    # Calculate s^2 = \frac{ Q }{ 4\pi }
    s2 = np.square(qs / 4 / np.pi)

    # Setup 2D numpy array with n rows and m columns, where n is the number of
    # elements, and m is the number of Q values being sampled.
    ff = np.array([s2] * len(elements), dtype=float)

    # Create atom dictionary to store which row corresponds to which element.
    aDict = {}
    for i,e in enumerate(elements):

        # Keep track of which row corresponds to which element.
        aDict[e] = i

        # Fetch empirical coefficients.
        A,a,B,b,C,c,D,d,E = empCoeff[e]

        row = ff[i,:]
        ff[i,:] =   A*np.exp(-a*row) + \
                    B*np.exp(-b*row) + \
                    C*np.exp(-c*row) + \
                    D*np.exp(-d*row) + \
                    E
    
    # --------------------------------------- #
    # Calculate pairwise partial form factors.
    # --------------------------------------- #
    # Define function from Pymatgen Composition class that gets atomic fraction.
    numConc = getattr(composition,"atomic_fraction")

    # Setup 2D numpy array with n rows and m columns, where n in the number of 
    # pairs of elements, and m is the number of Q values being sampled.
    ppff = np.zeros((len(atom_pairs),ff.shape[1]), dtype=np.float64)

    # numerator = \alpha_ij * c_i * c_j * f_i(s) * f_j(s)
    for i,(a1,a2) in enumerate(atom_pairs):
        ppff[i,:] = (2-bool(a1==a2)) * numConc(a1) * numConc(a2) * np.multiply(ff[aDict[a1]],ff[aDict[a2]])
                
    # denominator = [\sum_i c_i * f_i(Q)]^2
    # Get numConc for coefficients in correct order.
    coeffs = np.array([numConc(a) for a in elements])
    ppff /= np.square(np.sum(np.multiply(ff.T,coeffs).T,axis=0))

    return ppff
