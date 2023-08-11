"""
27.06.23
@tcnicholas
Plotting functions and formatting.
"""

from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import AutoMinorLocator


font = {'family':'sans-serif','size':7, 'sans-serif': ['Helvetica']}

sns.set(font=font, style='ticks', rc={
        'axes.edgecolor': 'k',
        'axes.facecolor': 'None',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'text.color': 'k',
        'figure.facecolor': 'white',
})


sns.set_context(rc={"font.size":12, "axes.titlesize":12, "axes.labelsize":12})

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['axes.linewidth'] = 2.0

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 2.0

plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 2.0
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 2.0


cm = 1/2.54


RDF_SCATTER_KWARGS = {
    'facecolors': 'none',
    'edgecolors': 'k',
    'lw': 2,
    'alpha': 1.0,
    's': 1
}


def _update_rdf_axes(axes: np.ndarray) -> None:
    """
    Update ticks and limits for axes.

    :param axes: axes to update.
    :return: None.
    """
    axes[0, 0].set_xticks(np.arange(0, 12, 2))
    axes[0, 0].set_yticks(np.arange(0, 5.0, 1.0))
    axes[1, 0].set_yticks(np.arange(0, 2.0, 0.5))

    ylim_min = -0.35428129893759086
    for ax in axes[0].ravel():
        ax.set_ylim([ylim_min, abs(ylim_min) + 4])
    for ax in axes[1].ravel():
        ax.set_ylim([ylim_min, abs(ylim_min) + 1.5])

    
def create_subplots():
    """
    Create a grid of subplots with specified attributes.
    """
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 6), 
                           gridspec_kw={'width_ratios': [1, 1, 1, 1], 
                            'height_ratios': [1, 3]}, sharex='col')
    return fig, ax


def plot_kde(ax, data1, data2=None, lw=3, color=['r', 'k'], alpha=None):
    """
    Plot a KDE plot for the given data.
    """
    if data2 is not None:
        sns.kdeplot(data=[data1, data2], ax=ax, palette=color, lw=lw, alpha=alpha)
    else:
        sns.kdeplot(data=[data1], ax=ax, color=color[0], lw=lw, alpha=alpha)


def clean_axes(ax):
    """Remove ticks, labels, and spines from axes."""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend([], [], frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, 
                   labelbottom=False, length=5)


def adjust_y_limit(ax):
    """Adjust the y limits of the plot."""
    yu = ax.get_ylim()
    mv = yu[1] * 0.05
    ax.set_ylim([yu[0] - mv, yu[1] + mv])


def set_axis_properties(ax, labels, xlimit, xticks, xlabel):
    """Set labels, limits and ticks for x and y axis."""
    ax.set_xlim(xlimit)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.legend(labels=labels, frameon=False, handlelength=1, loc='upper center', 
              bbox_to_anchor=(0.5, -0.5), fancybox=False, shadow=False, ncol=1,
            fontsize=7)
    ax.set_xlabel(xlabel)



def format_bond_and_angles_axes(axes: np.ndarray):
    """
    Quick clean-up of the axes for the bonds and angles distribution plots.

    :param axes: axes to format.
    :return: None.
    """

    for axx in axes[0, :]:
        clean_axes(axx)

    # Set y labels and ticks
    for axx in axes.ravel():
        axx.set_ylabel('')
        axx.set_yticks([])
        axx.set_yticklabels([])

    # Adjust y limit
    for axx in axes[0, :]:
        adjust_y_limit(axx)


def plot_rdf(
    ax: plt.Axes, 
    x: np.ndarray(1), 
    y: np.ndarray(1), 
    rmin: float = 2, 
    rmax: float = 10
):
    """
    Plot radial distribution function.

    :param ax: matplotlib axis object.
    :param x: distances (Å).
    :param y: probability density.
    :param rmin: minimum distance to plot (Å).
    :param rmax: maximum distance to plot (Å).
    :return: None.
    """
    ix = np.logical_and(x>rmin,x<rmax)
    ax.plot(x[ix], y[ix], c='k', lw=3)


def plot_histogram(
    ax: plt.Axes, 
    data: np.ndarray, 
    orientation: str = 'vertical', 
    color: str = 'white', 
    edgecolor: str = 'k',
    width: float = 1.0
) -> None:
    """
    Plot histogram with given data.

    :param ax: Axis to plot histogram on.
    :param data: Data to plot.
    :param orientation: Orientation of the histogram, either 'vertical' or 
        'horizontal'.
    :param color: Color of the bars.
    :param edgecolor: Edge color of the bars.
    """
    unique, counts = np.unique(data, return_counts=True)
    if orientation == 'vertical':
        ax.bar(unique, 100*counts/len(data), width, color=color, 
               edgecolor=edgecolor)
    else:
        ax.barh(unique, 100*counts/len(data), width, color=color, 
                edgecolor=edgecolor)
    ax.axis('off')


def scatter_hist(
    ax: plt.Axes, 
    ax_histx: plt.Axes, 
    ax_histy: plt.Axes, 
    x: np.ndarray, 
    y: np.ndarray
) -> None:
    """
    Generate a scatter plot with histograms on the sides.

    :param ax: The axis for the scatter plot.
    :param ax_histx: The axis for the X histogram.
    :param ax_histy: The axis for the Y histogram.
    :param x: Data for the x-axis.
    :param y: Data for the y-axis.
    """
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    bins = [np.arange(np.amin(data)-0.5,np.amax(data)+0.5, 1) for data in (x,y)]
    ax.hist2d(x, y, cmap="Greys", bins=bins)
    ax.set_xlim([np.amin(x)-0.5,np.amax(x)+0.5])
    ax.set_ylim([np.amin(y)-0.5,np.amax(y)+0.5])
    ax.set_xticks(np.arange(0,np.amax(x)+1))

    ax.set_xlabel(r"Ca–O$_{\rm{W}}$ Coordination Number")
    ax.set_ylabel(r"Ca–O$_{\rm{C}}$ Coordination Number")

    plot_histogram(ax_histx, x, 'vertical')
    plot_histogram(ax_histy, y, 'horizontal')


def get_all_values(
    data: Dict,
    model: str = 'hrmc',
    property: str = 'volumes'
) -> np.ndarray(1):
    """
    Gather all Voronoi volumes from the data.

    :param data: dictionary of Voronoi statistics.
    :param model: name of model (e.g. ljg, hrmc).
    :param property: name of property (e.g. volumes, neighbours).
    :return: array of values.
    """
    values = []
    for frame, stats in data[model].items():
        values.extend(stats[property])
    return np.array(values)


def plot_voronoi_volume_kde(
    ax: plt.Axes,
    data: Dict,
    model: str = 'hrmc',
    colour: str = 'k',
    alpha: float = 1.0,
    linewidth: float = 1.0,
    xrange: np.ndarray(1) = None,
) -> np.ndarray(1):
    """
    Plot the Voronoi volume kernel density estimate.

    :param ax: matplotlib axis object.
    :param data: dictionary of Voronoi statistics.
    :param model: name of model (e.g. ljg, hrmc).
    :param colour: colour of line.
    :param alpha: alpha of line.
    :param linewidth: linewidth of line.
    :param xrange: range of x values.
    :return: xrange.
    """
    value = get_all_values(data, model=model, property='volumes')
    kde = gaussian_kde(value)

    if xrange is None:
        xrange = np.linspace(value.min(), value.max(), len(value))

    sf = 1 if model == 'hrmc' else -1

    sns.lineplot(
        x=xrange, 
        y=kde(xrange)*sf, 
        color=colour,
        alpha=alpha, 
        lw=linewidth, 
        ax=ax
    )

    ax.fill_between(xrange, kde(xrange)*sf, color=colour, alpha=0.75)
    ax.set_xlabel(r'Voronoi cell volume ($\rm \AA{}^3$)')
    ax.set_ylabel(r'$\longleftarrow {\rm Likelihood} \longrightarrow$')
    ax.set_xticks(np.arange(50,150,25))
    ax.set_xlim([45,130])
    ax.set_yticks([])
    ax.set_ylim([-0.04,0.04])

    return xrange


def plot_voronoi_neighbour_histogram(
    ax: plt.Axes,
    data: Dict,
    model: str = 'hrmc',
    colour: str = 'k',
) -> None:
    """
    Plot the Voronoi neighbour histogram.

    :param ax: matplotlib axis object.
    :param data: dictionary of Voronoi statistics.
    :param model: name of model (e.g. ljg, hrmc).
    :param colour: colour of bar.
    :return: None.
    """

    sf = 1 if model == 'hrmc' else -1

    value = get_all_values(data, model=model, property='neighbours')
    nsbins, nscounts = np.unique(value, return_counts=True)
    ax.bar(nsbins, sf*100*nscounts/value.shape[0], 1.0, color=colour)

    ax.set_xlabel(r'Voronoi cell face count')
    ax.set_ylabel(r'$\longleftarrow {\rm Count} \longrightarrow$')
    ax.set_xticks(np.arange(12,22,2))
    ax.set_xlim([10,21])
    ax.set_ylim([-25,25])
    ax.set_yticks([])


def label_top_and_bottom_axes(
    axes: np.ndarray(1),
    top_label: str = 'HRMC',
    bottom_label: str = 'LJG(MC)',
) -> None:
    """
    Label the top and bottom axes with the model names.

    :param axes: array of matplotlib axis objects.
    :param top_label: label for top axis.
    :param bottom_label: label for bottom axis.
    :return: None.
    """
    for ax in axes:
        ax.text(1.0, 0.95, top_label, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)
        ax.text(1.0, 0.05, bottom_label, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes)
        

def get_all_ring_counts(
    data: Dict,
    model: str = 'hrmc',
    property: str = 'rings'
) -> np.ndarray(1):
    """
    Gather all Voronoi volumes from the data.

    :param data: dictionary of Voronoi statistics.
    :param model: name of model (e.g. ljg, hrmc).
    :param property: name of property (e.g. volumes, neighbours).
    :return: array of values.
    """
    ring_sizes = None
    ring_counts = []
    for frame, stats in data[model].items():

        if ring_sizes is None:
            ring_sizes = stats[property][:,0]

        # check the same r values are used before storing the data.
        assert np.allclose(ring_sizes, stats[property][:,0])
        ring_counts.append(stats[property][:,1])

    return {
        'ring_sizes': ring_sizes, 
        'ring_counts': np.vstack(ring_counts).sum(axis=0)
    }
    

def plot_ring_counts(
    ax: plt.Axes,
    data: Dict,
    model: str = 'hrmc',
    ring_type: str = 'rings',
):
    """
    Plot the ring counts.

    :param ax: matplotlib axis object.
    :param data: dictionary of Voronoi statistics.
    :param model: name of model (e.g. ljg, hrmc).
    :param ring_type: name of ring type (e.g. rings, cages).
    :param colour: colour of line.
    :return: None.
    """
    sf = 1 if model == 'hrmc' else -1
    colour = 'k' if model == 'hrmc' else 'r'

    ring = get_all_ring_counts(data, model, ring_type)
    ax.bar(ring['ring_sizes'], sf*ring['ring_counts'], width=1.0, color=colour)

    if ring_type == 'rings':
        ax.set_xlim([2.,11])
        ax.set_xticks(np.arange(3,11,1))
        ax.set_xlabel("Ring size")
    else:
        ax.set_xlim([2.,7])
        ax.set_xticks(np.arange(3,7,1))
        ax.set_xlabel("Strong ring size", fontsize=7)

    ax.set_yticks([])
    mv = np.amax(np.abs(ax.get_ylim()))
    ax.set_ylim([-mv-mv*0.05, mv+mv*0.05])
    ax.set_yticks([])
    ax.axhline(y=0, linestyle='-', linewidth=1, color='black')


def get_ymin_ymax(y: float) -> Tuple[float, float]:
    """
    Determine the minimum and maximum y values based on the input.

    :param y: y value.
    :return: tuple of ymin and ymax.
    """
    return (y, 0) if y < 0 else (0, y)


def get_vlines(
    x_values: List[float], 
    y_values: List[float]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract vertical line data from the given data.

    :param x_values: list of x values.
    :param y_values: list of y values.
    :return: tuple of x, ymin, and ymax values.
    """
    ymin, ymax = zip(*map(get_ymin_ymax, y_values))
    return list(x_values), list(ymin), list(ymax)


def plot_orientational_correlation(
    ax, 
    corr, 
    col="#ffa600",
    ylim=[-1,1], 
    alpha=0.5, 
    alpha2=0.2, 
    lw=2, 
    rmax=10,
    ptsize=2
):
    """
    Plot the orientational correlation function.

    :param ax: matplotlib axis object.
    :param corr: orientational correlation function.
    :param col: colour of the data points.
    :param ylim: y limits of the plot.
    :param alpha: alpha of the error bars.
    :param alpha2: alpha of the lines.
    :param lw: linewidth of the lines.
    :param rmax: maximum distance to plot.
    :param ptsize: size of the data points.
    """
    
    # limit the cutoff value for plotting.
    _rs, _corr, _count, _std_dev = corr
    use = np.argwhere(_rs<rmax).flatten()

    # plot the data and error bars.
    ax.errorbar(_rs[use], _corr[use], _std_dev[use], fmt="none", ecolor="r",
        elinewidth=lw/2, capsize=1, alpha=alpha, zorder=2)
    ax.scatter(_rs[use], _corr[use], edgecolors=col, label="HRMC", s=ptsize, 
        facecolors='none', linewidths=0.5, zorder=3)

    # plot lines from zero-line to the correlation value.
    x, ymin, ymax = get_vlines(_rs[use], _corr[use])
    ax.vlines(x=np.array(x), ymin=np.array(ymin), ymax=np.array(ymax), 
        colors=col, lw=lw, alpha=alpha2, zorder=1)

    ax.set_xlabel(r"$r~({\rm{\AA{}}})$")
    ax.set_ylabel(r"$\phi(r)$")
    ax.set_xlim([2, 10])
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(2,11,1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    return _rs, _corr, _std_dev


def plot_on_ax(ax, x, y, style, label=None, **kwargs):
    """
    Helper function to plot on an axis.
    """
    ax.plot(x, y, c=style['color'], label=label, **kwargs)


def add_hlines(ax, y, start, end, style):
    """
    Helper function to add horizontal lines to an axis.
    """
    ax.hlines(y, start, end, **style)


def plot_between_cutoffs(ax, rdf, r_min, r_max, label, offset=0.0, color='k'):
    """
    Plot the radial distribution function between r_min and r_max.
    """
    indices = np.logical_and(rdf[:,0]<r_max, rdf[:,0]>r_min)
    ax.plot(
        rdf[indices,0], rdf[indices,1] + offset, '-o', 
        markerfacecolor='none', markeredgecolor=color, lw=2, alpha=1.0, 
        markersize=5, markeredgewidth=2, color=color
    )
    ax.text(rdf[indices,0][-1], 1.4+offset, label, horizontalalignment='right')