"""
29.06.23
@tcnicholas
Generate g3 calculator input file.
"""


import tempfile
import subprocess
from dataclasses import dataclass, field


@dataclass
class G3InputFileParameters:
    """
    A class used to hold the parameters for the input file.

    :param n_atom: Number of atoms.
    :param n_type: Number of types.
    :param n_conf: Number of configurations.
    :param nr_grid: Grid dimension in r direction.
    :param na_grid: Grid dimension in angle direction.
    :param r_min: Minimum radius to consider bonded pairs.
    :param r_max: Maximum radius to consider bonded pairs.
    :param l_rcut: The total cutoff for considering neighbour interactions.
    :param type_c: Type C.
    :param type_e: Type E.
    :param input_file: Input file name.
    :param output_file: Output file name.
    """
    nAtom: int = field(default=1620, metadata={'help': "Number of atoms."})
    nType: int = field(default=1, metadata={'help': "Number of types."})
    nConf: int = field(default=1, metadata={'help': "Number of configurations."})
    nrGrid: int = field(default=401, metadata={'help': "Grid dimension in r direction."})
    naGrid: int = field(default=201, metadata={'help': "Grid dimension in angle direction."})
    Rmin: float = field(default=0.1, metadata={'help': "Minimum radius to consider bonded pairs."})
    Rmax: float = field(default=9.0, metadata={'help': "Maximum radius to consider bonded pairs."})
    LRcut: float = field(default=12.0, metadata={'help': "The total cutoff for considering neighbour interactions."})
    TypeC: int = field(default=1, metadata={'help': "Type C."})
    TypeE: int = field(default=1, metadata={'help': "Type E."})
    input: str = field(default='ca_mc.dump', metadata={'help': "Input file name."})
    output: str = field(default='lj_result.dat', metadata={'help': "Output file name."})


class G3InputFileGenerator:
    """
    A class used to generate an input file with specified parameters.

    :params InputFileParameters: an instance of InputFileParameters holding all 
        parameter values.

    :method generate_file(filename): writes the parameters to a file in the 
        format: '<value> <parameter>'
    """
    def __init__(self, 
        params: G3InputFileParameters = G3InputFileParameters()
    ) -> None:
        """
        Initialise the InputFileGenerator with provided or default parameters.

        :param params: an instance of InputFileParameters holding all parameter
            values.
        """
        self.params = params


    def generate_file(self, filename: str) -> None:
        """
        Writes the parameters to a file.

        :param filename: name of file to write to.
        """
        with open(filename, 'w') as file:
            for parameter, value in self.params.__dict__.items():
                file.write(f'{value:<15} #{parameter}\n')


def call_gnuplot(
    data_file_path: str,
    output_file_path: str
) -> None:
    """
    Call gnuplot to generate a plot of the g3 data.

    :param data_file_path: path to the data file.
    :return: None
    """
    gnuplot_script = f"""
    reset

    dpi = 300 ## dpi (variable)
    width = 90 ## mm (variable)
    height = 90 ## mm (variable)
    in2mm = 25.4 # mm (fixed)
    pt2mm = 0.3528 # mm (fixed)

    mm2px = dpi/in2mm
    ptscale = pt2mm*mm2px
    round(x) = x - floor(x) < 0.5 ? floor(x) : ceil(x)
    wpx = round(width * mm2px)
    hpx = round(height * mm2px)

    set pm3d map
    set cbrange [0:1.5]
    load 'scripts/g3/magma.pal'
    set yrange [1:-1]
    set xrange [2:12]
    set cbtics ('0.0' 0, 'Max' 1.5)

    set output "{output_file_path}"
    set terminal pngcairo size wpx, hpx fontscale ptscale linewidth ptscale pointscale ptscale

    splot "{data_file_path}" notitle

    replot
    unset output
    unset terminal
    """

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.plt') as temp:
        temp.write(gnuplot_script)

    subprocess.run(['gnuplot', temp.name])