"""
06.06.23
@tcnicholas
Parse LAMMPS log file.
"""


import io
import copy
from typing import List, Tuple, Dict

import ase
import numpy as np


block_int_fmt = ['step']
global_tags = ['units', 'atom_style', 'pair_style', 'timestep']
local_tags = ['variable', 'fix', 'compute']


class LogBlock:
    """
    Store all thermo outputs form this log file simulation block.
    """
    def __init__(self, number, headers, data, extra_info=None):
        """
        """
        self._id = number
        self._headers = [h.lower() for h in headers]
        self._data = self.__fmt_data__(data)
        self._info = extra_info
        
    
    @property
    def id(self):
        """
        Block ID.
        """
        return self._id
        
    
    @property
    def headers(self) -> List[str]:
        """
        The headers saved for this simulation block.
        """
        return self._headers
    
    
    @property
    def steps_range(self) -> Tuple[int]:
        """
        The start and end step counts.
        """
        return self.get_column('step')[[0,-1]]
    
    
    @property
    def time(self) -> float:
        """
        Time (seconds) spent on this stage.
        """
        return self.get_column('cpu')[-1]
    
    
    def get_column(self, property_name: str) -> np.ndarray:
        """
        Get the data corresponding to a given property.
        """
        ix = self._headers.index(property_name.lower())
        return self._data[:,ix]


    def __fmt_data__(self, data):
        data = np.array(data, dtype=object)
        int_col_ix = [
            i for i,h in enumerate(self._headers) if h in block_int_fmt
        ]
        float_col_ix = [
            i for i,h in enumerate(self._headers) if h not in block_int_fmt
        ]
        data[:, int_col_ix] = data[:, int_col_ix].astype(int)
        data[:, float_col_ix] = data[:, float_col_ix].astype(float)
        return data
        
    
    def __len__(self):
        """
        Number of recorded steps.
        """
        return len(self._data)
    
    
    def __repr__(self):
        """
        Pretty formatting.
        """
        step_fmt = '->'.join((str(x) for x in self.steps_range))
        return f"LogBlock(ID={self._id}, steps=[{step_fmt}])"


def parse_lammps_log(filename):
    """
    Read and parse LAMMPS log file.
    
    Work under the assumption that thermo output.
    """
    blocks = []
    block_id = 1
    in_block = False
    global_info = dict()
    local_info = {x:{} for x in local_tags}
    with open(filename, 'r') as f:
        this_block = []
        for i, line in enumerate(f):
            line = line.strip()
            
            # beginning of a simulation run block.
            if line.lower().startswith('step'):
                this_block_headers = line.split()
                in_block = True
            
            # end of a simulation run block.
            elif line.lower().startswith('loop time of '):
                
                blocks.append(
                    LogBlock(
                        block_id,
                        this_block_headers,
                        this_block,
                        copy.deepcopy(local_info)
                    )
                )
                
                this_block = []
                block_id += 1
                in_block = False
                
            # store all lines during a simulation block.
            elif in_block:
                this_block.append(line.split())
                
            # global tags are all formatted as key:value pairs.
            elif any([line.startswith(x) for x in global_tags]):
                split = line.split()
                global_info[split[0].strip()] = split[1].strip()
                
            elif any([line.startswith(x) for x in local_tags]):
                split = line.split()
                local_type = split[0].strip()
                local_name = split[1].strip()
                local_type_type = split[2].strip()
                local_details = " ".join(split[3:])
                local_info[local_type][local_name] = {
                    'kind':local_type_type,
                    'details': local_details
                }
    
    return blocks, global_info


class LammpsLogFile:
    
    def __init__(self, filename):
        """
        :param filename: path to LAMMPS log file.
        """
        self._data, self._info = self.__parse__(filename)
        
    
    def gather_property(self,
        property_name: str,
        data_start=0,
        data_end=None,
        col_by_temp=False,
    ):
        """
        Gather property across all simulation blocks.
        """

        data_end = len(self._data) if data_end is None else data_end
        steps = [block.get_column('step') for block in self._data[data_start:data_end]]
        prop = [block.get_column(property_name.lower()) for block in self._data[data_start:data_end]]

        result = prop[0]
        lengths = []
        for i in range(1, len(prop)):
            if steps[i][0] == steps[i-1][-1]:
                result = np.concatenate((result, prop[i][1:]))
                lengths.append(len(prop[i][1:]))
            else:
                result = np.concatenate((result, prop[i]))
                lengths.append(len(prop[i]))

        if col_by_temp:
            return result, lengths
        
        return result
        
        
    @property
    def data(self) -> List[LogBlock]:
        """
        Return all data.
        """
        return self._data
        
        
    @property
    def info(self) -> Dict:
        """
        Return extracted information.
        """
        return self._info
        
    
    @property
    def all_headers(self) -> List[str]:
        """
        Search log blocks for all possible headers. Note, this doesn't check if
        the header is present in all log blocks.
        """
        return list({h for block in self._data for h in block.headers})
    
    
    def __parse__(self, filename):
        """
        Gather all info and blocks of data from file.
        """
        return parse_lammps_log(filename)
        
        
    def __repr__(self) -> str:
        """
        Pretty formatting.
        """
        return f"LammpsLogFile(nblocks={len(self._data)})"
        

def get_dump_header(atoms: ase.Atoms, timestep: int = 0) -> str:
    """
    Return the header for the LAMMPS dump file.

    :param atoms: ASE atoms object.
    :return: header string.
    """
    cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
    return f"""ITEM: TIMESTEP
{timestep}
ITEM: NUMBER OF ATOMS
{len(atoms)}
ITEM: BOX BOUNDS pp pp pp
0 {cell_lengths[0]}
0 {cell_lengths[1]}
0 {cell_lengths[2]}
ITEM: ATOMS id type xu yu zu
"""


def make_lammps_dump_string(
    atoms: ase.Atoms,
    timestep: int = 0,
) -> str:
    """
    Prepare structure input file for g3 calculation. We need to format the files
    in LAMPS dump format.

    :param filename: output filename.
    :param atoms: ASE atoms object.
    :return: None.

    Notes
    -----
    The g3 code is formatted to read a standard LAMMPS dump conifguration file.
    So there are 5 expected columns: [ atom_id, atom_type, x, y, z ]

    Currently only orthorhombic cells are supported.

    The cooordiantes should be in the same units as the box vectors.
    """

    # get dump header.
    dump_file_string = get_dump_header(atoms, timestep)

    # then prepare the atoms string.
    atoms.wrap()
    coords = atoms.get_positions()
    indices = np.arange(len(atoms)) + 1
    data = np.vstack((indices, np.ones(len(atoms)), coords.T)).T

    # write to the StringIO object as if it was a file and retrieve the string
    output = io.StringIO()
    np.savetxt(output, data, fmt=['%.0f', '%.0f', '%.8f', '%.8f', '%.8f'])
    dump_file_string += output.getvalue()

    return dump_file_string