"""
27.06.23
@tcnicholas
Tasks for the main analysis.
"""


import numpy as np
import pandas as pd


def write_orientational_data(
    filename: str,
    rs: np.ndarray,
    correlation: np.ndarray,
    standard_error_of_mean: np.ndarray,
):
    """
    Write the orientational correlation function data to a file.

    :param filename: name of the file to write to.
    :param rs: array of distances.
    :param correlation: array of correlation values.
    :param standard_error_of_mean: array of standard error of mean values.
    :return: None.
    """

    # gather data in dictionary.
    data = {
        'r': rs,
        'phi': correlation,
        'sem': standard_error_of_mean,
    }

    # write to csv (via pandas).
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)