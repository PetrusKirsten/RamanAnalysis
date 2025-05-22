"""I/O utilities to read Raman map text files into ramanspy.SpectralImage."""
import re                                # regular expressions for parsing column names
import numpy as np
import ramanspy as rp
import pandas as pd
from pathlib import Path

def parse_coordinates(column_names: list) -> list:
    """
    Extract (x, y) coordinate tuples from column names formatted as '... (x/y)'.

    :param column_names: List of column header strings.
    :type column_names: list
    :return: List of (x, y) integer tuples.
    :rtype: list
    """

    coords = [re.search(r"\((\d+)/(\d+)\)", name) for name in column_names]

    return [(int(c.group(1)), int(c.group(2))) for c in coords if c]


def load_file(path: str | Path) -> rp.SpectralImage:
    """Load a single `.txt` map exported by XYZ instrument.

    Parameters
    ----------
    path : str or Path
        Path to the text file.

    Returns
    -------
    image : rp.SpectralImage
        Ramanspy spectral image object.
    """
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    raman_shift = df.iloc[:, 0].values
    spectra_columns = df.columns[1:]

    xy = parse_coordinates(spectra_columns)
    if not xy:
        raise ValueError("No (x/y) coordinates found; check file format.")

    max_x = max(x for x, y in xy) + 1
    max_y = max(y for x, y in xy) + 1
    data_cube = np.zeros((max_y, max_x, len(raman_shift)))

    for i, (x, y) in enumerate(xy):
        data_cube[y, x, :] = df.iloc[:, i + 1].values

    return rp.SpectralImage(data_cube, raman_shift)

