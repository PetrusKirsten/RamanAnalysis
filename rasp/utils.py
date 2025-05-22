import os
import numpy as np
import pandas as pd
import ramanspy as rp


def combine_spectra(spectra_list: list) -> rp.Spectrum:

    """
    Combine multiple RamanSPy Spectra by point-wise average.

    Assumes all spectra have identical spectral_axis arrays.

    Parameters
    ----------
    spectra_list : list of rp.Spectrum
        List of RamanSPy Spectra to combine.

    Returns
    -------
    combined_spectrum : rp.Spectrum
        Combined Spectrum object.
    """

    shifts = [spectrum.spectral_axis for spectrum in spectra_list]
    intensities = [spectrum.spectral_data for spectrum in spectra_list]

    # Check all axes are equal
    for i in range(1, len(shifts)):

        if not np.allclose(shifts[0], shifts[i], atol=1e-4):
            raise ValueError("Spectral axes are not identical between spectra.")

    mean_intensity = np.mean(intensities, axis=0)

    return rp.Spectrum(mean_intensity, shifts[0])
