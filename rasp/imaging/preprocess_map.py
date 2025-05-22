"""Batch preprocessing functions for Raman maps."""
import numpy as np	
import ramanspy as rp
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter  # filters for outlier correction

def preprocess_maps(images, region=(40, 1800), win_len=15):
    """Apply the same pipeline to a list of spectral images."""
    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=region),
        rp.preprocessing.denoise.SavGol(window_length=win_len, polyorder=3),
        # rp.preprocessing.baseline.ASLS(),
        # rp.preprocessing.normalise.MinMax()
    ])
    return [routine.apply(img) for img in images]


def detect_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Identify outliers using Z-score thresholding.

    :param data: Input numeric array.
    :type data: np.ndarray
    :param threshold: Z-score threshold multiplier.
    :type threshold: float
    :return: Boolean mask indicating outliers.
    :rtype: np.ndarray
    """

    mean = np.nanmean(data)
    std = np.nanstd(data)

    return np.abs(data - mean) > threshold * std


def correct_outliers(array: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Replace detected outliers in a 2D array with locally filtered values.

    :param array: Input 2D array.
    :type array: np.ndarray
    :param method: Filtering method, 'median' or 'mean'.
    :type method: str
    :return: Array with outliers corrected.
    :rtype: np.ndarray
    """

    mask = detect_outliers(array)
    corrected = array.copy()

    if method == 'median':
        filtered = median_filter(array, size=3)

    elif method == 'mean':
        filtered = uniform_filter(array, size=3)

    else:
        raise ValueError("Invalid method. Choose 'median' or 'mean'.")

    corrected[mask] = filtered[mask]

    return corrected


def sum_intensity(image: rp.SpectralImage):
    """Return topographic intensity map (sum or max)."""

    sum = np.sum(image.spectral_data, axis=-1)
    return correct_outliers(sum, method='mean')
