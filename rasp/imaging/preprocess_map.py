"""Batch preprocessing functions for Raman maps."""
import numpy as np	
import ramanspy as rp
from _config import config_figure
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter  # filters for outlier correction

def preprocess_maps(images, region=(0, 1800), win_len=9):
    """Apply the same pipeline to a list of spectral images."""
    routine = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=region),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=15),
        rp.preprocessing.denoise.SavGol(window_length=win_len, polyorder=3),
        rp.preprocessing.baseline.ASLS(),
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


def correct_outliers(array: np.ndarray, 
                     method: str = 'median',
                     low_pct: float = 2,
                     high_pct: float = 98) -> np.ndarray:
    """
    Replace detected outliers in a 2D array with locally filtered values.

    :param array: Input 2D array.
    :type array: np.ndarray
    :param method: Filtering method, 'median' or 'mean'.
    :type method: str
    :return: Array with outliers corrected.
    :rtype: np.ndarray
    """

    lo, hi = np.percentile(array, [low_pct, high_pct])
    clipped = np.clip(array, lo, hi)

    mask = detect_outliers(clipped)
    corrected = clipped.copy()

    if method == 'median':
        filtered = median_filter(clipped, size=3)
    elif method == 'mean':
        filtered = uniform_filter(clipped, size=3)
    else:
        raise ValueError("Invalid method. Choose 'median' or 'mean'.")

    corrected[mask] = filtered[mask]

    return corrected


def correct_shading(map_2d: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """
    Corrige efeito de shading/foco num mapa 2D qualquer (banda ou topografia).

    Parameters
    ----------
    map_2d : np.ndarray
        Mapa 2D (topografia ou banda)
    sigma : float
        Largura do filtro Gaussiano para suavizar o background

    Returns
    -------
    corrected : np.ndarray
        Mapa corrigido
    """
    
    def adjust_contrast(img, lower=1, upper=99):
        """Mapeia os percentis [lower, upper] para [0,1]"""
        p_low, p_high = np.percentile(img, (lower, upper))
        return np.clip((img - p_low) / (p_high - p_low + 1e-12), 0, 1)

    background = gaussian_filter(map_2d, sigma=sigma) + 1e-12 
    
    # return adjust_contrast(map_2d / background)
    return map_2d / background


def get_sum(image: rp.SpectralImage):
    """Return topographic intensity map (sum or max)."""

    return np.sum(image.spectral_data, axis=-1)


def plot_histogram(
        data, bins: int = 50,
        title: str = "Pixel Value Histogram",
        figsize: tuple = (9, 5),
        save = None ) -> None:
    """
    Plot a histogram of pixel values from a 2D map (topography, band, etc).

    :param data: Either a 2D numpy array or a SpectralImage.
    :type data: np.ndarray or rp.SpectralImage
    :param bins: Number of histogram bins.
    :type bins: int
    :param title: Title of the histogram plot.
    :type title: str
    :param figsize: Figure size in inches.
    :type figsize: tuple
    """
    # se for SpectralImage, extrai o 2D via sum_intensity

    if hasattr(data, "spectral_data"):
        arr2d = get_sum(data, method='mean')
    else:
        arr2d = data

    ax = plt.subplots(figsize=figsize)

    ax.set_title(title, color='k')
    ax.set_xlabel("Normalized intensity", color='k')
    ax.set_ylabel("Count", color='k')
    ax.set_xlim((0, 1))
    ax.tick_params(colors='k')
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    ax.hist(arr2d.flatten(), bins=bins, color='slategrey', alpha=0.8, rwidth=0.9)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300)
