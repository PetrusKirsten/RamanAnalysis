"""Band extraction & plotting."""
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage  import gaussian_filter

from preprocess_map import correct_outliers, correct_shading
from _config        import config_figure, config_bar, scale_ticks


def extract_band(img, center, width):
    mask = (img.spectral_axis >= center - width) & (img.spectral_axis <= center + width)

    return np.sum(img.spectral_data[..., mask], axis=-1)


def plot_band(img, center, width, title=None, save=None,
              correct_outliers_on=False, correct_shading_on=False):

    ax = config_figure(title or f'Band {center} ± {width}'+' '+'cm$^{-1}$', (2000, 2000))

    band = extract_band(img, center, width)
    if correct_outliers_on:
        band = correct_outliers(band)
    if correct_shading_on:
        band = correct_shading(band, sigma=7.0)

    im = ax.imshow(band, cmap='magma')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    config_bar(cbar)
    scale_ticks(ax)

    if save:
        plt.savefig(save, dpi=300)
