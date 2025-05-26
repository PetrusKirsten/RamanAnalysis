"""Band extraction & plotting."""
import numpy as np
import matplotlib.pyplot as plt

from preprocess_map import correct_outliers
from _config        import config_figure, config_bar, scale_ticks


def extract_band(img, center, width):
    mask = (img.spectral_axis >= center - width) & (img.spectral_axis <= center + width)

    return np.sum(img.spectral_data[..., mask], axis=-1)

def plot_band(img, center, width, title=None, save=None):

    ax = config_figure(title or f'Band {center} ± {width}'+' '+'cm$^{-1}$', (2000, 2000))

    band = extract_band(img, center, width)
    band_corrected = correct_outliers(band)

    im = ax.imshow(band_corrected, cmap='magma')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    config_bar(cbar)
    scale_ticks(ax)

    if save:
        plt.savefig(save, dpi=300)
