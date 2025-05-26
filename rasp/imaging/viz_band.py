"""Band extraction & plotting."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from _config import config_figure, normalize
from preprocess_map import get_sum

def extract_band(img, center, width):
    mask = (img.spectral_axis >= center-width) & (img.spectral_axis <= center+width)
    return np.sum(img.spectral_data[..., mask], axis=-1)

def plot_band(img, center, width, title=None, compensation=None, sigma=0.9, save=None):
    band = extract_band(img, center, width)

    if compensation == 'diff':
        topo = get_sum(img)
        diff = band - topo
        mask = topo > np.percentile(topo, 5)
        diff[~mask] = 0
        band = gaussian_filter(diff, sigma=sigma)
    
    band_norm = normalize(band)
    title = title or f'Band {center}±{width} cm$^{-1}$'
    ax = config_figure(title)
    ax.imshow(band_norm, cmap='inferno')
    ax.axis('off')
    if save:
        plt.savefig(save, dpi=300)
