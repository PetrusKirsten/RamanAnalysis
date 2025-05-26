"""Topography visualisation."""

import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from _config        import config_figure, config_bar, scale_ticks
from preprocess_map import get_sum, correct_outliers


def correct_shading(map_2d: np.ndarray, sigma: float = 7.0) -> np.ndarray:
    """
    Corrige efeito de shading/focal bias num mapa 2D qualquer (banda ou topografia).

    Parameters
    ----------
    map_2d : np.ndarray
        Mapa 2D (topografia ou banda integrada)
    sigma : float
        Largura do filtro gaussiano para suavizar o background

    Returns
    -------
    corrected : np.ndarray
        Mapa corrigido
    """
    
    background = gaussian_filter(map_2d, sigma=sigma) + 1e-12 

    return map_2d / background


def plot_topography(img, title='Topography', save=None):
    
    ax = config_figure(title, (2000, 2000))
    
    topo = get_sum(img)
    topo_corrected = correct_outliers(topo)
    # topo_corrected = correct_shading(correct_outliers(topo))
    
    im = ax.imshow(topo_corrected, cmap='pink')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    config_bar(cbar)
    scale_ticks(ax)
    
    if save:
        plt.savefig(save, dpi=300)
