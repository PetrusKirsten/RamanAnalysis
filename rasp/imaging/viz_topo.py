"""Topography visualisation."""

import matplotlib.pyplot as plt
from _config import config_figure, config_bar, scale_ticks
from preprocess_map import get_sum, correct_outliers

def plot_topography(img, cmap='bone', title='Topography', save=None):
    
    ax = config_figure(title, (2000, 2000))
    topo = correct_outliers(get_sum(img))
    
    scale_ticks(ax)
    im = ax.imshow(topo, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    config_bar(cbar)
    
    if save:
        plt.savefig(save, dpi=300)
