"""Topography visualisation."""

import matplotlib.pyplot as plt

from _config        import config_figure, config_bar, scale_ticks
from preprocess_map import get_sum, correct_outliers, correct_shading


def plot_topography(img, title='Topography', save=None,
                    correct_outliers_on=False, correct_shading_on=False):
    
    ax = config_figure(title, (2000, 2000))
    
    topo = get_sum(img)
    if correct_outliers_on:
        topo = correct_outliers(topo)
    if correct_shading_on:
        topo = correct_shading(topo)
    
    im = ax.imshow(topo, cmap='pink')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    config_bar(cbar)
    scale_ticks(ax)
    
    if save:
        plt.savefig(save, dpi=300)
        plt.close()
