"""Topography visualisation."""

import matplotlib.pyplot as plt
from _config import config_figure, normalize
from preprocess_map import get_sum, plot_histogram

def plot_topography(img, cmap='bone', title='Topography', save=None):
    
    ax = config_figure(title, (3200, 3000))
    topo = get_sum(img)
    
    ax.imshow(topo, cmap=cmap)
    ax.axis('off')
    
    if save:
        plt.savefig(save, dpi=300)
