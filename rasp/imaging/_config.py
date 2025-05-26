"""Shared plotting & utility helpers for imaging modules."""
import matplotlib.pyplot as plt
import numpy as np

def config_figure(fig_title: str,
                  size: tuple,
                  face: str = '#09141E',
                  edge: str = 'k') -> plt.Axes:
    
    """
    Create a styled Matplotlib Axes with specified background and edge colors.

    :param fig_title: Title text for the figure.
    :type fig_title: str
    :param size: Tuple specifying figure size in pixels (width, height).
    :type size: tuple
    :param face: Background color.
    :type face: str
    :param edge: Edge color for axes spines.
    :type edge: str
    :return: Configured Matplotlib Axes.
    :rtype: plt.Axes
    """

    dpi = 300
    w, h = size[0] / dpi, size[1] / dpi

    fig, ax = plt.subplots(figsize=(w, h), facecolor=face)
    ax.set_facecolor(face)

    ax.set_title(fig_title, color='whitesmoke', weight='bold', pad=12)
    ax.tick_params(colors='whitesmoke', direction='out', length=0, width=0, pad=2)
    ax.set_aspect('equal')
    ax.grid(False)  # remover grades de fundo, se usar 'whitegrid' pode manter leves
    for spine in ax.spines.values():
        spine.set_edgecolor(edge)
        spine.set_linewidth(.75)

    return ax

def normalize(arr, vmin=None, vmax=None):
    vmin = np.min(arr) if vmin is None else vmin
    vmax = np.max(arr) if vmax is None else vmax
    return (arr - vmin) / (vmax - vmin + 1e-12)
