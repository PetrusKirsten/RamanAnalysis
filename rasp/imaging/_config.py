"""
Shared plotting & utility helpers for imaging modules.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def set_font(font_path: str):

    """
    Set a custom font globally in matplotlib.

    Parameters
    ----------
    font_path : str
        Path to the font file (.ttf or .otf).
    """

    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()

    plt.rcParams.update({

        'axes.facecolor':   'w',
        'figure.facecolor': 'w',
        'axes.edgecolor':   'k',
        'axes.linewidth':   0.75,
        'xtick.color':      'k',
        'ytick.color':      'k',

        'font.family':      font_name,
        'text.color':       'k',
        'axes.labelcolor':  'k',
        'font.size':        14,
        'axes.titlesize':   14,
        'axes.labelsize':   14,
        'xtick.labelsize':  12,
        'ytick.labelsize':  12,
        'legend.fontsize':  12,

        'savefig.dpi':      300,

    })


def scale_ticks(ax,
                points_x:  int   = 100,    # pixels in X
                lines_y:   int   = 100,    # pixels in Y
                width_um:  float = 200.0,  # µm in X
                height_um: float = 200.0,  # µm in Y
                n_ticks:   int   = 5) -> None:
    """
    Configure X/Y axis ticks by converting pixel indices into micrometers.

    :param ax: Matplotlib Axes to configure.
    :type ax: matplotlib.axes.Axes
    :param points_x: Total number of pixels along the X axis.
    :type points_x: int
    :param lines_y: Total number of pixels along the Y axis.
    :type lines_y: int
    :param width_um: Physical scan width in micrometers (X direction).
    :type width_um: float
    :param height_um: Physical scan height in micrometers (Y direction).
    :type height_um: float
    :param n_ticks: Number of tick marks to display.
    :type n_ticks: int
    """
    import numpy as _np

    # generate equally spaced pixel positions
    x_pix = _np.linspace(0, points_x - 1, n_ticks)
    y_pix = _np.linspace(0, lines_y  - 1, n_ticks)

    # convert pixel positions to micrometers
    x_um = _np.linspace(0, width_um,  n_ticks)
    y_um = _np.linspace(0, height_um, n_ticks)

    # apply ticks and labels
    ax.set_xticks(x_pix); ax.set_xticklabels([f"{x:.0f}" for x in x_um])
    ax.set_yticks(y_pix); ax.set_yticklabels([f"{y:.0f}" for y in y_um[::-1]])  # reverse only the ticklabels

    # label axes in micrometers
    ax.set_xlabel("x (µm)", color='whitesmoke', weight='bold')
    ax.set_ylabel("y (µm)", color='whitesmoke', weight='bold')


def config_bar(colorbar) -> None:
    """
    Style colorbar ticks and label with white color.

    :param colorbar: Matplotlib Colorbar instance.
    :type colorbar: matplotlib.colorbar.Colorbar
    """

    colorbar.ax.yaxis.set_tick_params(color='whitesmoke')
    colorbar.ax.tick_params(color='#09141E', labelcolor='whitesmoke')
    colorbar.outline.set_edgecolor("#09141E")
    # colorbar.set_label('', color='whitesmoke', weight='bold', labelpad=8)


def config_figure(fig_title: str,
                  size: tuple,
                  face: str = '#09141E',
                  edge: str = "#09141E") -> plt.Axes:
    
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
    ax.tick_params(colors='whitesmoke', direction='out', length=0, width=0, pad=4)
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
