
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks

from _config import config_figure

def plot_mean_spectrum(image, title="Mean Spectrum of Map", save=None):
    
    """
    Plot the mean spectrum from a Raman spectral image.

    Parameters
    ----------
    image : ramanspy.SpectralImage
        The Raman spectral map.
    title : str, optional
        Title for the plot.
    save : str or Path, optional
        If provided, the figure is saved to this path.
    """
    
    # data manipulation
    prominence, n_peaks = .02, 5
    shifts = image.spectral_axis
    mean_spectrum = np.mean(image.spectral_data, axis=(0, 1))
    var_spectrum = np.var(image.spectral_data, axis=(0, 1))
    peaks, properties = find_peaks(
        var_spectrum,
        prominence=prominence * np.max(var_spectrum)
    )
    sorted_peaks = peaks[np.argsort(var_spectrum[peaks])[::-1]][:n_peaks]

    # figure and plots configs
    ax1 = config_figure(fig_title='', size=(2*1920, 2*1080),
                        face="#FFFFFF", edge="#383838",)
    ax1.set_aspect('auto')
    ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax1.tick_params(axis='x', colors='#383838', direction='out', length=4, width=.75, pad=4)

    color1 = "darkorange"
    ax1.set_ylabel("Mean Intensity", color=color1); ax1.set_ylim((0, 1.25*np.max(mean_spectrum)))
    ax1.tick_params(axis='y', labelcolor=color1, direction='out', length=0, width=.0, pad=4)
    ax1.plot(shifts, mean_spectrum, 
             label="Mean Spectrum",
             color=color1, lw=1., alpha=.75)

    ax2 = ax1.twinx() 
    color2 = "mediumslateblue"
    ax2.set_ylabel("Variance", color=color2); ax2.set_ylim((0, 1.25*np.max(var_spectrum)))
    ax2.tick_params(axis='y', labelcolor=color2, direction='out', length=0, width=.0, pad=4)
    ax2.plot(shifts, var_spectrum, 
             label="Variance Spectrum",
             color=color2, lw=.75, linestyle='--', alpha=.75)

    for p in sorted_peaks:
        shift = shifts[p]
        var_value = var_spectrum[p]

        ax2.scatter(shift, var_value, color=color2, s=40, zorder=5)
        ax2.annotate(
            f"{int(shift)}" + " cm$^{-1}$", fontsize=9,
            xy=(shift, var_value), xytext=(0, 10), textcoords="offset points", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color2, lw=0.8)
        )
    
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
        plt.close()
