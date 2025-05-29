
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
    prominence, n_peaks = .02, 8
    shifts = image.spectral_axis
    mean_spectrum = np.mean(image.spectral_data, axis=(0, 1))
    var_spectrum = np.var(image.spectral_data, axis=(0, 1))
    peaks, properties = find_peaks(
        var_spectrum,
        prominence=prominence * np.max(var_spectrum)
    )
    sorted_peaks = peaks[np.argsort(var_spectrum[peaks])[::-1]][:n_peaks]

    # figure and plots configs
    ax1 = config_figure(fig_title='', size=(1.5*1920, 1.5*1080),
                        face="#FFFFFF", edge="#383838",)
    ax1.set_aspect('auto')
    ax1.set_xlabel("Raman Shift (cm$^{-1}$)"); ax1.set_xlim((np.min(shifts), np.max(shifts)))
    ax1.tick_params(axis='x', colors='#383838', direction='out', length=4, width=.75, pad=4)

    color1 = "#036CCE"
    ax1.set_ylabel("Mean Intensity", color=color1)
    min_int, max_int = np.min(mean_spectrum), np.max(mean_spectrum) 
    ax1.set_ylim((min_int, 2*max_int))
    ax1.tick_params(axis='y', labelcolor=color1, direction='out', length=0, width=.0, pad=4)
    ax1.plot(shifts, mean_spectrum, label="Mean Spectrum",
             color=color1, lw=.75, alpha=.75,
             zorder=3)

    ax2 = ax1.twinx() 
    color2 = "#E21D48"
    ax2.set_ylabel("Variance", color=color2)
    min_var, max_var = np.min(var_spectrum), np.max(var_spectrum) 
    ax2.set_ylim((min_var - .5*max_var, 1.1*max_var))
    ax2.tick_params(axis='y', labelcolor=color2, direction='out', length=0, width=.0, pad=4)
    ax2.plot(shifts, var_spectrum, label="Variance Spectrum",
             color=color2, lw=.75, alpha=.75, linestyle='--',
             zorder=1)

    for p in sorted_peaks:
        shift = shifts[p]
        var_value = var_spectrum[p]

        ax2.scatter(shift, var_value, marker='|', 
                    color=color2, s=10, alpha=.5, 
                    zorder=3)
        ax2.annotate(   
            f"{int(shift)}"+" cm$^{-1}$", fontsize=9,
            xy=(shift, var_value), xytext=(0, 10), textcoords="offset points", ha="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec='w', lw=0),
            zorder=2
        )
    
    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
        plt.close()
