import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

from _config        import config_figure, scale_ticks
from preprocess_map import correct_shading

def plot_kmeans(img: rp.SpectralImage,
                     n_clusters: int = 3,
                     shading: bool = True,
                     sigma: float = 6.0,
                     figsize: tuple = (2000, 2000),
                     save: Path = None):
    """
    Plota o mapa de clusters do k-means para um mapa Raman.

    Parameters
    ----------
    img : rp.SpectralImage
        O mapa Raman.
    n_clusters : int
        Número de clusters para k-means.
    shading : bool
        Aplica shading correction antes do clustering (opcional).
    sigma : float
        Sigma do filtro Gaussiano (usado no shading correction).
    figsize : tuple
        Tamanho da figura em pixels.
    save : Path ou str, opcional
        Caminho para salvar o plot.
    """

    spectral_data = img.spectral_data.copy()
    if shading:
        spectral_data = correct_shading(spectral_data)
        # topo = np.sum(spectral_data, axis=-1)
        # shading_map = gaussian_filter(topo, sigma=sigma) + 1e-12
        # spectral_data = spectral_data / shading_map[..., None]  # corrige cada pixel espectralmente

    spec_mat = spectral_data.reshape(-1, spectral_data.shape[-1])

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(spec_mat)
    labels = km.labels_.reshape(img.spectral_data.shape[:-1])

    cmap = cm.get_cmap('tab10', n_clusters)

    ax = config_figure(f'k-means (k={n_clusters})', figsize, face='#09141E', edge='#09141E')
    ax.imshow(labels, cmap=cmap, interpolation='nearest')
    scale_ticks(ax)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, transparent=True)
        plt.close()
