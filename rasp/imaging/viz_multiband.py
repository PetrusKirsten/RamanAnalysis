
import numpy as np
import ramanspy as rp
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from _config        import config_figure, scale_ticks
from viz_topo       import get_sum
from viz_band       import extract_band


def apply_alpha_mask(rgb: np.ndarray,
                     channel_thresholds: tuple[float, float, float]) -> np.ndarray:
    """
    Dada uma imagem RGB (h×w×3) e thresholds por canal,
    zera (torna invisível) os canais cujos valores estão abaixo do threshold,
    pixel a pixel.

    Parameters
    ----------
    rgb : ndarray (h, w, 3)
        Imagem RGB com valores entre 0–1.
    channel_thresholds : (thr_r, thr_g, thr_b)
        Threshold individual para cada canal.

    Returns
    -------
    rgba : ndarray (h, w, 4)
        Imagem RGBA onde canais abaixo do threshold somem visualmente.
    """
    thr_r, thr_g, thr_b = channel_thresholds

    rgb_masked = rgb.copy()
    rgb_masked[..., 0][rgb[..., 0] < thr_r] = 0.0  # Red
    rgb_masked[..., 1][rgb[..., 1] < thr_g] = 0.0  # Green
    rgb_masked[..., 2][rgb[..., 2] < thr_b] = 0.0  # Blue

    alpha = (np.max(rgb_masked, axis=-1) > 0).astype(float)
    rgba = np.dstack((rgb_masked, alpha))

    return rgba


def plot_multiband_rgb(
    image,
    bands,
    idx_rgb=(0, 1, 2),
    title="Multiband RGB Map",
    shading=True,
    contrast=False,
    sigma=6.0,
    weights=(1.0, 1.0, 1.0),
    alpha_mask=True,
    alpha_threshold=0.35,
    contours=False,
    save=None
):
    """
    Gera e plota um mapa RGB combinando 3 bandas Raman, com opções de
    shading correction, contraste, pesos por banda, máscara de transparência e contornos.

    Parameters
    ----------
    image           : ramanspy.SpectralImage
    bands           : list of (center, width, label)
    idx_rgb         : tuple of (R, G, B)
    title           : str
    shading         : bool
    contrast        : bool
    sigma           : float
    weights         : tuple of (R, G, B)
    alpha_mask      : bool
    alpha_threshold : float
    contours        : bool
    save            : str ou Path
    """

    centers = [bands[i][0] for i in idx_rgb]
    widths  = [bands[i][1] for i in idx_rgb]

    layers = [extract_band(image, c, w) for c, w in zip(centers, widths)]

    if shading:
        topo = get_sum(image)
        shading_map = gaussian_filter(topo, sigma=sigma) + 1e-12
        layers = [layer / shading_map for layer in layers]

    if contrast:
        def stretch(arr):
            p1, p99 = np.percentile(arr, (2, 98))
            return np.clip((arr - p1) / (p99 - p1 + 1e-12), 0, 1)
        layers = [stretch(L) for L in layers]

    # Aplicar pesos
    layers = [L * w for L, w in zip(layers, weights)]
    layers = [np.clip(L, 0, 1) for L in layers]

    rgb = np.stack(layers, axis=-1)

    # Alpha mask
    if alpha_mask:
        topo = get_sum(image)
        alpha = (topo > alpha_threshold * np.max(topo)).astype(float)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    else:
        rgba = rgb

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgba)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.axis('off')

    # Contornos
    if contours:
        colors = ["red", "green", "blue"]
        for i, layer in enumerate(layers):
            ax.contour(layer, levels=[0.5], colors=colors[i], linewidths=0.8)

    if save:
        plt.tight_layout()
        plt.savefig(save, dpi=300)
        plt.close()


def plot_multiband(
        image:      rp.SpectralImage,
        bands:      list,
        colors:     list  = None,
        normalize:  str   = 'local',
        mode:       str   = 'continuous',
        thresholds: tuple = (0.1, 0.5, 1.1),
        shading:    bool  = True,
        save              = None
    ) -> None:
    """
    Combine multiple single-band maps into a false-color RGB image.

    :param image: SpectralImage instance.
    :type image: rp.SpectralImage
    :param bands: List of (center, width) tuples for each channel.
    :type bands: list
    :param figsize: Figure size in pixels.
    :type figsize: tuple
    :param method: Outlier correction method.
    :type method: str
    :param colors: List of RGB tuples for each band channel.
    :type colors: list
    :param compensation: 'raw' or 'diff' for topography compensation.
    :type compensation: str
    """

    def stretch(arr):
        """
        Stretch por percentil 2–98.
        
        Parameters
        ----------
        arr : np.ndarray ou list[np.ndarray]
            Array único (modo 'local') ou lista de arrays (modo 'global').
        mode : str
            'local' → normaliza cada banda individualmente.
            'global' → normaliza todas as bandas juntas no mesmo intervalo.

        Returns
        -------
        Array normalizado (ou lista de arrays normalizados no modo global).
        """

        if normalize == 'local':
            stretched = np.empty_like(arr)
            for i in range(arr.shape[-1]):
                p1, p99 = np.percentile(arr[..., i], (2, 98))
                stretched[..., i] = np.clip((arr[..., i] - p1) / (p99 - p1 + 1e-12), 0, 1)

        elif normalize == 'global':
            p1, p99 = np.percentile(arr, (2, 98))
            stretched = np.clip((arr - p1) / (p99 - p1 + 1e-12), 0, 1)

        else:
            raise ValueError("mode must be 'local' or 'global'")

        return stretched  # Sempre retorna array shape (h, w, n_bandas)
        
    chan = [extract_band(image, c, w) for c, w, _ in bands]

    if shading:
        from preprocess_map import correct_shading
        chan = [correct_shading(b) for b in chan]
        # region
        # topo = get_sum(image)
        # shading_map = gaussian_filter(topo, sigma=7.0) + 1e-12
        # chan = [b / shading_map for b in chan]
        # endregion
    
    chan[0] = chan[0] / chan[2]  # ratio of 851 cm-1 band to 478 cm-1
    chan = [stretch(b) for b in chan]
    
    if mode == 'binary':
        for i in range(len(chan)):
            chan[i] = (chan[i] >= thresholds[i]).astype(float)

    # region
    # if compensation == 'diff':
    # topo = sum_intensity(image, method=method)
    # chan = [normalize(np.clip(gaussian_filter(b - topo, sigma=0.9), -0.1, 0.1)) for b in chan]
    # endregion
    if colors is None:
        default = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        colors = default[:len(chan)]

    h, w = chan[0].shape
    rgb = np.zeros((h, w, 3))

    for i, band_img in enumerate(chan):
        cr, cg, cb = colors[i]
        rgb[..., 0] += band_img * cr
        rgb[..., 1] += band_img * cg
        rgb[..., 2] += band_img * cb

    rgb = np.clip(rgb, 0, 1)
    rgba = apply_alpha_mask(rgb, thresholds)

    # composition plot
    plot_composition_pie(rgb=rgba, thresholds=thresholds, strategy='binary', save=save)

    ax = config_figure(
        f"Bands: Red: {bands[0][0]}" + " cm$^{-1}$ " + f"Blue: {bands[-1][0]}" + " cm$^{-1}$",
        size=(2000, 2000))
    ax.imshow(rgba, origin='upper', interpolation='bilinear')
    scale_ticks(ax)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300)
        plt.close()


def plot_composition_pie(rgb, thresholds=(0.2, 0.2, 0.2), strategy='binary', save=None):
    """
    Gera gráfico de pizza com base na composição dos canais RGB.

    Parameters
    ----------
    rgb : np.ndarray (h, w, 3)
        Imagem RGB normalizada entre 0 e 1.
    thresholds : tuple
        Thresholds por canal (R, G, B).
    strategy : str
        'binary' para presença/ausência por pixel, ou
        'intensity' para soma contínua dos valores dos canais.
    """

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    if strategy == 'binary':
        r_mask = (r >= thresholds[0]).astype(int)
        g_mask = (g >= thresholds[1]).astype(int)
        b_mask = (b >= thresholds[2]).astype(int)

        only_r = (r_mask == 1) & (g_mask == 0) & (b_mask == 0)
        only_g = (g_mask == 1) & (r_mask == 0) & (b_mask == 0)
        only_b = (b_mask == 1) & (r_mask == 0) & (g_mask == 0)
        rg     = (r_mask == 1) & (g_mask == 1) & (b_mask == 0)
        rb     = (r_mask == 1) & (b_mask == 1) & (g_mask == 0)
        gb     = (g_mask == 1) & (b_mask == 1) & (r_mask == 0)
        all_3  = (r_mask == 1) & (g_mask == 1) & (b_mask == 1)
        none   = (r_mask == 0) & (g_mask == 0) & (b_mask == 0)

        counts = {
            "Only Red"     : np.sum(only_r),
            # "Only Green"   : np.sum(only_g),
            "Only Blue"    : np.sum(only_b),
            # "Red + Green"  : np.sum(rg),
            "Red + Blue"   : np.sum(rb),
            # "Green + Blue" : np.sum(gb),
            # "All Three"    : np.sum(all_3),
            "None"         : np.sum(none)
        }

    elif strategy == 'intensity':
        counts = {
            "Red (total)"   : np.sum(r),
            "Green (total)" : np.sum(g),
            "Blue (total)"  : np.sum(b)
        }

    else:
        raise ValueError("strategy must be 'binary' or 'intensity'")

    # Plot
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#d80000",  # red     
            #   '#43a047',  # green - not used 
              "#2b00c5",  # blue 
            #   '#fbc02d',  # yellow - not used
              "#e22bd3",  # purple/magenta 
            #   '#00acc1',  # ?
            #   '#6d4c41',  # all three - not used 
              '#9e9e9e']  # none

    ax = config_figure(fig_title=f"Pixel Composition ({strategy.capitalize()})", size=(2000, 2000),
                       face='#FFFFFF')

    ax.pie(values, labels=labels,
           colors=colors[:len(labels)], 
           startangle=90, counterclock=False,
           autopct='%1.1f%%', wedgeprops={'edgecolor': 'white'})
    
    plt.tight_layout()
    if save:
        plt.savefig(save.with_name("composition_" + save.name), dpi=300)
        plt.close()


def plot_composition_bar(rgb, thresholds=(0.2, 0.2, 0.2), strategy='binary'):
    """
    Gera gráfico de barras com base na composição dos canais RGB.

    Parameters
    ----------
    rgb : np.ndarray (h, w, 3)
        Imagem RGB normalizada entre 0 e 1.
    thresholds : tuple
        Thresholds por canal (R, G, B).
    strategy : str
        'binary' para presença/ausência por pixel, ou
        'intensity' para soma contínua dos valores dos canais.
    """

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    if strategy == 'binary':
        r_mask = (r >= thresholds[0]).astype(int)
        g_mask = (g >= thresholds[1]).astype(int)
        b_mask = (b >= thresholds[2]).astype(int)

        only_r = (r_mask == 1) & (g_mask == 0) & (b_mask == 0)
        only_g = (g_mask == 1) & (r_mask == 0) & (b_mask == 0)
        only_b = (b_mask == 1) & (r_mask == 0) & (g_mask == 0)
        rg     = (r_mask == 1) & (g_mask == 1) & (b_mask == 0)
        rb     = (r_mask == 1) & (b_mask == 1) & (g_mask == 0)
        gb     = (g_mask == 1) & (b_mask == 1) & (r_mask == 0)
        all_3  = (r_mask == 1) & (g_mask == 1) & (b_mask == 1)
        none   = (r_mask == 0) & (g_mask == 0) & (b_mask == 0)

        counts = {
            "Only Red"     : np.sum(only_r),
            "Only Green"   : np.sum(only_g),
            "Only Blue"    : np.sum(only_b),
            "Red + Green"  : np.sum(rg),
            "Red + Blue"   : np.sum(rb),
            "Green + Blue" : np.sum(gb),
            "All Three"    : np.sum(all_3),
            "None"         : np.sum(none)
        }

    elif strategy == 'intensity':
        counts = {
            "Red (total)"   : np.sum(r),
            "Green (total)" : np.sum(g),
            "Blue (total)"  : np.sum(b)
        }

    else:
        raise ValueError("strategy must be 'binary' or 'intensity'")

    # Plot
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(7, 5))
    bars = plt.barh(labels, values, color='skyblue')
    plt.xlabel("Value")
    plt.title(f"Pixel Composition ({strategy.capitalize()})")

    for bar in bars:
        w = bar.get_width()
        plt.text(w + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(w)}", va='center', fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300)
        plt.close()
