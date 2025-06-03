import re, os
import numpy as np
import pandas as pd
import ramanspy as rp
import matplotlib.pyplot as plt

from lmfit.models import VoigtModel
from typing import List, Dict, Tuple
from scipy.signal import find_peaks, peak_prominences, peak_widths



def get_peaks(spectrum: rp.Spectrum,
              height: float = None,
              distance: int = None,
              prominence: float = None) -> tuple:
    
    """
    Find peaks in a Raman spectrum.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Spectrum to analyze.
    height : float, optional
        Required height of peaks.
    distance : int, optional
        Minimum horizontal distance (in data points) between peaks.
    prominence : float, optional
        Required prominence of peaks.

    Returns
    -------
    peak_positions : np.ndarray
        Raman Shift positions of the detected peaks.
    peak_intensities : np.ndarray
        Intensities of the detected peaks.
    """
    peaks, props = find_peaks(
        spectrum.spectral_data,
        height=height,
        distance=distance,
        prominence=prominence
    )

    peak_positions = spectrum.spectral_axis[peaks]
    peak_intensities = spectrum.spectral_data[peaks]

    return peak_positions, peak_intensities


def get_band_area(spectrum: rp.Spectrum, min_shift: float, max_shift: float) -> float:

    """
    Calculate the area under a peak between two Raman shifts.

    Parameters
    ----------
    spectrum : rp.Spectrum
        Spectrum to analyze.
    min_shift : float
        Lower bound of integration.
    max_shift : float
        Upper bound of integration.

    Returns
    -------
    area : float
        Area under the curve.
    """

    mask = (spectrum.spectral_axis >= min_shift) & (spectrum.spectral_axis <= max_shift)

    x = spectrum.spectral_axis[mask]
    y = spectrum.spectral_data[mask]

    return np.trapz(y, x)


def extract_band_areas(
    spectra: List, 
    labels: List[Tuple[str, float]],
    bands: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Para cada espectro, calcula a área de cada banda definida.

    Parameters
    ----------
    spectra : list of rp.Spectrum
    labels : list of (group, conc)
    bands : dict
        {'nome da banda': (low_shift, high_shift), ...}

    Returns
    -------
    df : pandas.DataFrame
        Colunas: group, conc, cada banda como coluna de área.
    """
    rows = []
    for spec, (group, conc) in zip(spectra, labels):
        row = {'group': group, 'conc': float(conc)}
        for bname, (low, high) in bands.items():
            row[bname] = get_band_area(spec, low, high)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_band_by_formulation(
    df: pd.DataFrame, 
    band: str, 
    out_folder: str = None,
    save: bool = False
):
    """
    Plota área da banda vs concentração para cada group (St, kC, iC).

    Parameters
    ----------
    df : DataFrame de extract_band_areas()
    band : nome da coluna de banda ex. 'C–O–C (480)'
    out_folder : pasta para salvar
    save : se True, salva arquivo .png
    """
    from plot_utils import config_figure, addLegend

    ax = config_figure(f"{band}", (4*800, 4*600))
    colors = {
        "St": ['#E1C96B', '#FFE138', '#F1A836', '#E36E34'],
        "St kC": ['hotpink', 'mediumvioletred', '#A251C3', '#773AD1'],
        "St iC": ['lightskyblue', '#62BDC1', '#31A887', '#08653A'],
    }
    for group, grp_df in df.groupby('group'):
        # ordena pelo conc
        grp_df = grp_df.sort_values('conc')
        ax.plot(
            grp_df['conc'], grp_df[band], 
            marker='o', color=colors[group][1], linestyle='-',
            lw=.75, markersize=9, mec=colors[group][1], mfc='w', alpha=1.,
            label=group,
        )

    ax.set_xlabel("CaCl$_2$ concentration (mM)")
    ax.set_ylabel(f"Area under peak")
    ax.set_title(f"{band}" + " 1/cm")
    ax.set_xticks([0, 7, 14, 21])
    addLegend(ax)
    plt.tight_layout()

    if save and out_folder:
        plt.savefig(f"{out_folder}/band_{band.replace(' ','_')}.png", dpi=300)
    # # plt.show()


def plot_all_bands(
    df: pd.DataFrame,
    bands: List[str],
    out_folder: str = None,
    save: bool = False
):
    """
    Gera um subplot para cada banda, em uma figura única.
    """
    n = len(bands)
    cols = 2
    rows = (n + 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
    axes = axes.flatten()

    for ax, band in zip(axes, bands):
        for group, grp_df in df.groupby('group'):
            grp_df = grp_df.sort_values('conc')
            ax.plot(
                grp_df['conc'],
                grp_df[band],
                marker='o',
                linestyle='-',
                label=group
            )
        ax.set_title(band)
        ax.set_xlabel("CaCl$_2$ (mM)")
        ax.set_ylabel("Area")
        ax.legend(fontsize=8)
    # remove eixos extras
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    if save and out_folder:
        plt.savefig(f"{out_folder}/all_bands_comparison.png", dpi=300)
    # plt.show()


def extract_band_metrics(
    spectra: List,
    labels: List[Tuple[str, float]],
    bands: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Para cada espectro e cada banda, calcula:
      - área sob pico
      - centroid (posição do pico mais proeminente)
      - FWHM (em cm⁻¹)
    Retorna DataFrame com colunas:
      group, conc,
      area_<banda>, center_<banda>, fwhm_<banda>
    """
    
    rows = []
    for spec, (group, conc) in zip(spectra, labels):
        x, y = spec.spectral_axis, spec.spectral_data        
        
        row = {'group': group, 'conc': float(conc)}
        for name, (low, high) in bands.items():

            # Máscara para região da banda
            mask = (x >= low) & (x <= high)
            x_reg, y_reg = x[mask], y[mask]

            # Área sob pico
            area = get_band_area(spec, low, high)
            row[f'Area at {name} 1/cm'] = area

            # Detectar picos e escolher o mais proeminente
            if len(y_reg) == 0:
                row[f'Center at {name} 1/cm'] = np.nan
                row[f'Center at {name} 1/cm'] = np.nan
                continue

            peaks, props = find_peaks(y_reg, prominence=np.max(y_reg)*0.025)
            if peaks.size == 0:
                row[f'Center at {name} 1/cm'] = np.nan
                row[f'Center at {name} 1/cm'] = np.nan
                continue

            prominences = peak_prominences(y_reg, peaks)[0]
            top_peak = peaks[np.argmax(prominences)]

            # Centroid em cm-1
            center = x_reg[top_peak]
            row[f'Center at {name} 1/cm'] = center

            # FWHM em pontos
            results_half = peak_widths(y_reg, [top_peak], rel_height=0.5)
            width_points = results_half[0][0]

            # Converter pontos para cm-1
            if len(x_reg) > 1:
                delta = np.mean(np.diff(x_reg))
                fwhm = width_points * delta

            else:                
                fwhm = np.nan

            row[f'FWHM at {name} 1/cm'] = fwhm

        rows.append(row)

    return pd.DataFrame(rows)


def compute_ratio(
    df: pd.DataFrame, 
    numerator_band: str, 
    denominator_band: str
) -> pd.DataFrame:
    """
    Adiciona coluna de razão entre duas áreas de bandas.
    """
    
    col_name = f'ratio_{numerator_band}_to_{denominator_band}'
    df[col_name] = df[f'Area at {numerator_band} 1/cm'] / df[f'Area at {denominator_band} 1/cm']
    
    return df


def plot_band_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_folder: str = None,
    save: bool = False
):
    """
    Plota metric vs concentração, separado por group.
    """
    from rasp.plot_utils import config_figure, add_legend
    
    def safe_filename(s: str) -> str:
        name = re.sub(r'[^0-9a-zA-Z\-_]+', '_', s)
        name = re.sub(r'__+', '_', name).strip('_')

        return name

    ax = config_figure(f"{metric}", (3*800, 3*600))
    
    colors = {"St": '#F1A836', "St kC": 'mediumvioletred', "St iC": '#62BDC1'}

    for group, grp_df in df.groupby('group'):
        grp_df_sorted = grp_df.sort_values('conc')
        ax.plot(
            grp_df_sorted['conc'], grp_df_sorted[metric],
            marker='o', markersize=11, mec=colors[group], mfc='w', mew=1.,
            linestyle='-', lw=.75, color=colors[group],  
            label=group
        )
    
    ax.set_title(metric)
    ax.set_xlabel("CaCl$_2$ concentration (mM)"); ax.set_ylabel(ylabel)
    ax.set_xticks([0, 7, 14, 21])
    add_legend(ax)

    plt.tight_layout()
    if save and out_folder:
        plt.savefig(f"{out_folder}/{safe_filename(metric)}.png", dpi=300)
    
    # plt.show()


def plot_all_metrics(
    df: pd.DataFrame,
    bands: List[str],
    out_folder: str = None,
    save: bool = False
):
    """
    Gera subplots para cada banda: área, centro, FWHM, e uma razão exemplar.
    """
    
    metrics = []
    
    for name in bands:
        metrics.extend([f'Area at {name} 1/cm', f'Center at {name} 1/cm', f'Center at {name} 1/cm'])
    
    n = len(metrics)
    cols = 2
    rows = (n + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for group, grp_df in df.groupby('group'):
            grp_df_sorted = grp_df.sort_values('conc')
            ax.plot(
                grp_df_sorted['conc'],
                grp_df_sorted[metric],
                marker='o', linestyle='-',
                label=group
            )
        
        ax.set_title(metric)
        ax.set_xlabel("CaCl$_2$ (mM)")
        ax.set_ylabel(metric)
        ax.set_xticks([0, 7, 14, 21])
        ax.legend(fontsize=8)
    
    for ax in axes[n:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    
    if save and out_folder:
        plt.savefig(f"{out_folder}/all_band_metrics.png", dpi=300)
    
    # plt.show()


def deconvolve_batch(spectra, labels, region, n_peaks, center_targets,
                     save_figs=True, fig_folder="./figures/deconv",
                     save_csv=True, csv_path="./figures/deconv/metrics.csv"):
    """
    Aplica deconvolução a uma lista de espectros e salva resultados.

    Parâmetros:
    - spectra: lista de objetos Spectrum
    - labels: lista de labels (grupo, conc) correspondentes
    - region: tupla (min, max) com a região a analisar
    - n_peaks: número de picos a ajustar
    - save_figs: salva os gráficos se True
    - fig_folder: onde salvar as figuras
    - save_csv: salva o CSV com métricas se True
    - csv_path: caminho para o arquivo CSV de saída

    Retorna:
    - df_result: DataFrame com todas as métricas
    """
    
    def deconvolve_band(spectrum, width_targets=3):
        """
        Deconvolui uma banda Raman em uma região definida usando modelos Voigt.

        Parâmetros:
        - spectrum: objeto Spectrum do RASP
        - region: tupla (min, max) com limites da banda (em cm⁻¹)
        - n_peaks: número de picos a ajustar
        - plot: se True, plota resultado com os picos

        Retorna:
        - DataFrame com parâmetros de cada pico (centro, FWHM, área, amplitude)
        """

        # 1️⃣ Recortar região de interesse
        if 'kC' in label_str:
            print('kC')
        x = spectrum.spectral_axis
        y = spectrum.spectral_data
        mask = (x >= region[0]) & (x <= region[1])
        x_region = x[mask]
        y_region = y[mask]

        # 2️⃣ Construir modelo composto
        model = None
        params = None
        
        for i in range(n_peaks):
            prefix = f"p{i}_"
            peak = VoigtModel(prefix=prefix)
            # center_guess = np.linspace(region[0], region[1], n_peaks)[i]
            center_guess = center_targets[i] if i < len(center_targets) else np.mean(region)
            amp_guess = max(y_region)

            if model is None:
                model = peak
            else:
                model += peak

            p = peak.make_params(
                center=center_guess,
                amplitude=amp_guess,
                sigma=5,
                gamma=1,
            )
            # Adiciona restrição de faixa
            p[f"{prefix}center"].set(min=center_guess - width_targets,
                                     max=center_guess + width_targets)
            
            if params is None:
                params = p
            else:
                params.update(p)

        # 3️⃣ Ajustar modelo
        result = model.fit(y_region, params, x=x_region)

        # 4️⃣ Plot
        from rasp.plot_utils import config_figure

        ax = config_figure(fig_title=f"Deconvolution at {region[0]} – {region[1]}"+" cm$^{-1}$",
                            size=(2000, 2000))
        # TODO: probably a problem of underfittin => try to use a larger region to deconvolute 
        ax.plot(x_region, y_region, 
                color="#464646", alpha=.75, lw=1.3, ls='-', 
                label='Original', zorder=1)
        ax.plot(x_region, result.best_fit, 
                color="#E60032", lw=1.75, ls=':', 
                label='Total fit', zorder=2)
        
        comps = result.eval_components(x=x_region)
        for i in range(n_peaks):
            ax.plot(x_region, comps[f"p{i}_"],
                    alpha=.75, lw=1, ls=':', 
                    label=f'Peak {i+1}', zorder=2)
            ax.fill_between(x_region, 0, comps[f"p{i}_"], 
                            alpha=.15, 
                            zorder=1)

        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.set_xlim((min(x_region), max(x_region)))
        ax.set_ylim((min(y_region), max(1.25*y_region)))
        ax.legend()
        plt.tight_layout()

        # Salvar figura se ativado
        if save_figs:
            fig_name = f"{label_str}_{region[0]}-{region[1]}cm.png"
            fig_path = os.path.join(fig_folder, fig_name)
            plt.savefig(fig_path, dpi=300)
            plt.close()

        # 5️⃣ Resultados: extrair métricas
        rows = []
        for i in range(n_peaks):
            pref = f"p{i}_"
            amp = result.params[pref + 'amplitude'].value
            cen = result.params[pref + 'center'].value
            sig = result.params[pref + 'sigma'].value
            gam = result.params[pref + 'gamma'].value
            fwhm = 0.5346 * 2 * gam + np.sqrt(0.2166 * (2 * gam)**2 + (2.3548 * sig)**2)
            area = amp  # área aproximada no Voigt pode ser refinada, mas serve assim
            rows.append({
                "peak"      : i+1,
                "center"    : cen,
                "amplitude" : amp,
                "FWHM"      : fwhm,
                "area"      : area
            })

        df = pd.DataFrame(rows)
        df["region"]    = f"{region[0]}–{region[1]}"
        df["n_peaks"]   = n_peaks
        df["r_squared"] = result.rsquared if hasattr(result, "rsquared") else np.nan

        return df
    
    os.makedirs(fig_folder, exist_ok=True)
    all_results = []

    for spectrum, label in zip(spectra, labels):
        group, conc = label
        label_str = f"{group}_{conc}mM".replace(" ", "_")

        # Ajustar e plotar
        df = deconvolve_band(spectrum)

        # Adicionar info de amostra
        df["sample"] = label_str

        all_results.append(df)

    df_result = pd.concat(all_results, ignore_index=True)

    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_result.to_csv(csv_path, index=False)
        print(f"✅ Resultados salvos em: {csv_path}")

    return df_result
