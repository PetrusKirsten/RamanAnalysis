# rasp/imaging/batch.py  ← coloque no lugar do skeleton
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, datetime
from typing import Sequence, Tuple, List, Literal

import numpy as np
from tqdm import tqdm

import ramanspy as rp
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from io_map          import load_file
from preprocess_map  import preprocess_maps, get_sum
from viz_topo        import plot_topography
from viz_band        import extract_band, plot_band
from _config         import set_font, config_figure, normalize


# ───────────────────────────────────────── dataclass de parâmetros
BandTuple = Tuple[int, int, str]   # (center, width, label)

@dataclass
class BatchParams:
    input_folder   : str
    output_folder  : str
    region         : Tuple[int, int] = (280, 1800)
    win_len        : int = 15

    # ─ mapas a gerar ─────────────────
    do_topography  : bool = True
    do_bands       : bool = False
    do_multiband   : bool = False
    do_kmeans      : bool = False
    do_pca         : bool = False

    # opções band-map
    bands          : List[BandTuple] | None = None         # exigir se do_bands=True
    compensation   : Literal['raw', 'global', 'diff'] = 'diff'
    diff_sigma     : float = 0.9
    mask_percentile: float = 5.0

    # multiband RGB: passa índices (0,1,2) referindo-se a self.bands
    multiband_idx  : Tuple[int, int, int] = (0, 1, 2)

    # k-means / PCA
    n_clusters     : int = 4
    pca_components : int = 3


# ───────────────────────────────────────── helpers internos
def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=2, default=str)


def _make_out_dir(base: Path, sample_name: str) -> Path:
    d = base / sample_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_multiband_rgb(img: rp.SpectralImage,
                        bands: Sequence[BandTuple],
                        idx_rgb: Tuple[int, int, int],
                        out: Path):
    """Cria falso-RGB a partir de 3 bandas."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from .viz_band import extract_band

    centers = [bands[i][0] for i in idx_rgb]
    widths  = [bands[i][1] for i in idx_rgb]

    layers = [extract_band(img, c, w) for c, w in zip(centers, widths)]
    stacks = np.stack([normalize(L) for L in layers], axis=-1)

    ax = config_figure(f"RGB {centers}", (1400, 1200))
    ax.imshow(stacks)
    ax.axis('off')
    plt.savefig(out, dpi=300)
    plt.close()


def _plot_kmeans(img: rp.SpectralImage, n_clusters: int, out: Path):
    from matplotlib import cm
    spec_mat = img.spectral_data.reshape(-1, img.spectral_data.shape[-1])
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(spec_mat)
    labels = km.labels_.reshape(img.spectral_data.shape[:-1])
    cmap = cm.get_cmap('tab10', n_clusters)

    ax = config_figure(f'k-means (k={n_clusters})', (1400, 1200))
    ax.imshow(labels, cmap=cmap, interpolation='nearest')
    ax.axis('off')
    plt.savefig(out, dpi=300)
    plt.close()


def _plot_pca_maps(img: rp.SpectralImage, n_pc: int, out_dir: Path):
    spec_mat = img.spectral_data.reshape(-1, img.spectral_data.shape[-1])
    pca = PCA(n_components=n_pc).fit(spec_mat)
    scores = pca.transform(spec_mat).reshape(img.spectral_data.shape[:-1] + (n_pc,))

    for pc in range(n_pc):
        sc = scores[..., pc]
        ax = config_figure(f'PC{pc+1}', (1400, 1200))
        ax.imshow(normalize(sc), cmap='inferno')
        ax.axis('off')
        plt.savefig(out_dir / f'pca_PC{pc+1}.png', dpi=300)
        plt.close()


# ───────────────────────────────────────── função principal
def batch_process(params: BatchParams):
    """
    Percorre todos os .txt dentro de `input_folder`, gera mapas conforme flags,
    salva resultados organizados e loga parâmetros.

    Exemplo rápido:
    >>> from rasp.imaging.batch import BatchParams, batch_process
    >>> params = BatchParams(
    ...     input_folder="data/St CLs",
    ...     output_folder="figures/maps_St",
    ...     bands=[(862,20,"862"), (1080,10,"1080"), (1650,30,"1650")]
    ... )
    >>> batch_process(params)
    """
    inp  = Path(params.input_folder)
    outb = Path(params.output_folder)
    outb.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(f for f in inp.glob("*.txt") if f.is_file())
    if not txt_files:
        raise FileNotFoundError(f"No .txt maps found in {inp}")

    # 1) Discover and load all map files
    raw_maps = []
    map_files = [f for f in inp.glob("*.txt") if "Map" in f.name]

    # log.info(f"Found {len(map_files)} map files:"); time.sleep(.5)
    for f in map_files:
        print(f'\t\t→ {f.name}')
    # time.sleep(.5)
    for f in tqdm(map_files, desc="Loading maps", unit="file"):
        raw_maps.append(load_file(str(f)))

    # 2) Preprocess all maps at once
    maps_pp = []
    for m in tqdm(raw_maps, desc="Preprocessing maps", unit="maps"):
        maps_pp.append(preprocess_maps([m], region=params.region, win_len=params.win_len)[0])

    # |-|-| Log global
    run_ts  = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = outb / f'run_{run_ts}'
    run_dir.mkdir(exist_ok=True)

    _save_json({"params": asdict(params),
                "n_maps": len(txt_files),
                "timestamp": run_ts},
               run_dir / "run_log.json")

    # ─ 2. Loop por mapa
    for f, img in tqdm(list(zip(txt_files, maps_pp)), desc="Processing maps"):
        sample_name = f.stem.replace(' ', '_')
        sample_out  = run_dir / sample_name
        sample_out.mkdir(exist_ok=True)

        # 2.1 Topografia
        if params.do_topography:
            plot_topography(img, title=f, save=sample_out / "topography.png")

        # 2.2 Bandas
        if params.do_bands and params.bands:
            for center, width, label in params.bands:
                save_path = sample_out / f'band_{label}.png'
                plot_band(img,
                          center=center,
                          width=width,
                          title=f'{label} cm$^{-1}$',
                          compensation=params.compensation,
                          sigma=params.diff_sigma,
                          save=save_path)

        # 2.3 RGB multiband
        if params.do_multiband and params.bands and len(params.bands) >= 3:
            _plot_multiband_rgb(
                img,
                bands=params.bands,
                idx_rgb=params.multiband_idx,
                out=sample_out / "multiband_RGB.png"
            )

        # 2.4 k-means
        if params.do_kmeans:
            _plot_kmeans(img, params.n_clusters, sample_out / "kmeans.png")

        # 2.5 PCA
        if params.do_pca:
            _plot_pca_maps(img, params.pca_components, sample_out)

    print(f"✅ Batch finished. Results in {run_dir}")

if __name__ == "__main__":

    font_path = ("D:/Documents/GitHub/Raman-Analysis-Software/data/fonts/Helvetica-Light.ttf")   
    set_font(font_path)

    params = BatchParams(
        input_folder  ="./data/St CLs",
        output_folder ="./figures/maps_StkC",
        bands=[(862, 20, "862"), (1080, 10, "1080"), (1650, 30, "1650")],
        n_clusters=3)
    batch_process(params)
