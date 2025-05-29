from __future__ import annotations
# rasp/imaging/batch.py  ← coloque no lugar do skeleton
import json, datetime
from pathlib import Path
from typing import Sequence, Tuple, List, Literal
from dataclasses import dataclass, asdict

import numpy as np
from tqdm import tqdm
import ramanspy as rp

from matplotlib     import pyplot as plt

from scipy.ndimage  import gaussian_filter

from sklearn.decomposition  import PCA
from sklearn.cluster        import KMeans

from io_map          import load_file
from preprocess_map  import preprocess_maps
from _config         import set_font, config_figure, normalize
from viz_topo        import plot_topography
from viz_band        import plot_band
from viz_spectrum    import plot_mean_spectrum
from viz_multiband   import plot_multiband
from viz_kmeans      import plot_kmeans


# ───────────────────────────────────────── dataclass de parâmetros
BandTuple = Tuple[int, int, str]   # (center, width, label)

@dataclass
class BatchParams:
    input_folder   : str
    output_folder  : str
    region         : Tuple[int, int] = (315, 1780)

    # ─ mapas a gerar ─────────────────
    do_spectra     : bool = True
    do_topography  : bool = True
    do_bands       : bool = True
    do_multiband   : bool = True
    do_kmeans      : bool = False
    do_pca         : bool = False

    # opções band-map
    bands          : List[BandTuple] | None = None

    # multiband RGB: passa índices (0, 1, 2) referindo-se a self.bands
    mb_idx        : Tuple[int, int, int]       = (0, 1, 2)
    mb_thresholds : Tuple[float, float, float] = (0.75, 1.1, 0.45)

    # k-means / PCA
    n_clusters     : int = 2
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
    inp, outb  = Path(params.input_folder), Path(params.output_folder)
    outb.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(f for f in inp.glob("*.txt") if f.is_file())
    if not txt_files:
        raise FileNotFoundError(f"No .txt maps found in {inp}")

    # 1) Discover and load all map files
    map_files = [f for f in inp.glob("*.txt") if "Map" in f.name]
    for f in map_files:
        print(f'\t\t→ {f.name}')

    raw_maps = []
    for f in tqdm(map_files, desc="Loading maps", unit="file"):
        raw_maps.append(load_file(str(f)))

    # 2) Preprocess all maps at once
    maps_pp = []
    for m in tqdm(raw_maps, desc="Preprocessing maps", unit="maps"):
        maps_pp.append(preprocess_maps([m], region=params.region)[0])

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
        
        # 2.0 Spectra
        if params.do_spectra:
            plot_mean_spectrum(img, save=sample_out / "spectra.png")
        
        # 2.1 Topografia
        if params.do_topography:
            plot_topography(img, title=f, save=sample_out / "topography-raw.png")
            plot_topography(img, title=f, save=sample_out / "topography-outliers_correction.png",
                            correct_outliers_on=True, correct_shading_on=False)
            plot_topography(img, title=f, save=sample_out / "topography-outliersAndShading_correction.png",
                            correct_outliers_on=True, correct_shading_on=True)

        # 2.2 Bandas
        if params.do_bands and params.bands:
            for center, width, label in params.bands:
                plot_band(img,
                          center = center,
                          width  = width,
                          save   = sample_out / f'band_{label}-raw.png')
                
                plot_band(img,
                          center = center,
                          width  = width,
                          save   = sample_out / f'band_{label}-outliersAndShading_correction.png',
                          correct_outliers_on=True, correct_shading_on=True)

        # 2.3 RGB multiband
        if params.do_multiband and params.bands and len(params.bands) >= 3:
            mode = 'continuous'
            plot_multiband(
                img,
                bands       = params.bands,
                save        = sample_out / f'multibands_global_{mode}.png',
                thresholds  = params.mb_thresholds,
                normalize   = 'global',
                mode=mode
            )

            mode = 'binary'
            plot_multiband(
                img,
                bands       = params.bands,
                save        = sample_out / f'multibands_global_{mode}.png',
                thresholds  = params.mb_thresholds,
                normalize   = 'global',
                mode=mode
            )

        # 2.4 k-means
        if params.do_kmeans:
            plot_kmeans(img, n_clusters=params.n_clusters, save=sample_out / "kmeans.png")

        # 2.5 PCA
        if params.do_pca:
            _plot_pca_maps(img, params.pca_components, sample_out)

    print(f"✅ Batch finished. Results in {run_dir}")

if __name__ == "__main__":

    set_font("D:/Documents/GitHub/Raman-Analysis-Software/data/fonts/Helvetica-Light.ttf")

    # for sample in [' ',]:
    for sample in [' ', ' kC ', ' iC ']:
        params = BatchParams(
            input_folder  = f"./data/St{sample}CLs",
            output_folder = f"./figures/maps-St {sample}CLs",
            bands=[
                (851, 5, "851"),   # Red
                (939, 10, "939"),  # Green
                (478, 20, "478")   # Blue
            ],
            n_clusters=3)
    
        batch_process(params)
