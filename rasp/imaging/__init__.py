"""RASP Imaging sub‑package.

Handles 2‑D Raman maps: I/O, preprocessing, visualisation and
multivariate analysis (k‑means, PCA).
"""

from .io_map import load_file as load_map
from .batch  import batch_process

__all__ = ["load_map", "batch_process"]
