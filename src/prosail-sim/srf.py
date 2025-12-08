"""Sensor response function handling and spectral convolution."""

from typing import Dict
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import SENSOR_SRF_PATH


def load_srf(path=SENSOR_SRF_PATH) -> Dict[str, np.ndarray]:
    """Load SRF from Excel; assumes first column is wavelength and others are bands."""
    df = pd.read_excel(path)
    wl = df.iloc[:, 0].to_numpy()
    srf: Dict[str, np.ndarray] = {}
    for col in df.columns[1:]:
        srf[col] = np.vstack([wl, df[col].to_numpy()])
    return srf


def convolve_to_multispectral(
    refl_hs: np.ndarray,
    wl_hs: np.ndarray,
    srf: Dict[str, np.ndarray],
) -> np.ndarray:
    """Convolve hyperspectral reflectance to multispectral bands."""
    n_samples = refl_hs.shape[0]
    band_names = list(srf.keys())
    n_bands = len(band_names)
    refl_ms = np.zeros((n_samples, n_bands), dtype=float)

    for j, b in enumerate(band_names):
        wl_band, rsp = srf[b]
        rsp_interp = np.interp(wl_hs, wl_band, rsp)
        rsp_norm = rsp_interp / (rsp_interp.sum() + 1e-8)
        for i in range(n_samples):
            refl_ms[i, j] = simpson(refl_hs[i, :] * rsp_norm, wl_hs)
    return refl_ms
