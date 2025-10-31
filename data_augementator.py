# --------------------------------------------------------------------------------------    
# Clase para realizar el data augmentation del training set
# --------------------------------------------------------------------------------------    

import numpy as np
from dataloader.SpectrumObject import SpectrumObject

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
from scipy.signal import fftconvolve

# ----------------------------------------
# GETTERS Y SETTERS
# ----------------------------------------
def get_xy(spec) -> Tuple[np.ndarray, np.ndarray]:
    return spec.mz, spec.intensity

def set_xy(spec, x: np.ndarray, y: np.ndarray):
    spec.mz = x
    spec.intensity = y
    return spec

# ----------------------------------------
# OPERACIONES PARA REALIZAR EL AUMENTO 
# ----------------------------------------
def intensity_scale(y, scale_range=(0.85, 1.15), p=0.7, rng=None):
    if rng.random() > p: 
        return y
    
    s = rng.uniform(*scale_range)
    result = y * s
    return result

def mz_shift(x, y, ppm_range=(-10, 10), p=0.5, rng=None):
    if rng.random() > p: 
        return x, y
    
    ppm = rng.uniform(*ppm_range)
    frac = ppm * 1e-6
    x_shifted = x * (1 + frac)
    y_interp = np.interp(x, x_shifted, y, left=y[0], right=y[-1])

    return x, y_interp

def gaussian_noise(y, snr_db_range=(20, 40), p=0.7, rng=None):
    if rng.random() > p: 
        return y
    
    signal_power = float(np.mean(y**2)) + 1e-12
    snr_db = rng.uniform(*snr_db_range)
    noise_power = signal_power / (10**(snr_db/10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    result = y + noise

    return result

def baseline_poly(x, y, order=2, amp_frac_range=(-0.03, 0.03), p=0.4, rng=None):
    if rng.random() > p: 
        return y
    
    xc = (x - x.mean()) / (x.std() + 1e-12)
    coefs = rng.uniform(amp_frac_range[0], amp_frac_range[1], size=order+1)
    base = sum(coefs[i] * xc**i for i in range(order+1))
    result = y + base * (np.max(np.abs(y)) + 1e-12)

    return result

def peak_broadening(y, fwhm_pts_range=(1, 4), p=0.5, rng=None):
    if rng.random() > p: 
        return y
    
    fwhm = rng.uniform(*fwhm_pts_range)
    sigma = fwhm / (2*np.sqrt(2*np.log(2)) + 1e-12)
    radius = int(np.ceil(4*sigma))

    if radius < 1: 
        return y
    
    t = np.arange(-radius, radius+1)
    g = np.exp(-(t**2)/(2*sigma**2))
    g /= g.sum()

    return fftconvolve(y, g, mode="same")

def random_dropout(y, frac_range=(0.0, 0.01), p=0.25, rng=None):
    if rng.random() > p: 
        return y
    
    frac = rng.uniform(*frac_range)
    n = int(frac * y.size)

    if n <= 0: 
        return y
    
    idx = rng.choice(y.size, size=n, replace=False)
    y2 = y.copy()
    y2[idx] = 0.0

    return y2

def spikes(y, n_spikes_range=(0, 2), amp_frac_range=(0.1, 0.6), p=0.25, rng=None):
    if rng.random() > p: 
        return y
    
    y2 = y.copy()
    n_spikes = rng.integers(n_spikes_range[0], n_spikes_range[1]+1)
    amp = (np.max(np.abs(y)) + 1e-12) * rng.uniform(*amp_frac_range)

    for _ in range(n_spikes):
        i = rng.integers(0, y.size)
        y2[i] += rng.choice([-1, 1]) * amp

    return y2

# ----------------------------------------
# CLASE QUE REALIZA EL AUMENTO
# ----------------------------------------
class DataAugmentor:

    # Aumentador de los espectros de los datos iniciales

    def __init__(self, seed: Optional[int] = None, clip_nonneg: bool = True):
        self.rng = np.random.default_rng(seed)
        self.clip_nonneg = clip_nonneg

    def augment_individual(self, spec):
        # Obtenemos los datos utilizando el getter
        x, y = get_xy(spec)
        x_aug = x.copy()
        y_aug = y.astype(float).copy()

        # Realizamos las distintas operaciones creadas para el data augmenter
        y_aug = intensity_scale(y_aug, rng=self.rng)

        x_aug, y_aug = mz_shift(x_aug, y_aug, rng=self.rng)

        y_aug = baseline_poly(x_aug, y_aug, rng=self.rng)

        y_aug = peak_broadening(y_aug, rng=self.rng)

        y_aug = gaussian_noise(y_aug, rng=self.rng)

        y_aug = random_dropout(y_aug, rng=self.rng)

        y_aug = spikes(y_aug, rng=self.rng)

        if self.clip_nonneg:
            y_aug = np.maximum(y_aug, 0.0)

        # Los datos obtenidos se guardan utilizando el setter
        set_xy(spec, x_aug, y_aug)
        return spec

    def augment_variantes(self, spectra: Iterable, k_per_spectrum: int = 1) -> List:
        # Generamo un nº de variantes igual a k_per_spectrum por cada espectro de entrada y devolvemos al final una lista con todas las variantes
        out = []
        for spec in spectra:
            for _ in range(k_per_spectrum):
                x, y = get_xy(spec)
                spec_copy = type(spec)() if hasattr(type(spec), "__call__") else spec
                spec_copy = type(spec)(mz=x.copy(), intensity=y.copy())
                out.append(self.augment_individual(spec_copy))

        return out

    def augment_dataset(
        self,
        spectra: List,
        ids: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        k_per_spectrum: int = 1,
        id_suffix: str = "aug"
    ):
        # Aumenta la colección
        aug_specs = self.augment_variantes(spectra, k_per_spectrum=k_per_spectrum)

        aug_ids = None
        if ids is not None:
            aug_ids = []
            for id_ in ids:
                for i in range(k_per_spectrum):
                    aug_ids.append(f"{id_}{id_suffix}{i+1}")

        aug_labels = None
        if labels is not None:
            # Repite cada etiqueta k veces, en el mismo orden que 'spectra'
            aug_labels = np.repeat(labels, k_per_spectrum)

        return aug_specs, aug_ids, aug_labels
