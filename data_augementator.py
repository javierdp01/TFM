# --------------------------------------------------------------------------------------    
# Clase para realizar el data augmentation del training set
# --------------------------------------------------------------------------------------    

import numpy as np
from Código.dataloader.SpectrumObject import SpectrumObject

# ================== DATA AUGMENTATION PARA ESPECTROS 1D ==================
rng = np.random.default_rng(42)

AUGMENT_K = 2  # nº de réplicas aumentadas por espectro de entrenamiento (ajusta según tamaño/clase)
# Si quieres balancear clases, puedes variar K por clase (ver nota más abajo).

def _gaussian_kernel(size=7, sigma=1.2):
    x = np.arange(size) - (size-1)/2
    k = np.exp(-(x**2)/(2*sigma**2))
    return k / k.sum()

def aug_shift_ppm(s, ppm_std=15.0):
    """Desplaza el eje m/z con una N(0, ppm_std) y reinterpela a la malla original."""
    mz = s.mz
    I = s.intensity
    ppm = rng.normal(0.0, ppm_std)
    factor = 1.0 + ppm * 1e-6
    mz_shifted = mz * factor
    I_shifted = np.interp(mz, mz_shifted, I, left=0.0, right=0.0)
    return SpectrumObject(mz=mz.copy(), intensity=I_shifted)

def aug_intensity_scale_jitter(s, scale_range=(0.9, 1.1), jitter_std_frac=0.01):
    """Escala globalmente y añade jitter gaussiano relativo al máximo."""
    mz = s.mz
    I = s.intensity
    scale = rng.uniform(*scale_range)
    noise = rng.normal(0.0, jitter_std_frac * max(1e-12, I.max()), size=I.shape)
    I_new = np.clip(I * scale + noise, a_min=0.0, a_max=None)
    return SpectrumObject(mz=mz.copy(), intensity=I_new)

def aug_baseline_drift(s, amp_frac=0.02):
    """Añade una deriva suave de baseline (que luego tu SNIP debería corregir)."""
    mz = s.mz
    I = s.intensity
    L = len(I)
    t = np.linspace(0.0, 1.0, L)
    # combinación lineal + seno muy lento
    a = rng.uniform(-1.0, 1.0) * amp_frac * max(1e-12, I.max())
    b = rng.uniform(-1.0, 1.0) * amp_frac * max(1e-12, I.max())
    drift = a * (t - 0.5) + b * np.sin(2*np.pi * t * rng.uniform(0.5, 1.5))
    I_new = np.clip(I + drift, a_min=0.0, a_max=None)
    return SpectrumObject(mz=mz.copy(), intensity=I_new)

def aug_peak_broadening(s, ksize=5, sigma=1.0):
    """Ligero ensanchamiento de picos por convolución gaussiana."""
    from scipy.signal import fftconvolve
    mz = s.mz
    I = s.intensity
    k = _gaussian_kernel(size=ksize, sigma=sigma)
    I_new = fftconvolve(I, k, mode='same')
    return SpectrumObject(mz=mz.copy(), intensity=I_new)

def apply_augmentations(s):
    """
    Cadena de augmentaciones suaves y aleatorias.
    Orden: shift m/z -> (opcional) baseline drift -> scale+jitter -> broadening ligero
    """
    x = s
    # 70–80% de probabilidad de aplicar cada una, para variedad:
    if rng.uniform() < 0.9:
        x = aug_shift_ppm(x, ppm_std=15.0)
    if rng.uniform() < 0.5:
        x = aug_baseline_drift(x, amp_frac=0.02)
    if rng.uniform() < 0.9:
        x = aug_intensity_scale_jitter(x, scale_range=(0.9, 1.1), jitter_std_frac=0.01)
    if rng.uniform() < 0.6:
        x = aug_peak_broadening(x, ksize=5, sigma=1.0)
    return x
# ========================================================================
