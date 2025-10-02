from dataloader.SpectrumObject import SpectrumObject
import numpy as np
from typing import Optional, Callable


class DataAugmenter:
    """
    A class for augmenting SpectrumObject instances with various transformations
    to simulate real-world variability in spectral data.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initializes the DataAugmenter class.

        Parameters
        ----------
        random_state : int, optional
            Seed for reproducibility of random operations. Default is None.
        """
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y=None):
        """
        Fit method for compatibility with Scikit-learn pipelines. Does nothing.

        Parameters
        ----------
        X : list of SpectrumObject
            Input spectra.
        y : None
            Ignored.

        Returns
        -------
        self : DataAugmenter
            The fitted DataAugmenter instance.
        """
        return self

    def transform(self, X):
        """
        Transform method for compatibility with Scikit-learn pipelines. Applies augmentations.

        Parameters
        ----------
        X : list of SpectrumObject
            Input spectra.

        Returns
        -------
        list of SpectrumObject
            Augmented spectra.
        """
        return [self.augment(spectrum) for spectrum in X]

    def augment(self, spectrum: SpectrumObject) -> SpectrumObject:
        """
        Default augmentation pipeline. Applies intensity variability, shifting,
        and machine variability sequentially.

        Parameters
        ----------
        spectrum : SpectrumObject
            The SpectrumObject to be augmented.

        Returns
        -------
        SpectrumObject
            The augmented SpectrumObject.
        """
        spectrum = self.intensity_variability(spectrum, p1=0.5, v1=0.1)
        spectrum = self.shifting(spectrum, p0=0.5, p3=0.3, p6=0.15, p9=0.05)
        spectrum = self.machine_variability(spectrum, p1=0.15, v3=0.1)
        return spectrum

    def intensity_variability(self, spectrum: SpectrumObject, p1: float = 0.5, v1: float = 0.1) -> SpectrumObject:
        """
        Adds variability to the intensities of a SpectrumObject to simulate differences
        between mediums, weeks, or experimental conditions.

        Parameters
        ----------
        spectrum : SpectrumObject
            The SpectrumObject to augment.
        p1 : float, default=0.5
            The fraction of peaks to perturb (0 < p1 <= 1).
        v1 : float, default=0.1
            The variance of the Gaussian noise added to the peaks.

        Returns
        -------
        SpectrumObject
            The augmented SpectrumObject with perturbed intensities.
        """
        augmented_intensities = np.copy(spectrum.intensity)
        augmented_mz = np.copy(spectrum.mz)
        if not (0 < p1 <= 1):
            raise ValueError("p1 must be a fraction between 0 and 1.")
        if v1 <= 0:
            raise ValueError("v1 must be a positive variance.")

        non_zero_indices = np.flatnonzero(spectrum.intensity)
        if len(non_zero_indices) == 0:
            raise ValueError("Spectrum has no non-zero intensities to augment.")

        n_samples = int(p1 * len(non_zero_indices))
        selected_indices = self.rng.choice(non_zero_indices, n_samples, replace=False)
        noise = self.rng.normal(0, np.sqrt(v1), n_samples)

        augmented_intensities[selected_indices] += noise
        augmented_intensities = np.clip(augmented_intensities, 0, None)

        # Recreate the SpectrumObject with the augmented intensities
        spectrum.intensity = augmented_intensities
        spectrum.mz = augmented_mz
        return spectrum

    def proportional_shifting(self, spectrum: SpectrumObject, bin_size: int = 3, p0: float = 0.2) -> SpectrumObject:
        """
        Shifts a proportion of peaks in the spectrum, where higher m/z values are shifted more than lower m/z values.

        Parameters
        ----------
        spectrum : SpectrumObject
            The spectrum to augment.
        bin_size : int, default=3
            The bin size for determining the shift amounts.
        p0 : float, default=0.2
            The proportion of peaks to shift.

        Returns
        -------
        SpectrumObject
            The augmented spectrum.
        """
        # Copy the spectrum to avoid modifying the original
        mz = np.copy(spectrum.mz)
        intensities = np.copy(spectrum.intensity)

        # Validate inputs
        if not (0 < p0 <= 1):
            raise ValueError("p0 must be a fraction between 0 and 1. Current value: {}".format(p0))
        if len(mz) == 0:
            raise ValueError("Spectrum is empty.")

        # Select a proportion of peaks to shift
        n_peaks = int(p0 * len(mz))
        selected_indices = np.random.choice(len(mz), size=n_peaks, replace=False)

        # Compute shift amounts proportional to the position in the spectrum
        max_shift_bins = np.floor(9 / bin_size)  # Maximum possible shift (9 Da)
        relative_positions = np.linspace(0, 1, len(mz))  # Relative position from 0 to 1
        shift_amounts = relative_positions[selected_indices] * max_shift_bins

        # Randomly decide shift direction (+1 for right, -1 for left)
        shift_directions = np.random.choice([-1, 1], size=n_peaks)
        applied_shifts = shift_directions * shift_amounts

        # Apply shifts to the m/z values
        shifted_mz = np.copy(mz)
        shifted_mz[selected_indices] += applied_shifts

        # Interpolate intensities to match the new m/z values
        augmented_intensities = np.interp(mz, shifted_mz, intensities, left=0, right=0)

        # Update the augmented spectrum
        spectrum.mz = shifted_mz
        spectrum.intensity = augmented_intensities

        return spectrum

                

    def machine_variability(self, spectrum: SpectrumObject, p1: float = 0.15, v3: float = 0.1) -> SpectrumObject:
        """
        Simulates machine variability by adding Gaussian noise to zero-intensity peaks.

        Parameters
        ----------
        spectrum : SpectrumObject
            The SpectrumObject to augment.
        p1 : float, default=0.15
            The fraction of zero-intensity locations to perturb.
        v3 : float, default=0.1
            The variance of the Gaussian noise.

        Returns
        -------
        SpectrumObject
            The augmented SpectrumObject.
        """
        if not (0 < p1 <= 1):
            raise ValueError("p1 must be a fraction between 0 and 1.")
        if v3 <= 0:
            raise ValueError("v3 must be a positive variance.")

        zero_indices = np.flatnonzero(spectrum.intensity == 0)
        if len(zero_indices) == 0:
            raise ValueError("Spectrum has no zero-intensity locations to augment.")

        n_peaks = int(p1 * len(zero_indices))
        selected_indices = self.rng.choice(zero_indices, n_peaks, replace=False)
        noise = self.rng.normal(0, np.sqrt(v3), n_peaks)

        augmented_intensities = spectrum.intensity.copy()
        augmented_intensities[selected_indices] += noise
        augmented_intensities = np.clip(augmented_intensities, 0, None)

        spectrum.intensity = augmented_intensities
        spectrum.mz = spectrum.mz
        return spectrum
