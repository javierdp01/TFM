import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import os
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


class SpectrumObject:
    """Base Spectrum Object class

    Can be instantiated directly with 1-D np.arrays for mz and intensity.
    Alternatively, can be read from csv files or from bruker output data.
    Reading from Bruker data is based on the code in https://github.com/sgibb/readBrukerFlexData

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    """

    def __init__(self, mz=None, intensity=None):
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points in the spectrum as individualpeaks, instead of connecting the points in the spectrum, by default False
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)

    def __repr__(self):
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF

        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Read a spectrum from Bruker's format

        Parameters
        ----------
        acqu_file : str
            "acqu" file bruker folder
        fid_file : str
            "fid" file in bruker folder

        Returns
        -------
        SpectrumObject
        """
        with open(acqu_file, "rb") as f:
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            TD = len(intensity)
        TOF = DELAY + np.arange(TD) * DW

        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        intensity[intensity < 0] = 0

        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=" "):
        """Read a spectrum from txt

        Parameters
        ----------
        file : str
            path to csv file
        sep : str, optional
            separator in the file, by default " "

        Returns
        -------
        SpectrumObject
        """
        s = pd.read_table(
            file, sep=sep, index_col=None, comment="#", header=None
        ).values
        mz = s[:, 0]
        intensity = s[:, 1]
        return cls(mz=mz, intensity=intensity)

class Binner:

    """Pre-processing function for binning spectra in equal-width bins.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    step : int, optional
        width of every bin, by default 3
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, step=3, aggregation="sum"):
        self.bins = np.arange(start, stop + 1e-8, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation

    def __call__(self, SpectrumObj):
        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s


class Normalizer:
    """Pre-processing function for normalizing the intensity of a spectrum.
    Commonly referred to as total ion current (TIC) calibration.

    Parameters
    ----------
    sum : int, optional
        Make the total intensity of the spectrum equal to this amount, by default 1
    """

    def __init__(self, type="only-tic", sum=1):
        # type can be "only-tic" or "tic-and-"
        self.sum = sum

    def __call__(self, SpectrumObj):
        s = SpectrumObject()
        # Check if the sum is nonzero
        if SpectrumObj.intensity.sum() == 0:
            return SpectrumObj
        s = SpectrumObject(
            intensity=SpectrumObj.intensity / SpectrumObj.intensity.sum() * self.sum,
            mz=SpectrumObj.mz,
        )
        return s

from scipy.signal import find_peaks

def detect_peaks(SpectrumObj, halfWindowSize=20, noiseMethod="MAD", SNR=2):
    """
    Detect peaks in a MALDI spectrum using a noise estimation method and find_peaks.
    
    Parameters
    ----------
    SpectrumObj : SpectrumObject
        An object with attributes 'mz' (mass/charge values) and 'intensity' (intensity values).
    halfWindowSize : int, optional
        Half window size for local peak consideration (not used directly in this implementation, 
        but can be used for further refinement), by default 20.
    noiseMethod : str, optional
        Method to estimate noise. Currently supports "MAD" (median absolute deviation), by default "MAD".
    SNR : float, optional
        Signal-to-noise ratio multiplier. Peaks must exceed (noise level * SNR) to be considered.
    
    Returns
    -------
    peaks_mz : array
        An array of m/z values where peaks were detected.
    peak_indices : array
        Indices in SpectrumObj.mz corresponding to the detected peaks.
    """
    intensities = SpectrumObj.intensity
    
    # Estimate noise using the Median Absolute Deviation (MAD).
    if noiseMethod.upper() == "MAD":
        # Calculate the median and MAD.
        median_intensity = np.median(intensities)
        mad = np.median(np.abs(intensities - median_intensity))
        # Estimate noise level: scale MAD to approximate standard deviation.
        noise_level = 1.4826 * mad
    else:
        # Fallback: use a fixed noise level or raise an error.
        raise ValueError("Unsupported noiseMethod: use 'MAD'.")
    
    # Define a threshold based on the noise level and desired SNR.
    threshold = median_intensity + SNR * noise_level
    
    # Use SciPy's find_peaks to detect peaks above the threshold.
    peak_indices, _ = find_peaks(intensities, height=threshold)
    
    # Extract m/z values corresponding to the detected peak indices.
    peaks_mz = SpectrumObj.mz[peak_indices]
    
    return peaks_mz, peak_indices

class Aligner:
    """
    Pre-processing function for aligning a MALDI spectrum to a reference peak list.
    
    Parameters
    ----------
    reference_peaks : array-like
        The reference peak m/z values to which the spectra will be aligned.
    halfWindowSize : int, optional
        Half window size used in peak detection, by default 20.
    noiseMethod : str, optional
        Noise estimation method (see detect_peaks), by default "MAD".
    SNR : float, optional
        Signal-to-noise ratio threshold, by default 2.
    tolerance : float, optional
        Maximal relative deviation of a peak position to be considered as matching (e.g. 5e-6 for 5 ppm), by default 5e-6.
    warpingMethod : str, optional
        The warping method to use ("lowess" or "dtw"), by default "lowess".
    allowNoMatches : bool, optional
        If True, don't throw an error if no warping matches are found, by default False.
    emptyNoMatches : bool, optional
        If True, set intensity values to zero if no matches are found, by default False.
    """
    def __init__(self, reference_peaks, halfWindowSize=20, noiseMethod="MAD", SNR=2,
                 tolerance=5e-6, warpingMethod="lowess", allowNoMatches=False, emptyNoMatches=False):
        self.reference_peaks = np.array(reference_peaks)
        self.halfWindowSize = halfWindowSize
        self.noiseMethod = noiseMethod
        self.SNR = SNR
        self.tolerance = tolerance
        self.warpingMethod = warpingMethod.lower()
        self.allowNoMatches = allowNoMatches
        self.emptyNoMatches = emptyNoMatches

    def __call__(self, SpectrumObj):
        # 1. Detect peaks in the input spectrum.
        peaks = detect_peaks(SpectrumObj, halfWindowSize=self.halfWindowSize,
                             noiseMethod=self.noiseMethod, SNR=self.SNR)
        
        # 2. Compute the warping function using the chosen method.
        if self.warpingMethod == "lowess":
            warp_func = self._compute_lowess_warping(peaks, self.reference_peaks, self.tolerance)
        elif self.warpingMethod == "dtw":
            warp_func = self._compute_dtw_warping(peaks, self.reference_peaks, self.tolerance)
        else:
            raise ValueError("Unsupported warping method: choose 'lowess' or 'dtw'.")
        
        # 3. If no warping function is found, handle according to settings.
        if warp_func is None:
            if self.emptyNoMatches:
                new_intensity = np.zeros_like(SpectrumObj.intensity)
                return SpectrumObject(intensity=new_intensity, mz=SpectrumObj.mz)
            elif not self.allowNoMatches:
                raise ValueError("No warping function could be computed for spectrum with id: {}".format(getattr(SpectrumObj, 'id', 'unknown')))
        
        # 4. Apply the warping function to the m/z axis.
        new_mz = warp_func(SpectrumObj.mz)
        # 5. Interpolate the intensities onto the new m/z grid.
        new_intensity = np.interp(new_mz, SpectrumObj.mz, SpectrumObj.intensity)
        
        # 6. Return the warped SpectrumObject.
        return SpectrumObject(intensity=new_intensity, mz=new_mz)

    def _compute_lowess_warping(self, detected_peaks, ref_peaks, tolerance=9):
        """
        Compute a warping function using a lowess-based approach.
        For demonstration, this dummy implementation matches detected peaks to reference peaks
        if their relative difference is within the tolerance, then returns a simple linear interpolation.
        """
        detected_peaks = np.array(detected_peaks)
        ref_peaks = np.array(ref_peaks)
        
        # Find matching peaks.
        matches_detected = []
        matches_ref = []
        
        for dp in detected_peaks:
            if len(dp) > len(ref_peaks):
                dp = dp[:len(ref_peaks)]
            elif len(ref_peaks) > len(dp):
                ref_peaks = ref_peaks[:len(dp)]
                
            # Compute relative differences.
            diffs = np.abs(dp - ref_peaks) / ref_peaks
            if np.any(diffs < tolerance):
                idx = np.argmin(diffs)
                matches_detected.append(dp)
                matches_ref.append(ref_peaks[idx])
        
        # Require at least a few matches to build a warping function.
        if len(matches_detected) < 3:
            return None
        
        # Sort matches.
        matches_detected = np.array(matches_detected)
        matches_ref = np.array(matches_ref)
        sort_idx = np.argsort(matches_detected)
        matches_detected = matches_detected[sort_idx]
        matches_ref = matches_ref[sort_idx]
        
        # Define a warping function using linear interpolation between matched peaks.
        def warp_func(mz_vals):
            return np.interp(mz_vals, matches_detected, matches_ref)
        
        return warp_func

    def _compute_dtw_warping(self, detected_peaks, ref_peaks, tolerance):
        """
        Compute a warping function using a DTW-based approach.
        This is a placeholder. In a real implementation, you would use a DTW algorithm to align detected peaks to ref_peaks.
        Here we simply return an identity mapping as a dummy.
        """
        # For a real DTW, you might use a package like fastdtw or dtw-python.
        def warp_func(mz_vals):
            return mz_vals  # Dummy: no warping applied.
        return warp_func


class Trimmer:
    """Pre-processing function for trimming ends of a spectrum.
    This can be used to remove inaccurate measurements.

    Parameters
    ----------
    min : int, optional
        remove all measurements with mz's lower than this value, by default 2000
    max : int, optional
        remove all measurements with mz's higher than this value, by default 20000
    """

    def __init__(self, min=2000, max=20000):
        self.range = [min, max]

    def __call__(self, SpectrumObj):
        indices = (self.range[0] < SpectrumObj.mz) & (SpectrumObj.mz < self.range[1])

        s = SpectrumObject(
            intensity=SpectrumObj.intensity[indices], mz=SpectrumObj.mz[indices]
        )
        return s


import numpy as np

class IntensityThresholding:
    """Pre-processing function for thresholding low intensities in a spectrum.
    
    This function sets to 0 all intensity values that are lower than a computed threshold.
    The threshold can be determined by different methods:
    
      - 'zscore': Uses the mean and standard deviation of the intensities.
                The threshold is computed as: mean + (z_multiplier * std).
      - 'percentile': Uses a specified percentile of the intensity distribution.
      - 'fixed': A fixed absolute intensity value.
      
    Parameters
    ----------
    method : str, optional
        The method to determine the threshold. Options are 'zscore', 'percentile', or 'fixed'.
        Default is 'zscore'.
    threshold : float, optional
        The threshold value or multiplier:
          - If method is 'zscore', this is the z-score multiplier (e.g. 1.0).
          - If method is 'percentile', this is the percentile (0 to 100) below which intensities are zeroed, e.g. 10.
          - If method is 'fixed', this is the absolute intensity threshold, e.g. 0.1.
        Defaults: 1.0 for 'zscore', 10 for 'percentile', and 0.1 for 'fixed'.
    """
    
    def __init__(self, method='percentile', threshold=None):
        self.method = method.lower()
        if self.method == 'zscore':
            self.threshold = threshold if threshold is not None else 1.0  # multiplier for standard deviation
        elif self.method == 'percentile':
            self.threshold = threshold if threshold is not None else 10  # percentile (e.g., 10th percentile)
        elif self.method == 'fixed':
            self.threshold = threshold if threshold is not None else 0.1  # fixed intensity value
        else:
            raise ValueError("Method must be one of 'zscore', 'percentile', or 'fixed'")
    
    def __call__(self, SpectrumObj):
        intensities = SpectrumObj.intensity
        if self.method == 'zscore':
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)
            # Compute threshold as mean plus z-score multiplier times standard deviation.
            computed_threshold = mean_val + self.threshold * std_val
        elif self.method == 'percentile':
            computed_threshold = np.percentile(intensities, self.threshold)
        elif self.method == 'fixed':
            computed_threshold = self.threshold
        
        # Set intensities below the computed threshold to 0.
        new_intensity = np.where(intensities < computed_threshold, 0, intensities)
        
        # Return a new SpectrumObject with the thresholded intensities.
        new_spectrum = SpectrumObj.__class__(intensity=new_intensity, mz=SpectrumObj.mz)
        return new_spectrum


class VarStabilizer:
    """Pre-processing function for manipulating intensities.
    Commonly performed to stabilize their variance.

    Parameters
    ----------
    method : str, optional
        function to apply to intensities.
        can be either "sqrt", "log", "log2" or "log10", by default "sqrt"
    """

    def __init__(self, method="sqrt"):
        methods = {"sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10}
        self.fun = methods[method]

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=self.fun(SpectrumObj.intensity), mz=SpectrumObj.mz)
        return s


class BaselineCorrecter:
    """Pre-processing function for baseline correction (also referred to as background removal).

    Support SNIP, ALS and ArPLS.
    Some of the code is based on https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    Parameters
    ----------
    method : str, optional
        Which method to use
        either "SNIP", "ArPLS" or "ALS", by default None
    als_lam : float, optional
        lambda value for ALS and ArPLS, by default 1e8
    als_p : float, optional
        p value for ALS and ArPLS, by default 0.01
    als_max_iter : int, optional
        max iterations for ALS and ArPLS, by default 10
    als_tol : float, optional
        stopping tolerance for ALS and ArPLS, by default 1e-6
    snip_n_iter : int, optional
        iterations of SNIP, by default 10
    """

    def __init__(
        self,
        method=None,
        als_lam=1e8,
        als_p=0.01,
        als_max_iter=10,
        als_tol=1e-6,
        snip_n_iter=10,
    ):
        self.method = method
        self.lam = als_lam
        self.p = als_p
        self.max_iter = als_max_iter
        self.tol = als_tol
        self.n_iter = snip_n_iter

    def __call__(self, SpectrumObj):
        if "LS" in self.method:
            baseline = self.als(
                SpectrumObj.intensity,
                method=self.method,
                lam=self.lam,
                p=self.p,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        elif self.method == "SNIP":
            baseline = self.snip(SpectrumObj.intensity, self.n_iter)

        s = SpectrumObject(
            intensity=SpectrumObj.intensity - baseline, mz=SpectrumObj.mz
        )
        return s

    def als(self, y, method="ArPLS", lam=1e8, p=0.01, max_iter=10, tol=1e-6):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(
            D.transpose()
        )  # Precompute this term since it does not depend on `w`

        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        crit = 1
        count = 0
        while crit > tol:
            z = sparse.linalg.spsolve(W + D, w * y)

            if method == "AsLS":
                w_new = p * (y > z) + (1 - p) * (y < z)
            elif method == "ArPLS":
                d = y - z
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                w_new = 1 / (1 + np.exp(np.minimum(2 * (d - (2 * s - m)) / s, 70)))

            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)
            count += 1
            if count > max_iter:
                break
        return z

    def snip(self, y, n_iter):
        y_prepr = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        for i in range(1, n_iter + 1):
            rolled = np.pad(y_prepr, (i, i), mode="edge")
            new = np.minimum(
                y_prepr, (np.roll(rolled, i) + np.roll(rolled, -i))[i:-i] / 2
            )
            y_prepr = new
        return (np.exp(np.exp(y_prepr) - 1) - 1) ** 2 - 1


class Smoother:
    """Pre-processing function for smoothing. Uses Savitzky-Golay filter.

    Parameters
    ----------
    halfwindow : int, optional
        halfwindow of savgol_filter, by default 10
    polyorder : int, optional
        polyorder of savgol_filter, by default 3
    """

    def __init__(self, halfwindow=10, polyorder=3):
        self.window = halfwindow * 2 + 1
        self.poly = polyorder

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=np.maximum(
                savgol_filter(SpectrumObj.intensity, self.window, self.poly), 0
            ),
            mz=SpectrumObj.mz,
        )
        return s



class LocalMaximaPeakDetector:
    """
    Detects peaks a la MaldiQuant

    Parameters
    ----------
    SNR : int, optional
        Signal to noise radio. This function computes a SNR value as the median absolute deviation from the median intensity (MAD).
        Only peaks with intensities a multiple of this SNR are considered. By default 2.
    halfwindowsize: int, optional
        half window size, an intensity can only be a peak if it is the highest value in a window. By default 20, for a total window size of 41.
    """

    def __init__(
        self,
        SNR=2,
        halfwindowsize=20,
    ):
        self.hw = halfwindowsize
        self.SNR = SNR

    def __call__(self, SpectrumObj):
        SNR = (
            np.median(np.abs(SpectrumObj.intensity - np.median(SpectrumObj.intensity)))
            * self.SNR
        )

        local_maxima = np.argmax(
            np.lib.stride_tricks.sliding_window_view(
                SpectrumObj.intensity, (int(self.hw * 2 + 1),)
            ),
            -1,
        ) == int(self.hw)
        s_int_local = SpectrumObj.intensity[self.hw : -self.hw][local_maxima]
        s_mz_local = SpectrumObj.mz[self.hw : -self.hw][local_maxima]
        return SpectrumObject(
            intensity=s_int_local[s_int_local > SNR], mz=s_mz_local[s_int_local > SNR]
        )

class PeakFilter:
    """Pre-processing function for filtering peaks.

    Filters in two ways: absolute number of peaks and height.

    Parameters
    ----------
    max_number : int, optional
        Maximum number of peaks to keep. Prioritizes peaks to keep by height.
        by default None, for no filtering
    min_intensity : float, optional
        Min intensity of peaks to keep, by default None, for no filtering
    """

    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=SpectrumObj.intensity, mz=SpectrumObj.mz)

        if self.max_number is not None:
            indices = np.argsort(-s.intensity, kind="stable")
            take = np.sort(indices[: self.max_number])

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        if self.min_intensity is not None:
            take = s.intensity >= self.min_intensity

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        return s


class RandomPeakShifter:
    """Pre-processing function for adding random (gaussian) noise to the mz values of peaks.

    Parameters
    ----------
    std : float, optional
        stdev of the random noise to add, by default 1
    """

    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.normal(scale=self.std, size=SpectrumObj.mz.shape),
        )
        return s
    
class NoiseRemoval:
    """Pre-processing function for removing all peaks with intensities below a certain threshold. All peaks with intensities below the mean minus the standard deviation are removed.
    
    Implemented by Rafael Rodríguez Palomo.
    ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡ DEPRECATED: Use IntensityThresholding instead !!!!!!!!!!!!!
    
        Parameters
        ----------
        SpectrumObj : SpectrumObject with the MALDI data.
            The MALDI data from which noise is to be removed.

        Returns
        -------
        denoised_data : DataFrame
            The denoised MALDI data.
      """
    def __init__(self):
        pass
    
    def __call__(self, SpectrumObj):
        data = SpectrumObj.intensity
        mean_data = np.mean(data)
        std_data = np.std(data)
        # Remove all peaks with intensities below the mean minus the standard deviation
        denoised_data = data[data > mean_data - std_data]
        SpectrumObj.intensity = denoised_data
        
        return SpectrumObj

class PIKEScaleNormalizer():
    """
    Normalizes each spectrum between [1, 2]. All peaks are scaled to be greater or equal to 1.
    """

    def __init__(self):
        pass
    
    def __call__(self, SpectrumObj):
        data = SpectrumObj.intensity
        # Apply x= (b-a)(x-min(x))/(max(x)-min(x)) + a where a=1 and b=2
        data = (2-1)*(data-np.min(data))/(np.max(data)-np.min(data)) + 1
        ## Scale by the minimum value non zero
        # data = data / np.min(data[np.nonzero(data)])
        SpectrumObj.intensity = data
        return SpectrumObj



class UniformPeakShifter:
    """Pre-processing function for adding uniform noise to the mz values of peaks.

    Parameters
    ----------
    range : float, optional
        let each peak shift by maximum this value, by default 1.5
    """

    def __init__(self, range=1.5):
        self.range = range

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.uniform(
                low=-self.range, high=self.range, size=SpectrumObj.mz.shape
            ),
        )
        return s


class Binarizer:
    """Pre-processing function for binarizing intensity values of peaks.

    Parameters
    ----------
    threshold : float
        Threshold for the intensities to become 1 or 0.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=(SpectrumObj.intensity > self.threshold).astype(
                SpectrumObj.intensity.dtype
            ),
            mz=SpectrumObj.mz,
        )
        return s


class SequentialPreprocessor:
    """Chain multiple preprocessors so that a pre-processing pipeline can be called with one line.

    Example:
    ```python
    preprocessor = SequentialPreprocessor(
        VarStabilizer(),
        Smoother(),
        BaselineCorrecter(method="SNIP"),
        Normalizer(),
        Binner()
    )
    preprocessed_spectrum = preprocessor(spectrum)
    ```
    """

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, SpectrumObj):
        for step in self.preprocessors:
            SpectrumObj = step(SpectrumObj)
        return SpectrumObj