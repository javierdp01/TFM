import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch

class EfficientPIKE():
    def __init__(self, t):
        self.t = t

    def __call__(self, X_mz, X_i, Y_mz=None, Y_i=None, th=1e-6):
        '''
        Returns the PIKE kernel value k(X, Y) computed using GPU acceleration.

        Parameters
        ----------
        X_mz:   array of spectra positions (mz) with shape (n_samples_X, spectrum_length_X).
        X_i:    array of spectra intensities with shape (n_samples_X, spectrum_length_X).
        Y_mz:   array of spectra positions (mz) with shape (n_samples_Y, spectrum_length_X).
        Y_i:    array of spectra intensities with shape (n_samples_Y, spectrum_length_X).
        th:     threshold for significant distances.

        Returns
        -------
        K:      array, shape (n_samples_X, n_samples_Y) containing the kernel values.
        '''
        if Y_mz is None and Y_i is None:
            Y_mz = X_mz
            Y_i = X_i

        # Move data to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_mz = torch.tensor(X_mz, dtype=torch.float32, device=device)
        X_i = torch.tensor(X_i, dtype=torch.float32, device=device)
        Y_mz = torch.tensor(Y_mz, dtype=torch.float32, device=device)
        Y_i = torch.tensor(Y_i, dtype=torch.float32, device=device)

        # Precompute distances on GPU
        print("Calculating distances")
        positions_x = X_mz[0, :].view(-1, 1)
        positions_y = Y_mz[0, :].view(-1, 1)
        distances = torch.cdist(positions_x, positions_y)
        distances = torch.exp(-distances / (4 * self.t))
        d = int(torch.where(distances[0] < th)[0][0])

        # Kernel matrix
        K = torch.zeros((X_mz.shape[0], Y_mz.shape[0]), device=device)
        P = torch.zeros_like(K, device=K.device)

        # Compute the kernel row-by-row with progress tracking
        print("Calculating kernel")
        for i in tqdm(range(X_i.shape[1]), desc="Processing Kernel Rows"):
            x = X_i[:, i]  # Shape: (n_samples_X,)
            intensities_y = Y_i[:, :i + d]  # Shape: (n_samples_Y, i + d)
            di = distances[i, :i + d].view(1, -1)  # Shape: (1, i + d)

            # Compute product intensities_y * di
            prod = intensities_y * di  # Broadcasting to match shapes

            # Broadcast x to match the product shape
            x_broadcast = x.view(-1, 1).expand(-1, prod.shape[1])  # Shape: (n_samples_X, i + d)

            # Compute the partial kernel matrix
            P[:] = torch.matmul(x_broadcast, prod.T)  # Shape: (n_samples_X, n_samples_Y)

            # Accumulate into the kernel matrix
            K += P
            
            # del intermidiate variables
            del x, intensities_y, di, prod, x_broadcast

        # Normalize still in cuda
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        K = K / (4 * self.t * torch.pi)
        # K = K.cpu().numpy() / (4 * self.t * np.pi)
        return K


class KernelPipeline():
    '''
    A pipeline for processing MALDI data with different peak removal strategies and computing the E-PIKE.
    
    Attributes
    ----------
    peak_removal : str or None
        The strategy for peak removal. Options are None, 'masked', or 'spr'.
    common_peaks : array-like or None
        The common peak indicies in Da used when `peak_removal` is not None.
    '''
    def __init__(self, peak_removal=None):
        self.peak_removal = peak_removal
        # Ribosomal peaks defined by the microbiologists
        self.common_peaks = [2178,2545,2687,2832,2873,3125,3155,3252,3323,3577,3635,3933,4010,4183,4363,4436,4495,4523,4775,5094,5379,5750,5924,6253,6313,6409,6506,6806,7155,7271,7646,7868,8346,8366,8872,8990,9059,9221,9532,9550,9708,10134,10295,10512]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Strategies
        # self.processing_mode = {
        #     None: self._no_peak_removal,
        #     'masked': self._masked_peaks,
        #     'spr': self._spr
        # }

        # # Access variables
        # self.binned_data = None
        # self.norm_data = None
        # self.denoised_data = None
        # self.scaled_data = None
        # self.prepared_data = None
        # self.kernel = None
        # self.norm_kernel = None
        # self.Kspr = None
        # self.scaler = None
        # self.Xcp = None
        # self.norm_Xcp = None
        # self.scaled_Xcp = None
        # self.prepared_Xcp = None
        # self.K = None
        # self.Kcp = None

    def _prepare_data_PIKE(self, data):
        '''
        Prepares the MALDI data for PIKE kernel computation by converting it into the required format.

        Parameters
        ----------
        data : list of SpectrumObject
            The MALDI data to be prepared.

        Returns
        -------
        x : ndarray
            The prepared data in the format required for PIKE kernel computation (samples, positions, intensities).
        '''
        intensities = np.vstack([d['spectrum_intensity'] for d in data])
        mz = np.array([d['spectrum_mz'] for d in data])
        x = np.array([np.stack((mz[i], intensities[i])) for i in range(len(intensities))])
        x = np.transpose(x.astype(float), axes=(0,2,1))
        return x
    
    def _cosine_normalization(self, kernel):
        '''
        Normalizes a kernel matrix using cosine normalization.

        Parameters
        ----------
        kernel : ndarray
            The kernel matrix to be normalized.

        Returns
        -------
        norm_kernel : ndarray
            The cosine-normalized kernel matrix.
        '''
        # diag_K = np.diag(kernel)
        # outer_diag_K = np.sqrt(np.outer(diag_K, diag_K))
        # norm_kernel = kernel / outer_diag_K
        assert kernel.shape[0] == kernel.shape[1]
        
        diag_K = torch.diag(kernel)
        outer_diag_K = torch.sqrt(torch.outer(diag_K, diag_K))
        norm_kernel = kernel / outer_diag_K

        return norm_kernel
    
    def _mask_peaks(self, data, peaks, masking_window):
        '''
        Masks peaks in the MALDI data within a specified window around given peak positions.

        Parameters
        ----------
        data : DataFrame
            The MALDI data in which peaks are to be masked.
        peaks : array-like
            The positions of the peaks to be masked.
        masking_window : int
            The size of the window around each peak position to mask.

        Returns
        -------
        masked_data : DataFrame
            The MALDI data with specified peaks masked.
        '''
        masked_data = data.copy()
        for idx in peaks:
            start_idx = max(0, idx - masking_window)
            end_idx = min(data.shape[1] - 1, idx + masking_window)
                
            columns_to_flatten = [col for col in masked_data.columns if start_idx <= int(col) <= end_idx]
            masked_data.loc[:, columns_to_flatten] = 0
        
        return masked_data
    
    def _create_common_peaks_MALDI(self, binned_data, bin_size):
        '''
        Creates a common peaks matrix from binned MALDI data for use in SPR kernel computation.

        Parameters
        ----------
        binned_data : DataFrame
            The binned MALDI data.
        peaks : array-like
            The positions of the common peaks.
        bin_size : int
            The bin size used for data binning.
        mz_range : array-like
            The range of mz values in Da.

        Returns
        -------
        Xcp : DataFrame
            The common peaks matrix.
        '''
        bin_indices = (self.common_peaks - 2000) // bin_size # Translate mz values to bin indices
        selected_bins = binned_data[bin_indices]
        median_intensities = np.median(selected_bins, axis=0)
    
        # Create common peaks matrix with 0s for missing peaks
        Xcp = np.zeros_like(binned_data[0])
        Xcp[bin_indices] = median_intensities
        
        # Scale it to have a minimum value of 1
        Xcp = Xcp / np.min(Xcp[Xcp > 0])
    
        return Xcp

    def __call__(self, maldi_data, t=4, th=1e-6, masking_window=30):
        '''
        Executes the pipeline on the provided MALDI data with the specified parameters.

        Parameters
        ----------
        maldi_data : DataFrame
            The MALDI data to process.
        bin_size : int, optional
            The bin size to use for data binning. Default is 3.
        t : int, optional
            The smoothing factor parameter for the PIKE kernel. Default is 4.
        th : float, optional
            The threshold for the distance optimization in the E-PIKE kernel. Default is 1e-6.
        masking_window : int, optional
            The window size for masking peaks in Da. Default is 30.

        Returns
        -------
        norm_kernel : ndarray
            The normalized kernel matrix.
        '''
        return self.processing_mode[self.peak_removal](maldi_data, t, th, masking_window)

    def _compute_kernel(self, prepared_data, t, th):
        epike = EfficientPIKE(t=t)
        kernel = epike(prepared_data[:,:,0], prepared_data[:,:,1], th=th)
        norm_kernel = self._cosine_normalization(kernel)
        return norm_kernel.cpu().numpy()
    
    def _no_peak_removal(self, maldi_data, t=4, th=1e-6, masking_window=5):

        self.prepared_data = self._prepare_data_PIKE(maldi_data)

        self.kernel = self._compute_kernel(self.prepared_data, t, th)
        
        return self.kernel
    
    def _masked_peaks(self, maldi_data, t=4, th=1e-6, masking_window=5):
        masked_data = self._mask_peaks(maldi_data, self.common_peaks, masking_window)
        self.prepared_data = self._prepare_data_PIKE(masked_data)

        self.kernel = self._compute_kernel(self.prepared_data, t, th)
        
        return self.kernel
    

    def _spr(self, maldi_data, bin_size, t=4, th=1e-6, masking_window=5):
        # Calculate complete kernel
        self.prepared_data = self._prepare_data_PIKE(maldi_data)

        self.complete_kernel = self._compute_kernel(self.prepared_data, t, th)
        
        # Xcp
        self.Xcp_scaled = self._create_common_peaks_MALDI(maldi_data, bin_size)
        self.prepared_Xcp = self._prepare_data_PIKE(self.scaled_Xcp)

        # Compute K
        epike = EfficientPIKE(t=t)
        self.K = epike(self.prepared_data[:,:,0], self.prepared_data[:,:,1], 
                       self.prepared_Xcp[:,:,0], self.prepared_Xcp[:,:,1], th=th)
        
        # Compute Kcp
        self.Kcp = epike(self.prepared_Xcp[:,:,0], self.prepared_Xcp[:,:,1], th=th)

        # Remove peaks in kernel space
        ones_n = np.ones((1, self.complete_kernel.shape[0]))
        ones_nn = np.ones_like(self.complete_kernel)
        self.Kspr = self.complete_kernel - self.K @ ones_n - ones_n.T @ self.K.T + self.Kcp * ones_nn

        self.norm_Kspr = self._cosine_normalization(self.Kspr)
        return self.norm_Kspr
    
    def get_intermediate_matrices(self):
        return {
            'binned_data': self.binned_data,
            'norm_data': self.norm_data,
            'denoised_data': self.denoised_data,
            'scaled_data': self.scaled_data,
            'prepared_data': self.prepared_data,
            'kernel': self.kernel,
            'norm_kernel': self.norm_kernel,
            'Kspr': self.Kspr,
            'scaler': self.scaler,
            'Xcp': self.Xcp,
            'norm_Xcp': self.norm_Xcp,
            'scaled_Xcp': self.scaled_Xcp,
            'prepared_Xcp': self.prepared_Xcp,
            'K': self.K,
            'Kcp': self.Kcp
        }

