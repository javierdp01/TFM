import numpy as np
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

class TSNESpectre:
    def __init__(self,
                 n_points: int = 4096,
                 tic_norm: bool = True,
                 do_log1p: bool = True,
                 pca_dims: int = 50,
                 perplexity: float = 30.0,
                 seed: int = 0,
                 max_iter: int = 2500):
        self.n_points = int(n_points) 
        self.tic_norm = tic_norm
        self.do_log1p = do_log1p
        self.pca_dims = pca_dims
        self.perplexity = perplexity
        self.seed = seed
        self.max_iter = max_iter
        self.grid_: Optional[np.ndarray] = None
        self.pca_: Optional[PCA] = None
        self.tsne_: Optional[TSNE] = None

    # ---- Plantilla común ----
    def _build_common_grid(self, spectra: List[object],
                           mz_min: Optional[float] = None,
                           mz_max: Optional[float] = None) -> np.ndarray:
        """
        Función para crear una plantilla común para los datos a representar y evitar problemas de tamaño
        """
        if mz_min is None or mz_max is None:
            mins = [float(np.min(s.mz)) for s in spectra]
            maxs = [float(np.max(s.mz)) for s in spectra]
            mz_min, mz_max = float(np.min(mins)), float(np.max(maxs))
        return np.linspace(mz_min, mz_max, self.n_points, dtype=np.float32)

    # ---- Remuestreo a la plantilla ----
    def _resample_to_grid(self, spec: object, grid: np.ndarray) -> np.ndarray:
        x, y = spec.mz, spec.intensity
        if x[0] > x[-1]:
            x = x[::-1]; y = y[::-1]
        return np.interp(grid, x, y, left=0.0, right=0.0).astype(np.float32)

    # ---- Featurización ----
    def featurize(self, spectra: List[object], grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.grid_ = grid if grid is not None else self._build_common_grid(spectra)  # <-- corregido
        X = np.stack([self._resample_to_grid(s, self.grid_) for s in spectra], axis=0)
        if self.tic_norm:
            X = X / (X.sum(axis=1, keepdims=True) + 1e-12)
        if self.do_log1p:
            X = np.log1p(X)
        return X, self.grid_

    # ---- PCA + t-SNE ----
    def run_tsne(self, X: np.ndarray) -> np.ndarray:
        self.pca_ = PCA(n_components=min(self.pca_dims, X.shape[1]), random_state=self.seed)
        X_pca = self.pca_.fit_transform(X)
        self.tsne_ = TSNE(
            n_components=2,
            perplexity=min(self.perplexity, max(5, X.shape[0] // 3)),
            learning_rate="auto",
            init="pca",
            max_iter=self.max_iter,
            random_state=self.seed,
            metric="euclidean",
        )
        Z = self.tsne_.fit_transform(X_pca)
        return Z

    # ---- Atajo: todo en uno ----
    def fit_transform(self, spectra: List[object]) -> Tuple[np.ndarray, np.ndarray]:
        X, grid = self.featurize(spectra)
        Z = self.run_tsne(X)
        return Z, grid

    # ---- Plot ----
    def plot(self, Z: np.ndarray, labels=None, class_names=None, title="t-SNE de espectros", show=False, type="random"):
        plt.figure(figsize=(7, 6))

        if labels is None:
            plt.scatter(Z[:, 0], Z[:, 1], s=20, alpha=0.85)
        else:
            labels = np.array(labels)
            for c in np.unique(labels):
                idx = labels == c
                name = class_names[c] if (class_names is not None and c < len(class_names)) else str(c)
                plt.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.9, label=name)
            plt.legend(frameon=False)

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2") 
        plt.title(f"{title} {type}")
        plt.tight_layout()

        save_plot_as = f"{title}_{type}.jpg"
        path = r"C:\Users\javie\Desktop\ImagenesTFM"
        os.makedirs(path, exist_ok=True)
        path_to_save = os.path.join(path, save_plot_as)
        print("Guardando la figura generada en " + path_to_save)

        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()