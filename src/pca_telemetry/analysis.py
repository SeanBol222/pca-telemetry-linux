"""PCA on the standardized telemetry and the quantitative results.

Notation (rows = sampling instants, columns = metrics):
    Xz   standardized data matrix, N x P
    W_k  first k eigenvectors of the covariance matrix (loadings), P x k
    Z    projection Z = Xz W_k, N x k
    X^   reconstruction X^ = Z W_k^T
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.80, 0.90, 0.95)


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------

def fit_pca(Xz: pd.DataFrame) -> PCA:
    """Fit a PCA with all the components on the standardized matrix."""
    return PCA().fit(Xz)


def variance_table(pca: PCA) -> pd.DataFrame:
    """Explained variance ratio of each component and its cumulative sum."""
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(ratio) + 1)],
            "eigenvalue": pca.explained_variance_,
            "explained_variance": ratio,
            "cumulative_variance": np.cumsum(ratio),
        }
    )


def n_components_for(pca: PCA, threshold: float) -> int:
    """Smallest k whose cumulative explained variance reaches ``threshold``."""
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold - 1e-12) + 1)


def components_table(
    pca: PCA, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """k needed for each threshold and the corresponding reduction k/P."""
    n_metrics = pca.n_features_in_
    rows = []
    for threshold in thresholds:
        k = n_components_for(pca, threshold)
        rows.append(
            {
                "threshold": threshold,
                "k": k,
                "n_metrics": n_metrics,
                "reduction_ratio": k / n_metrics,
                "cumulative_variance": float(np.sum(pca.explained_variance_ratio_[:k])),
            }
        )
    return pd.DataFrame(rows)


def project(Xz: pd.DataFrame, pca: PCA, k: int) -> pd.DataFrame:
    """Coordinates of every sample on the first k components (Z)."""
    Z = (Xz.to_numpy() - pca.mean_) @ pca.components_[:k].T
    return pd.DataFrame(Z, index=Xz.index, columns=[f"PC{i}" for i in range(1, k + 1)])


def reconstruct(Xz: pd.DataFrame, pca: PCA, k: int) -> pd.DataFrame:
    """Approximation of Xz using only the first k components (X^)."""
    W = pca.components_[:k]
    Xc = Xz.to_numpy() - pca.mean_
    X_hat = Xc @ W.T @ W + pca.mean_
    return pd.DataFrame(X_hat, index=Xz.index, columns=Xz.columns)


def relative_error(X: pd.DataFrame | np.ndarray, X_hat: pd.DataFrame | np.ndarray) -> float:
    """Relative reconstruction error ||X - X^||_F / ||X||_F."""
    X, X_hat = np.asarray(X, dtype=float), np.asarray(X_hat, dtype=float)
    return float(np.linalg.norm(X - X_hat) / np.linalg.norm(X))


def reconstruction_curve(Xz: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    """Relative reconstruction error for every k from 1 to P."""
    rows = [
        {"k": k, "relative_error": relative_error(Xz, reconstruct(Xz, pca, k))}
        for k in range(1, pca.n_components_ + 1)
    ]
    return pd.DataFrame(rows)


def loadings(pca: PCA, columns: list[str] | pd.Index, k: int | None = None) -> pd.DataFrame:
    """Weight of every metric (rows) on each of the first k components (columns)."""
    k = pca.n_components_ if k is None else k
    return pd.DataFrame(
        pca.components_[:k].T,
        index=list(columns),
        columns=[f"PC{i}" for i in range(1, k + 1)],
    )


def dominant_metrics(load: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The n metrics with the largest absolute loading on each component."""
    rows = []
    for component in load.columns:
        top = load[component].abs().sort_values(ascending=False).head(n).index
        for rank, metric in enumerate(top, start=1):
            rows.append(
                {
                    "component": component,
                    "rank": rank,
                    "metric": metric,
                    "loading": float(load.loc[metric, component]),
                }
            )
    return pd.DataFrame(rows)
