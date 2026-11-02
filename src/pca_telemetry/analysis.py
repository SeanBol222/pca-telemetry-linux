"""PCA on the standardized telemetry and the quantitative results.

Notation (rows = sampling instants, columns = metrics):
    Xz   standardized data matrix, N x P
    W_k  first k eigenvectors of the covariance matrix (loadings), P x k
    Z    projection Z = Xz W_k, N x k
    X^   reconstruction X^ = Z W_k^T

The module also contains the baselines and the clustering comparison used to
judge whether the reduced space keeps the behavior of the server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


# --------------------------------------------------------------------------
# Baseline 1: select metrics by correlation and reconstruct the rest
# --------------------------------------------------------------------------

def correlation_filter(Xz: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Keep a metric only if its |r| with every metric already kept is <= threshold.

    Metrics are visited in column order, so the result is deterministic.
    """
    corr = Xz.corr().abs()
    kept: list[str] = []
    for metric in Xz.columns:
        if all(corr.loc[metric, other] <= threshold for other in kept):
            kept.append(metric)
    return kept


def subset_reconstruction(Xz: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Best linear reconstruction of all metrics from a subset of them.

    Solves Xz ~ S B by least squares, where S holds the selected columns.
    """
    S = Xz[columns].to_numpy()
    B, *_ = np.linalg.lstsq(S, Xz.to_numpy(), rcond=None)
    return pd.DataFrame(S @ B, index=Xz.index, columns=Xz.columns)


def compare_reconstruction(
    Xz: pd.DataFrame, pca: PCA, threshold: float = 0.9
) -> pd.DataFrame:
    """Reconstruction error of PCA versus the correlation filter, with the same k.

    The correlation filter decides k (number of metrics it keeps); PCA uses the
    same number of components so both methods have the same dimension.
    """
    subset = correlation_filter(Xz, threshold)
    k = len(subset)
    return pd.DataFrame(
        [
            {
                "method": "pca",
                "k": k,
                "relative_error": relative_error(Xz, reconstruct(Xz, pca, k)),
            },
            {
                "method": f"correlation_filter(|r|>{threshold})",
                "k": k,
                "relative_error": relative_error(Xz, subset_reconstruction(Xz, subset)),
            },
        ]
    )


# --------------------------------------------------------------------------
# Baseline 2: classify the load scenario with all metrics, PCA or a subset
# --------------------------------------------------------------------------

def time_block_folds(y: pd.Series, n_splits: int = 5) -> np.ndarray:
    """Fold index of every row using contiguous time blocks per scenario.

    Rows of each scenario (already in time order) are split into ``n_splits``
    consecutive blocks; block i of every scenario goes to fold i. Consecutive
    seconds are almost identical, so random folds would leak information
    between training and test and inflate the accuracy.
    """
    folds = np.empty(len(y), dtype=int)
    positions = np.arange(len(y))
    for label in pd.unique(y):
        idx = positions[np.asarray(y == label)]
        for fold, block in enumerate(np.array_split(idx, n_splits)):
            folds[block] = fold
    return folds


def _cv_accuracy(model, X: pd.DataFrame, y: pd.Series, folds: np.ndarray) -> np.ndarray:
    return cross_val_score(model, X, y, cv=PredefinedSplit(folds), scoring="accuracy")


def compare_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    subset: list[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Accuracy of a logistic regression that predicts the load scenario.

    Compares the P original metrics, the first k principal components and,
    optionally, a subset of original metrics. Scaling and PCA are fitted
    inside each fold to avoid leakage. ``X`` is in original units.
    """
    folds = time_block_folds(y, n_splits)
    classifier = LogisticRegression(max_iter=2000)

    candidates = {
        "all_metrics": (make_pipeline(StandardScaler(), classifier), X),
        "pca": (make_pipeline(StandardScaler(), PCA(n_components=k), classifier), X),
    }
    if subset is not None:
        candidates["correlation_subset"] = (
            make_pipeline(StandardScaler(), classifier),
            X[subset],
        )

    rows = []
    for method, (model, features) in candidates.items():
        scores = _cv_accuracy(model, features, y, folds)
        n_features = k if method == "pca" else features.shape[1]
        rows.append(
            {
                "method": method,
                "n_features": n_features,
                "accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std()),
                "n_folds": len(scores),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Clustering: is the behavior still recognizable in the reduced space?
# --------------------------------------------------------------------------

def clustering_scores(
    features: pd.DataFrame | np.ndarray,
    y: pd.Series,
    n_clusters: int | None = None,
    sample_size: int | None = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    """K-Means on ``features`` evaluated with Silhouette and Adjusted Rand Index.

    Silhouette (-1 to 1) measures how separated the clusters are without using
    the labels. ARI (about 0 = random, 1 = perfect) measures how well the
    clusters match the real load scenarios.
    """
    n_clusters = int(pd.Series(y).nunique()) if n_clusters is None else n_clusters
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(features)
    sample = None if sample_size is None or len(labels) <= sample_size else sample_size
    return {
        "silhouette": float(
            silhouette_score(features, labels, sample_size=sample, random_state=random_state)
        ),
        "adjusted_rand_index": float(adjusted_rand_score(y, labels)),
    }


def compare_clustering(
    Xz: pd.DataFrame,
    pca: PCA,
    k: int,
    y: pd.Series,
    n_clusters: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Clustering quality with all standardized metrics versus k components.

    Same comparison as Lee et al. (2023): K-Means with and without PCA.
    """
    rows = []
    for method, features in (
        ("all_metrics", Xz),
        ("pca", project(Xz, pca, k)),
    ):
        scores = clustering_scores(features, y, n_clusters, random_state=random_state)
        rows.append({"method": method, "n_features": features.shape[1], **scores})
    return pd.DataFrame(rows)
