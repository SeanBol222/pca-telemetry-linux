"""Figures for the notebook and the slides.

Every function returns a matplotlib Figure with title, labeled axes (with
units) and legend. ``save_figure`` writes it as PNG (300 dpi) and PDF.

Figure text is in Spanish by default because the figures go to the
presentation; pass ``lang="en"`` for English.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

SCENARIO_COLORS: dict[str, str] = {
    "idle": "#4c72b0",
    "cpu": "#dd8452",
    "memory": "#55a868",
    "disk": "#c44e52",
    "network": "#8172b3",
}

METRIC_UNITS: dict[str, str] = {
    "%user": "%", "%nice": "%", "%system": "%", "%iowait": "%", "%steal": "%", "%idle": "%",
    "kbmemfree": "kB", "kbavail": "kB", "kbmemused": "kB", "%memused": "%",
    "kbbuffers": "kB", "kbcached": "kB", "kbcommit": "kB", "%commit": "%",
    "kbactive": "kB", "kbinact": "kB", "kbdirty": "kB", "kbshmem": "kB",
    "tps": "transfers/s", "rtps": "transfers/s", "wtps": "transfers/s", "dtps": "transfers/s",
    "bread/s": "blocks/s", "bwrtn/s": "blocks/s", "bdscd/s": "blocks/s",
    "proc/s": "tasks/s", "cswch/s": "switches/s",
    "rxpck/s": "packets/s", "txpck/s": "packets/s", "rxkB/s": "kB/s", "txkB/s": "kB/s",
}

_TEXT = {
    "es": {
        "corr_title": "Correlación entre métricas de sar",
        "corr_bar": "Coeficiente de correlación de Pearson r",
        "scree_title": "Varianza explicada por componente principal",
        "component": "Componente principal",
        "explained": "Varianza explicada (%)",
        "per_component": "Por componente",
        "cumulative": "Acumulada",
        "threshold": "Umbral {:.0f} %",
        "load_title": "Cargas de las métricas en las componentes principales",
        "metric": "Métrica",
        "load_bar": "Carga (peso de la métrica en la componente)",
        "proj_title": "Muestras proyectadas en PC1 y PC2",
        "pc_axis": "{} ({:.1f} % de la varianza)",
        "scenario": "Escenario",
        "rec_title": "{}: original vs. reconstruida con k = {}",
        "time": "Tiempo (s)",
        "original": "Original",
        "reconstructed": "Reconstruida",
        "curve_title": "Error de reconstrucción según el número de componentes",
        "curve_error": "Error relativo de reconstrucción (%)",
        "cmp_title": "Comparación de métodos",
        "method": "Método",
        "scenarios": {"idle": "Reposo", "cpu": "CPU", "memory": "Memoria",
                      "disk": "Disco", "network": "Red"},
        "methods": {"all_metrics": "Todas las métricas", "pca": "PCA",
                    "correlation_subset": "Filtro de correlación"},
    },
    "en": {
        "corr_title": "Correlation between sar metrics",
        "corr_bar": "Pearson correlation coefficient r",
        "scree_title": "Explained variance per principal component",
        "component": "Principal component",
        "explained": "Explained variance (%)",
        "per_component": "Per component",
        "cumulative": "Cumulative",
        "threshold": "{:.0f} % threshold",
        "load_title": "Metric loadings on the principal components",
        "metric": "Metric",
        "load_bar": "Loading (weight of the metric in the component)",
        "proj_title": "Samples projected on PC1 and PC2",
        "pc_axis": "{} ({:.1f} % of the variance)",
        "scenario": "Scenario",
        "rec_title": "{}: original vs. reconstructed with k = {}",
        "time": "Time (s)",
        "original": "Original",
        "reconstructed": "Reconstructed",
        "curve_title": "Reconstruction error versus number of components",
        "curve_error": "Relative reconstruction error (%)",
        "cmp_title": "Method comparison",
        "method": "Method",
        "scenarios": {"idle": "Idle", "cpu": "CPU", "memory": "Memory",
                      "disk": "Disk", "network": "Network"},
        "methods": {"all_metrics": "All metrics", "pca": "PCA",
                    "correlation_subset": "Correlation filter"},
    },
}


def _t(lang: str) -> dict:
    if lang not in _TEXT:
        raise ValueError(f"lang must be one of {sorted(_TEXT)}")
    return _TEXT[lang]


def save_figure(fig: Figure, name: str, outdir: str | Path = "results/figures") -> list[Path]:
    """Save ``fig`` as ``<outdir>/<name>.png`` (300 dpi) and ``.pdf``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{name}.png", outdir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def plot_correlation(Xz: pd.DataFrame, lang: str = "es") -> Figure:
    """Heatmap of the correlation matrix of the metrics."""
    t = _t(lang)
    corr = Xz.corr()
    size = max(6.0, 0.35 * len(corr))
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.set_title(t["corr_title"])
    fig.colorbar(image, ax=ax, label=t["corr_bar"], shrink=0.8)
    fig.tight_layout()
    return fig


def plot_scree(pca: PCA, threshold: float = 0.9, max_components: int | None = None,
               lang: str = "es") -> Figure:
    """Bars with the variance of each component and the cumulative curve."""
    t = _t(lang)
    ratio = pca.explained_variance_ratio_ * 100
    n = len(ratio) if max_components is None else min(max_components, len(ratio))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, ratio[:n], color="#4c72b0", label=t["per_component"])
    ax.plot(x, np.cumsum(ratio)[:n], color="#dd8452", marker="o", label=t["cumulative"])
    ax.axhline(threshold * 100, color="gray", linestyle="--", label=t["threshold"].format(threshold * 100))
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.set_xlabel(t["component"])
    ax.set_ylabel(t["explained"])
    ax.set_title(t["scree_title"])
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_loadings(load: pd.DataFrame, lang: str = "es") -> Figure:
    """Heatmap of the loadings: metrics (rows) by components (columns)."""
    t = _t(lang)
    limit = float(np.abs(load.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(1.0 * load.shape[1] + 3, 0.32 * load.shape[0] + 1.5))
    image = ax.imshow(load, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(load.shape[1]), load.columns)
    ax.set_yticks(range(load.shape[0]), load.index, fontsize=8)
    ax.set_xlabel(t["component"])
    ax.set_ylabel(t["metric"])
    ax.set_title(t["load_title"])
    fig.colorbar(image, ax=ax, label=t["load_bar"])
    fig.tight_layout()
    return fig


def plot_projection(Z: pd.DataFrame, y: pd.Series, pca: PCA, lang: str = "es") -> Figure:
    """Scatter of the samples on PC1 and PC2, colored by load scenario."""
    t = _t(lang)
    ratio = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = np.asarray(y)
    for scenario in pd.unique(labels):
        mask = labels == scenario
        ax.scatter(Z.iloc[mask, 0], Z.iloc[mask, 1], s=6, alpha=0.5,
                   color=SCENARIO_COLORS.get(scenario), label=t["scenarios"].get(scenario, scenario))
    ax.set_xlabel(t["pc_axis"].format("PC1", ratio[0]))
    ax.set_ylabel(t["pc_axis"].format("PC2", ratio[1]))
    ax.set_title(t["proj_title"])
    ax.legend(title=t["scenario"], markerscale=3)
    fig.tight_layout()
    return fig


def plot_reconstruction(original: pd.Series, reconstructed: pd.Series, metric: str, k: int,
                        lang: str = "es") -> Figure:
    """Time series of one metric and its reconstruction, in original units."""
    t = _t(lang)
    unit = METRIC_UNITS.get(metric, "")
    seconds = np.arange(len(original))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(seconds, np.asarray(original), color="#4c72b0", linewidth=1, label=t["original"])
    ax.plot(seconds, np.asarray(reconstructed), color="#dd8452", linewidth=1,
            linestyle="--", label=t["reconstructed"])
    ax.set_xlabel(t["time"])
    ax.set_ylabel(f"{metric} ({unit})" if unit else metric)
    ax.set_title(t["rec_title"].format(metric, k))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reconstruction_curve(curve: pd.DataFrame, k: int | None = None,
                              lang: str = "es") -> Figure:
    """Relative reconstruction error for every k, optionally marking one k."""
    t = _t(lang)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["k"], curve["relative_error"] * 100, color="#4c72b0", marker="o")
    if k is not None:
        error = float(curve.loc[curve["k"] == k, "relative_error"].iloc[0]) * 100
        ax.axvline(k, color="gray", linestyle="--", label=f"k = {k} ({error:.1f} %)")
        ax.legend()
    ax.set_xticks(curve["k"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel(t["component"])
    ax.set_ylabel(t["curve_error"])
    ax.set_title(t["curve_title"])
    fig.tight_layout()
    return fig


def plot_method_comparison(table: pd.DataFrame, value: str, ylabel: str,
                           error: str | None = None, lang: str = "es") -> Figure:
    """Bar chart comparing methods (rows of ``table``) on one metric."""
    t = _t(lang)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = table[error] if error else None
    # "correlation_filter(|r|>0.9)" and "correlation_subset" share one label.
    names = [
        t["methods"]["correlation_subset"] if m.startswith("correlation") else t["methods"].get(m, m)
        for m in table["method"]
    ]
    bars = ax.bar(names, table[value], yerr=yerr, capsize=4,
                  color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: len(table)])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel(t["method"])
    ax.set_ylabel(ylabel)
    ax.set_title(t["cmp_title"])
    fig.tight_layout()
    return fig
