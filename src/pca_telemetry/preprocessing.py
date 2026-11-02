"""Prepare the telemetry table for PCA.

Steps, in order: trim the transients at the start and end of each scenario,
remove incomplete or duplicated rows, drop constant metrics and standardize
the remaining ones (z-score).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

ID_COLUMNS: tuple[str, ...] = ("timestamp", "scenario")
"""Columns that identify a sample and are not metrics."""


def metric_columns(data: pd.DataFrame) -> list[str]:
    """Return the names of the metric columns (everything except the IDs)."""
    return [c for c in data.columns if c not in ID_COLUMNS]


def trim_transients(data: pd.DataFrame, seconds: int = 5) -> pd.DataFrame:
    """Drop the first and last ``seconds`` of every scenario.

    While a load starts or stops the server is in neither state, so these
    samples would mix two behaviors.
    """
    if seconds <= 0:
        return data.copy()

    def _trim(group: pd.DataFrame) -> pd.DataFrame:
        start, end = group["timestamp"].min(), group["timestamp"].max()
        keep = (group["timestamp"] >= start + seconds) & (group["timestamp"] <= end - seconds)
        return group[keep]

    parts = [_trim(g) for _, g in data.groupby("scenario", observed=True, sort=False)]
    return pd.concat(parts).reset_index(drop=True)


def clean(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows with missing values and duplicated samples.

    Returns the clean table and a report with the number of removed rows.
    """
    before = len(data)
    no_missing = data.dropna(subset=metric_columns(data))
    after_missing = len(no_missing)
    no_duplicates = no_missing.drop_duplicates(subset=["scenario", "timestamp"])

    report = {
        "rows_before": before,
        "missing_removed": before - after_missing,
        "duplicates_removed": after_missing - len(no_duplicates),
        "rows_after": len(no_duplicates),
    }
    return no_duplicates.reset_index(drop=True), report


def drop_constant(
    X: pd.DataFrame, tolerance: float = 1e-8
) -> tuple[pd.DataFrame, list[str]]:
    """Drop metrics whose standard deviation is below ``tolerance``.

    A constant metric carries no information and cannot be standardized.
    Returns the reduced matrix and the names of the dropped metrics.
    """
    std = X.std(axis=0, ddof=0)
    dropped = std.index[std <= tolerance].tolist()
    return X.drop(columns=dropped), dropped


def standardize(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale every metric to zero mean and unit variance (z-score).

    The metrics have different units (%, kB, packets/s), so without this step
    the ones with large numbers would dominate the principal components.
    """
    scaler = StandardScaler()
    Xz = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return Xz, scaler


def prepare(
    data: pd.DataFrame, trim_seconds: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Run the full preprocessing pipeline.

    Returns:
        X: metrics after cleaning and removing constants (original units).
        Xz: standardized metrics, the input of PCA.
        y: scenario of each row (for interpretation and baselines).
        report: rows removed, dropped constant metrics and the fitted scaler.
    """
    trimmed = trim_transients(data, trim_seconds)
    cleaned, report = clean(trimmed)
    X, dropped = drop_constant(cleaned[metric_columns(cleaned)])
    Xz, scaler = standardize(X)

    report["transient_rows_removed"] = len(data) - len(trimmed)
    report["constant_metrics"] = dropped
    report["scaler"] = scaler
    return X, Xz, cleaned["scenario"], report
