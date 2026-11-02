import numpy as np
import pandas as pd
import pytest

from pca_telemetry.preprocessing import (
    clean,
    drop_constant,
    metric_columns,
    prepare,
    standardize,
    trim_transients,
)


def test_trim_removes_seconds_at_both_ends_of_each_scenario(telemetry):
    trimmed = trim_transients(telemetry, seconds=5)
    # 200 samples per scenario, 5 s removed at each end (inclusive bounds).
    assert trimmed["scenario"].value_counts().unique().tolist() == [190]


def test_clean_reports_missing_and_duplicated_rows(telemetry):
    dirty = telemetry.copy()
    dirty.loc[0, "%user"] = np.nan
    dirty = pd.concat([dirty, dirty.iloc[[5]]], ignore_index=True)
    cleaned, report = clean(dirty)
    assert report["missing_removed"] == 1
    assert report["duplicates_removed"] == 1
    assert len(cleaned) == len(telemetry) - 1


def test_constant_metrics_are_dropped(telemetry):
    X, dropped = drop_constant(telemetry[metric_columns(telemetry)])
    assert dropped == ["%steal"]
    assert "%steal" not in X.columns


def test_standardize_gives_zero_mean_and_unit_variance(telemetry):
    X, _ = drop_constant(telemetry[metric_columns(telemetry)])
    Xz, _ = standardize(X)
    assert np.allclose(Xz.mean(), 0, atol=1e-12)
    assert np.allclose(Xz.std(ddof=0), 1)


def test_prepare_returns_aligned_outputs(telemetry):
    X, Xz, y, report = prepare(telemetry, trim_seconds=5)
    assert len(X) == len(Xz) == len(y) == 5 * 190
    assert list(X.columns) == list(Xz.columns)
    assert report["constant_metrics"] == ["%steal"]
    assert report["transient_rows_removed"] == 5 * 10
    assert Xz.mean().abs().max() == pytest.approx(0, abs=1e-12)
