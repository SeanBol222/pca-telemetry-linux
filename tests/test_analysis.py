import numpy as np
import pandas as pd
import pytest

from pca_telemetry import analysis
from pca_telemetry.preprocessing import prepare


@pytest.fixture
def prepared(telemetry):
    X, Xz, y, _ = prepare(telemetry)
    return X, Xz, y, analysis.fit_pca(Xz)


def test_identical_metrics_need_a_single_component():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=500)
    Xz = pd.DataFrame({"a": signal, "b": signal, "c": signal})
    pca = analysis.fit_pca(Xz)
    assert analysis.n_components_for(pca, 0.9) == 1
    assert analysis.variance_table(pca)["explained_variance"].iloc[0] == pytest.approx(1.0)


def test_cumulative_variance_ends_at_one(prepared):
    *_, pca = prepared
    table = analysis.variance_table(pca)
    assert table["cumulative_variance"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_variance"].is_monotonic_increasing


def test_components_table_meets_each_threshold(prepared):
    *_, pca = prepared
    table = analysis.components_table(pca)
    assert (table["cumulative_variance"] >= table["threshold"] - 1e-12).all()
    assert table["k"].is_monotonic_increasing


def test_reconstruction_error_decreases_to_zero(prepared):
    _, Xz, _, pca = prepared
    curve = analysis.reconstruction_curve(Xz, pca)
    assert curve["relative_error"].is_monotonic_decreasing
    assert curve["relative_error"].iloc[-1] == pytest.approx(0, abs=1e-10)


def test_error_matches_unexplained_variance(prepared):
    _, Xz, _, pca = prepared
    k = 2
    expected = np.sqrt(1 - pca.explained_variance_ratio_[:k].sum())
    error = analysis.relative_error(Xz, analysis.reconstruct(Xz, pca, k))
    assert error == pytest.approx(expected, rel=1e-6)


def test_loadings_have_unit_norm(prepared):
    _, Xz, _, pca = prepared
    load = analysis.loadings(pca, Xz.columns, k=3)
    assert load.shape == (Xz.shape[1], 3)
    assert np.allclose(np.linalg.norm(load, axis=0), 1)
    top = analysis.dominant_metrics(load, n=2)
    assert len(top) == 6


def test_correlation_filter_drops_redundant_metrics(prepared):
    _, Xz, _, _ = prepared
    kept = analysis.correlation_filter(Xz, threshold=0.9)
    # %idle mirrors %user, the memory, disk and network pairs are redundant.
    assert "%idle" not in kept
    assert len(kept) < Xz.shape[1]


def test_pca_reconstructs_at_least_as_well_as_the_filter(prepared):
    _, Xz, _, pca = prepared
    table = analysis.compare_reconstruction(Xz, pca).set_index("method")
    assert table.loc["pca", "relative_error"] <= table.iloc[1]["relative_error"] + 1e-12


def test_time_block_folds_are_contiguous_and_balanced(prepared):
    _, _, y, _ = prepared
    folds = analysis.time_block_folds(y, n_splits=5)
    assert set(folds) == set(range(5))
    for label in pd.unique(y):
        f = folds[np.asarray(y == label)]
        assert np.all(np.diff(f) >= 0)


def test_classifier_comparison_reports_every_method(prepared):
    X, Xz, y, pca = prepared
    k = analysis.n_components_for(pca, 0.9)
    table = analysis.compare_classifiers(X, y, k, analysis.correlation_filter(Xz))
    assert list(table["method"]) == ["all_metrics", "pca", "correlation_subset"]
    assert table["accuracy_mean"].between(0, 1).all()
    assert table.loc[table["method"] == "pca", "n_features"].item() == k


def test_clustering_recovers_well_separated_scenarios(prepared):
    _, Xz, y, pca = prepared
    k = analysis.n_components_for(pca, 0.9)
    table = analysis.compare_clustering(Xz, pca, k, y)
    assert (table["adjusted_rand_index"] > 0.9).all()
    assert table["silhouette"].between(-1, 1).all()
