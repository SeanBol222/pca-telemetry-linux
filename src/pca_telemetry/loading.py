"""Load the CSV files exported by ``collection/export_csv.sh`` into one table.

Each scenario folder contains one CSV per metric group (``cpu``, ``mem``,
``io``, ``task`` and ``net``). They are joined by ``timestamp`` so that every
row is one sampling instant and every column is one metric.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCENARIOS: tuple[str, ...] = ("idle", "cpu", "memory", "disk", "network")
"""Load scenarios in the order they are collected."""

GROUPS: tuple[str, ...] = ("cpu", "mem", "io", "task")
"""Metric groups with one row per timestamp."""

NET_COLUMNS: tuple[str, ...] = ("rxpck/s", "txpck/s", "rxkB/s", "txkB/s")
"""Network metrics kept from ``sar -n DEV``."""

# Columns written by sadf that describe the sample, not the system.
_META_COLUMNS = ("hostname", "interval", "CPU", "IFACE")


def read_group(path: str | Path) -> pd.DataFrame:
    """Read one metric group CSV and index it by ``timestamp``."""
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=[c for c in _META_COLUMNS if c in df.columns])
    return df.set_index("timestamp").sort_index()


def read_network(
    path: str | Path, exclude_interfaces: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Read ``net.csv`` and add up the traffic of all interfaces per timestamp.

    The loopback interface ``lo`` is kept by default because the ``network``
    scenario generates its traffic there.
    """
    df = pd.read_csv(path, sep=";")
    if exclude_interfaces:
        df = df[~df["IFACE"].isin(exclude_interfaces)]
    return df.groupby("timestamp")[list(NET_COLUMNS)].sum().sort_index()


def load_scenario(
    scenario_dir: str | Path, exclude_interfaces: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Join every metric group of one scenario by ``timestamp``.

    Only timestamps present in all groups are kept.
    """
    scenario_dir = Path(scenario_dir)
    missing = [
        f"{g}.csv" for g in (*GROUPS, "net") if not (scenario_dir / f"{g}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{scenario_dir}: missing {', '.join(missing)}")

    frames = [read_group(scenario_dir / f"{g}.csv") for g in GROUPS]
    frames.append(read_network(scenario_dir / "net.csv", exclude_interfaces))
    return pd.concat(frames, axis=1, join="inner")


def load_dataset(
    raw_dir: str | Path,
    scenarios: tuple[str, ...] = SCENARIOS,
    exclude_interfaces: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load all scenarios into one table.

    Returns a DataFrame with a ``timestamp`` column, one column per metric and
    a categorical ``scenario`` column (used only to interpret the results).
    """
    raw_dir = Path(raw_dir)
    frames = []
    for scenario in scenarios:
        df = load_scenario(raw_dir / scenario, exclude_interfaces)
        df.insert(0, "scenario", scenario)
        frames.append(df)

    data = pd.concat(frames).reset_index()
    data["scenario"] = pd.Categorical(data["scenario"], categories=list(scenarios))
    return data


def save_dataset(data: pd.DataFrame, path: str | Path) -> Path:
    """Save the merged table as CSV (comma separated) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path
