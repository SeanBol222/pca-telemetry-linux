from pathlib import Path

import pandas as pd
import pytest

from pca_telemetry.loading import load_dataset, load_scenario, read_network


def write_scenario(folder: Path, timestamps: list[int]) -> None:
    """Write the five CSV files in the format produced by export_csv.sh."""
    folder.mkdir(parents=True)
    head = "hostname;interval;timestamp"
    files = {
        "cpu": (f"{head};CPU;%user;%idle", lambda t: f"-1;{t % 7}.00;{100 - t % 7}.00"),
        "mem": (f"{head};kbmemfree;%memused", lambda t: f"{1000 + t};{t % 5}.50"),
        "io": (f"{head};tps;bwrtn/s", lambda t: f"{t % 3}.00;{t * 2}.00"),
        "task": (f"{head};proc/s;cswch/s", lambda t: f"1.00;{500 + t}.00"),
    }
    for name, (header, row) in files.items():
        lines = [header] + [f"srv;1;{t};{row(t)}" for t in timestamps]
        (folder / f"{name}.csv").write_text("\n".join(lines) + "\n")

    net = [f"{head};IFACE;rxpck/s;txpck/s;rxkB/s;txkB/s;rxcmp/s;txcmp/s;rxmcst/s;%ifutil"]
    for t in timestamps:
        net.append(f"srv;1;{t};lo;10.00;10.00;1.00;1.00;0.00;0.00;0.00;0.00")
        net.append(f"srv;1;{t};eth0;5.00;3.00;0.50;0.25;0.00;0.00;0.00;0.00")
    (folder / "net.csv").write_text("\n".join(net) + "\n")


def test_network_traffic_is_summed_over_interfaces(tmp_path):
    write_scenario(tmp_path / "idle", [1, 2, 3])
    net = read_network(tmp_path / "idle" / "net.csv")
    assert list(net.columns) == ["rxpck/s", "txpck/s", "rxkB/s", "txkB/s"]
    assert net.loc[1, "rxpck/s"] == pytest.approx(15.0)
    assert net.loc[1, "txkB/s"] == pytest.approx(1.25)


def test_network_can_exclude_interfaces(tmp_path):
    write_scenario(tmp_path / "idle", [1, 2])
    net = read_network(tmp_path / "idle" / "net.csv", exclude_interfaces=("lo",))
    assert net.loc[1, "rxpck/s"] == pytest.approx(5.0)


def test_scenario_groups_are_joined_by_timestamp(tmp_path):
    write_scenario(tmp_path / "cpu", [10, 11, 12, 13])
    df = load_scenario(tmp_path / "cpu")
    assert len(df) == 4
    assert {"hostname", "interval", "CPU", "IFACE"}.isdisjoint(df.columns)
    assert {"%user", "%memused", "tps", "cswch/s", "rxpck/s"} <= set(df.columns)


def test_missing_group_file_raises(tmp_path):
    write_scenario(tmp_path / "cpu", [1, 2])
    (tmp_path / "cpu" / "io.csv").unlink()
    with pytest.raises(FileNotFoundError, match="io.csv"):
        load_scenario(tmp_path / "cpu")


def test_dataset_has_scenario_column_in_order(tmp_path):
    write_scenario(tmp_path / "idle", [1, 2, 3])
    write_scenario(tmp_path / "cpu", [100, 101])
    data = load_dataset(tmp_path, scenarios=("idle", "cpu"))
    assert list(data.columns[:2]) == ["timestamp", "scenario"]
    assert isinstance(data["scenario"].dtype, pd.CategoricalDtype)
    assert data["scenario"].value_counts().to_dict() == {"idle": 3, "cpu": 2}
