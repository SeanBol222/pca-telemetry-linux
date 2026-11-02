"""Shared fixtures: small synthetic datasets that mimic the sar telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Mean %user, %memused, disk tps and network packets/s of each scenario.
PROFILES = {
    "idle": (1.0, 5.0, 0.0, 0.0),
    "cpu": (99.0, 5.0, 0.0, 0.0),
    "memory": (50.0, 66.0, 0.0, 0.0),
    "disk": (30.0, 5.0, 10_000.0, 0.0),
    "network": (25.0, 5.0, 0.0, 100_000.0),
}


def make_telemetry(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Five scenarios with correlated metrics, a constant one and a 60 s gap."""
    rng = np.random.default_rng(seed)
    rows, start = [], 1_700_000_000
    for scenario, (user, mem, tps, pkts) in PROFILES.items():
        for second in range(n):
            noise = rng.normal()
            rows.append(
                {
                    "timestamp": start + second,
                    "scenario": scenario,
                    "%user": user + 3 * noise,
                    "%idle": 100 - user - 3 * noise,
                    "%memused": mem + rng.normal(),
                    "kbmemused": mem * 1e5 + 1e5 * rng.normal(),
                    "tps": tps + 50 * rng.normal(),
                    "bwrtn/s": tps * 80 + 4_000 * rng.normal(),
                    "rxpck/s": pkts + 100 * rng.normal(),
                    "txpck/s": pkts + 100 * rng.normal(),
                    "%steal": 0.0,
                }
            )
        start += n + 60
    data = pd.DataFrame(rows)
    data["scenario"] = pd.Categorical(data["scenario"], categories=list(PROFILES))
    return data


@pytest.fixture
def telemetry() -> pd.DataFrame:
    return make_telemetry()
