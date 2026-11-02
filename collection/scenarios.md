# Data collection: load scenarios

This folder contains the scripts that generate the dataset of the project:
Linux telemetry recorded with `sar` (sysstat) while `stress-ng` applies
controlled and reproducible loads.

## Requirements

- Linux server with `sysstat` (`sar`, `sadf`) and `stress-ng` installed.
  - Debian/Ubuntu: `sudo apt install sysstat stress-ng`
  - Fedora/RHEL: `sudo dnf install sysstat stress-ng`
  - Arch: `sudo pacman -S sysstat stress-ng`
- No root privileges are needed to run the scripts.
- Run the experiments when the server is **not serving other users**: the
  loads affect any service running on it.

## Scenarios

| Scenario  | Purpose                                  | Load                                                          |
| :-------- | :--------------------------------------- | :------------------------------------------------------------ |
| `idle`    | Baseline: server without induced load    | `sleep <duration>`                                            |
| `cpu`     | Saturate every CPU core                  | `stress-ng --cpu 0`                                           |
| `memory`  | Occupy most of the RAM (may reach swap)  | `stress-ng --vm 1 --vm-bytes <m>% --vm-keep`                  |
| `disk`    | Intensive disk reads and writes          | `stress-ng --hdd 2 --hdd-bytes 1g --hdd-opts direct`          |
| `network` | Intensive socket traffic over loopback   | `stress-ng --sock 2`                                          |

Notes:

- `--cpu 0` starts one worker per online CPU.
- `memory` uses 75 % of the available memory by default (`-m`, max. 95).
  High values can make the server slow or trigger the OOM killer.
- `disk` writes temporary files in `.stress-tmp/` (option `-t`) with direct
  I/O (`O_DIRECT`), so the writes reach the disk instead of staying in the page
  cache. The directory **must not be on tmpfs**: on Debian 13 `/tmp` is tmpfs
  and the disk metrics would stay at zero. The script stops with an error in
  that case. `stress-ng` removes the files when it finishes.
- `network` generates traffic on the `lo` interface, so it does not saturate the
  office network.

## Parameters

| Option | Meaning                         | Default          |
| :----- | :------------------------------ | :--------------- |
| `-d`   | Duration of each scenario (s)   | `1800` (30 min)  |
| `-i`   | Sampling interval of `sar` (s)  | `1`              |
| `-p`   | Pause between scenarios (s)     | `60`             |
| `-m`   | Memory for `memory` (%)         | `75`             |
| `-o`   | Output directory                | `data/raw`       |
| `-t`   | Scratch directory for `disk`    | `.stress-tmp`    |

With the defaults, a full run takes about 2 h 34 min (5 × 30 min + 4 × 1 min)
and produces 1800 samples per scenario (9000 in total).

## How to reproduce

```bash
# 1. Quick test: 60 s per scenario
./collection/collect.sh -d 60 -p 10 -o /tmp/test_raw
./collection/export_csv.sh -i /tmp/test_raw

# 2. Full collection (run it inside tmux or screen)
./collection/collect.sh

# 3. Export the binary files to CSV
./collection/export_csv.sh
```

Specific scenarios can be run alone, for example `./collection/collect.sh cpu network`.

## Output

```
data/raw/
├── server_specs.txt     # kernel, OS, CPU, RAM, disks, tool versions, parameters
├── run_log.csv          # scenario, start/end epoch, duration, interval, samples
└── <scenario>/
    ├── <scenario>.sa    # binary sar file (not versioned, see .gitignore)
    ├── load.log         # stress-ng output
    ├── cpu.csv          # sar -u
    ├── mem.csv          # sar -r
    ├── io.csv           # sar -b
    ├── task.csv         # sar -w
    └── net.csv          # sar -n DEV (one row per interface and timestamp)
```

The CSV files use `;` as separator and the `timestamp` column is Unix epoch
time (UTC), so the metric groups can be joined by timestamp.

## Variables

| Group  | File       | `sar` option | Main columns                                                 |
| :----- | :--------- | :----------- | :----------------------------------------------------------- |
| CPU    | `cpu.csv`  | `-u`         | `%user`, `%nice`, `%system`, `%iowait`, `%steal`, `%idle`    |
| Memory | `mem.csv`  | `-r`         | `kbmemfree`, `kbavail`, `kbmemused`, `%memused`, `kbbuffers`, `kbcached` |
| Disk   | `io.csv`   | `-b`         | `tps`, `rtps`, `wtps`, `bread/s`, `bwrtn/s`                  |
| Tasks  | `task.csv` | `-w`         | `proc/s`, `cswch/s`                                          |
| Net    | `net.csv`  | `-n DEV`     | `rxpck/s`, `txpck/s`, `rxkB/s`, `txkB/s` (per interface)     |
