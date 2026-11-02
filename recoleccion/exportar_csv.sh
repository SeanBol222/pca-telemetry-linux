#!/usr/bin/env bash
# Export the binary sar files (.sa) of each scenario to CSV with sadf.
#
# For every <dir>/<scenario>/<scenario>.sa it writes one CSV per metric group
# in the same folder:
#   cpu.csv   sar -u       %user %nice %system %iowait %steal %idle
#   mem.csv   sar -r       kbmemfree kbavail kbmemused %memused kbbuffers kbcached ...
#   io.csv    sar -b       tps rtps wtps dtps bread/s bwrtn/s bdscd/s
#   task.csv  sar -w       proc/s cswch/s
#   net.csv   sar -n DEV   IFACE rxpck/s txpck/s rxkB/s txkB/s ... (one row per interface)
#
# CSV files use ";" as separator and a "timestamp" column in Unix epoch (UTC).
#
# Usage:
#   ./exportar_csv.sh [-i DIR]
#
# Options:
#   -i DIR   directory with one folder per scenario (default: <repo>/data/raw)
#   -h       show this help

set -euo pipefail

# Use the C locale so numbers keep a dot as decimal separator and the
# command output (lscpu, free, ...) is in English, whatever the system language.
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDIR="$REPO_ROOT/data/raw"

usage() {
    sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while getopts ":i:h" opt; do
    case "$opt" in
        i) INDIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) echo "Error: option -$OPTARG requires a value." >&2; exit 1 ;;
        *) echo "Error: unknown option -$OPTARG." >&2; usage; exit 1 ;;
    esac
done

if ! command -v sadf >/dev/null 2>&1; then
    echo "Error: 'sadf' is not installed (package sysstat)." >&2
    exit 1
fi

# group name -> sar options
declare -A GROUPS_OPTS=(
    [cpu]="-u"
    [mem]="-r"
    [io]="-b"
    [task]="-w"
    [net]="-n DEV"
)

export_group() {
    local sa_file="$1" opts="$2" out="$3"
    # -d: CSV-like output, -U: timestamps in epoch seconds.
    # Drop the "# " before the header and any LINUX-RESTART marker lines.
    # shellcheck disable=SC2086
    sadf -d -U "$sa_file" -- $opts \
        | grep -v 'LINUX-RESTART' \
        | sed '1s/^# //' > "$out"
}

found=0
for scenario_dir in "$INDIR"/*/; do
    scenario="$(basename "$scenario_dir")"
    sa_file="$scenario_dir$scenario.sa"
    [[ -f "$sa_file" ]] || continue
    found=$(( found + 1 ))

    echo "$scenario:"
    for group in cpu mem io task net; do
        out="$scenario_dir$group.csv"
        export_group "$sa_file" "${GROUPS_OPTS[$group]}" "$out"
        rows=$(( $(wc -l < "$out") - 1 ))
        echo "  $group.csv ($rows rows)"
    done
done

if (( found == 0 )); then
    echo "No .sa files found in $INDIR/<scenario>/. Run ./recoleccion/recolectar.sh first." >&2
    exit 1
fi

echo "Export finished: $found scenario(s)."
