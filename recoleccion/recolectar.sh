#!/usr/bin/env bash
# Collect Linux telemetry with sar while stress-ng applies controlled loads.
#
# For each scenario the script starts sar in the background (one sample per
# interval, saved as a binary .sa file), runs the load for the given duration,
# waits for sar to finish and pauses before the next scenario.
#
# Usage:
#   ./recolectar.sh [options] [scenario ...]
#
# Scenarios: reposo cpu memoria disco red (default: all, in that order)
#
# Options:
#   -d SECONDS   duration of each scenario      (default: 1800)
#   -i SECONDS   sampling interval of sar       (default: 1)
#   -p SECONDS   pause between scenarios        (default: 60)
#   -m PERCENT   memory used by "memoria"       (default: 75)
#   -o DIR       output directory               (default: <repo>/data/raw)
#   -h           show this help
#
# Example (quick test, 60 s per scenario):
#   ./recolectar.sh -d 60 -p 10

set -euo pipefail

# Use the C locale so numbers keep a dot as decimal separator and the
# command output (lscpu, free, ...) is in English, whatever the system language.
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DURATION=1800
INTERVAL=1
PAUSE=60
MEM_PERCENT=75
OUTDIR="$REPO_ROOT/data/raw"
ALL_SCENARIOS=(reposo cpu memoria disco red)

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while getopts ":d:i:p:m:o:h" opt; do
    case "$opt" in
        d) DURATION="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        p) PAUSE="$OPTARG" ;;
        m) MEM_PERCENT="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) echo "Error: option -$OPTARG requires a value." >&2; exit 1 ;;
        *) echo "Error: unknown option -$OPTARG." >&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

SCENARIOS=("$@")
if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
    SCENARIOS=("${ALL_SCENARIOS[@]}")
fi

for value in "$DURATION" "$INTERVAL" "$MEM_PERCENT"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: -d, -i and -m must be positive integers." >&2
        exit 1
    fi
done
if ! [[ "$PAUSE" =~ ^[0-9]+$ ]]; then
    echo "Error: -p must be a non-negative integer." >&2
    exit 1
fi
if (( MEM_PERCENT > 95 )); then
    echo "Error: -m must be at most 95 to keep the server responsive." >&2
    exit 1
fi
if (( DURATION < INTERVAL )); then
    echo "Error: the duration (-d) must be at least the interval (-i)." >&2
    exit 1
fi

needs_stress=false
for scenario in "${SCENARIOS[@]}"; do
    case "$scenario" in
        reposo) ;;
        cpu|memoria|disco|red) needs_stress=true ;;
        *) echo "Error: unknown scenario '$scenario'. Valid: ${ALL_SCENARIOS[*]}" >&2; exit 1 ;;
    esac
done

require() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' is not installed ($2)." >&2
        exit 1
    fi
}
require sar "package sysstat"
if $needs_stress; then
    require stress-ng "package stress-ng"
fi

mkdir -p "$OUTDIR"
RUN_LOG="$OUTDIR/run_log.csv"
SPECS="$OUTDIR/server_specs.txt"

SAR_PID=""
LOAD_PID=""
cleanup() {
    [[ -n "$LOAD_PID" ]] && kill "$LOAD_PID" 2>/dev/null || true
    [[ -n "$SAR_PID" ]] && kill "$SAR_PID" 2>/dev/null || true
}
trap 'echo; echo "Interrupted: stopping sar and the load."; cleanup; exit 130' INT TERM

save_specs() {
    {
        echo "# Server specifications"
        echo "date: $(date --iso-8601=seconds)"
        echo
        echo "## Kernel";      uname -a
        echo
        echo "## OS";          head -n 4 /etc/os-release 2>/dev/null || true
        echo
        echo "## CPU";         lscpu 2>/dev/null | head -n 20 || true
        echo
        echo "## Memory";      free -h
        echo
        echo "## Disks";       lsblk -d -o NAME,SIZE,ROTA,MODEL 2>/dev/null || true
        echo
        echo "## Tools"
        sar -V 2>&1 | head -n 1
        if command -v stress-ng >/dev/null 2>&1; then stress-ng --version 2>&1 | head -n 1; fi
        echo
        echo "## Parameters"
        echo "duration_s=$DURATION interval_s=$INTERVAL pause_s=$PAUSE mem_percent=$MEM_PERCENT"
        echo "scenarios=${SCENARIOS[*]}"
    } > "$SPECS"
}

run_load() {
    local scenario="$1"
    case "$scenario" in
        reposo)  sleep "$DURATION" ;;
        cpu)     stress-ng --cpu 0 --timeout "${DURATION}s" --metrics-brief ;;
        memoria) stress-ng --vm 1 --vm-bytes "${MEM_PERCENT}%" --vm-keep --timeout "${DURATION}s" --metrics-brief ;;
        disco)   stress-ng --hdd 2 --hdd-bytes 1g --temp-path "$OUTDIR" --timeout "${DURATION}s" --metrics-brief ;;
        red)     stress-ng --sock 2 --timeout "${DURATION}s" --metrics-brief ;;
    esac
}

save_specs
if [[ ! -f "$RUN_LOG" ]]; then
    echo "scenario,start_epoch,end_epoch,duration_s,interval_s,samples" > "$RUN_LOG"
fi

samples=$(( DURATION / INTERVAL ))
total=${#SCENARIOS[@]}
index=0

echo "Output directory: $OUTDIR"
echo "Scenarios: ${SCENARIOS[*]} | ${DURATION}s each | interval ${INTERVAL}s | pause ${PAUSE}s"

for scenario in "${SCENARIOS[@]}"; do
    index=$(( index + 1 ))
    scenario_dir="$OUTDIR/$scenario"
    sa_file="$scenario_dir/$scenario.sa"
    mkdir -p "$scenario_dir"
    rm -f "$sa_file"

    echo "[$index/$total] $scenario: recording $samples samples..."
    start_epoch=$(date +%s)

    sar -o "$sa_file" "$INTERVAL" "$samples" >/dev/null 2>&1 &
    SAR_PID=$!

    run_load "$scenario" > "$scenario_dir/load.log" 2>&1 &
    LOAD_PID=$!
    wait "$LOAD_PID" || echo "  Warning: the load for '$scenario' ended with an error (see load.log)."
    LOAD_PID=""

    wait "$SAR_PID" || true
    SAR_PID=""
    end_epoch=$(date +%s)

    echo "$scenario,$start_epoch,$end_epoch,$DURATION,$INTERVAL,$samples" >> "$RUN_LOG"
    echo "  done: $sa_file"

    if (( index < total && PAUSE > 0 )); then
        echo "  pausing ${PAUSE}s..."
        sleep "$PAUSE"
    fi
done

echo "Collection finished. Next step: ./recoleccion/exportar_csv.sh"
