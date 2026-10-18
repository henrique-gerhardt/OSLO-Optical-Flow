#!/usr/bin/env bash
# One file with what is needed to analyse every queue job: outputs/_queue/report.txt
# (queue events, per job: status, setup lines, every validation line, errors, the final
# tables; host memory and OOM events), and report.tar.gz (report.txt, the full job logs
# and every metrics JSON written since the queue started). Called by box_queue.sh after
# each job; can be run by hand at any time.
set -uo pipefail

cd "$(dirname "$0")/.."
QUEUE="${QUEUE:-scripts/box_queue.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
STATE="$OUTPUT_DIR/_queue"
REPORT="$STATE/report.txt"

{
  echo "# box queue report, $(date '+%F %T'), git $(git rev-parse --short HEAD)"
  echo
  echo "## queue events"
  cat "$STATE/queue.log" 2>/dev/null
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
    name="${line%%|*}"
    name="${name%"${name##*[![:space:]]}"}"
    log="$STATE/$name.log"
    echo
    echo "## job $name"
    if [ -e "$STATE/$name.done" ]; then echo "status: done"
    elif [ -e "$STATE/$name.failed" ]; then echo "status: FAILED ($(cat "$STATE/$name.failed"))"
    elif [ -e "$log" ]; then echo "status: running or interrupted"
    else echo "status: pending"; continue
    fi
    [ -e "$STATE/$name.attempt1.log" ] && {
      echo "first attempt (out of memory), last lines:"
      tail -5 "$STATE/$name.attempt1.log"
    }
    echo "--- setup, validation and errors"
    grep -E "^(=== |skip |grid=|params=|patch training|pyramid|val@|metric node)|Error|error:|Traceback|Killed" "$log"
    echo "--- last 60 lines"
    tail -60 "$log"
  done < "$QUEUE"
  echo
  echo "## host"
  free -g
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null
  echo "OOM events (dmesg):"
  { dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null; } | grep -i -E "out of memory|killed process" | tail -20
} > "$REPORT"

find "$OUTPUT_DIR" -name "*.json" -newer "$STATE/.started" -not -path "*/_queue/*" > "$STATE/.json_list"
tar -czf "$STATE/report.tar.gz" "$REPORT" "$STATE"/*.log -T "$STATE/.json_list" 2>/dev/null
echo "$REPORT"
