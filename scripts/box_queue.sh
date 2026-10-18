#!/usr/bin/env bash
# Runs the jobs of a queue file one after another on the box, on the HOST (it calls
# docker compose). Run it inside tmux:
#
#   tmux new -s queue
#   SHARDS_HOST=/home/henriquegerhardt/Developer/sfprep/shards bash scripts/box_queue.sh
#
# Queue file (QUEUE, default scripts/box_queue.txt): one job per line,
#   name | script | VAR=value; VAR=value; ...
# '#' starts a comment line; a literal \n inside a value becomes a newline. Every VAR is
# passed to the container with -e. The file is re-read before each job, so jobs can be
# added or reordered while the queue runs.
#
# State lives in $OUTPUT_DIR/_queue: <name>.log (full output), <name>.done or
# <name>.failed, queue.log (one line per event), report.txt + report.tar.gz (rebuilt
# after every job by scripts/box_queue_report.sh). A job with a .done or .failed marker
# is skipped; delete the marker to run it again.
#
# Failsafes:
# - A failed job is marked and the queue moves on.
# - The container gets a host-RAM cap (OSLO_MEM_LIMIT; default total RAM minus
#   MEM_RESERVE_GB=6), so running out of memory kills a process in the container, not
#   tmux, sshd or this script.
# - A job that fails out of memory is retried once with NUM_WORKERS=2 (the first log is
#   kept as <name>.attempt1.log).
# - A job whose log has not changed for STALL_MIN=90 minutes is killed and marked failed.
# - Two jobs in a row failing within 10 minutes each = a setup error (compose, image,
#   paths), not a job error: the queue stops instead of failing every job.
#
# Before each job: waits for any running oslo-raft container to exit, then (PULL=1,
# the default) runs git pull --ff-only, so pushed queue edits and code are picked up.
# WAIT=1 keeps polling for new jobs every 10 minutes instead of exiting at the end.
set -uo pipefail

cd "$(dirname "$0")/.."
QUEUE="${QUEUE:-scripts/box_queue.txt}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
STATE="$OUTPUT_DIR/_queue"
STALL_MIN="${STALL_MIN:-90}"
COMPOSE="docker compose -f docker-compose.oslo_raft.yml"
mkdir -p "$STATE"
[ -e "$STATE/.started" ] || touch "$STATE/.started"
# /proc/meminfo, not free: free's labels follow the locale ("Mem.:" in pt_BR).
mem_gb() { awk -v k="$1:" '$1 == k {print int($2 / 1048576)}' "${MEMINFO:-/proc/meminfo}"; }
if [ -z "${OSLO_MEM_LIMIT:-}" ]; then
  total_gb="$(mem_gb MemTotal)"
  if ! [ "${total_gb:-0}" -gt $((${MEM_RESERVE_GB:-6} + 8)) ] 2>/dev/null; then
    echo "could not read total RAM (got '${total_gb:-}'); set OSLO_MEM_LIMIT, e.g. OSLO_MEM_LIMIT=56g" >&2
    exit 1
  fi
  OSLO_MEM_LIMIT="$((total_gb - ${MEM_RESERVE_GB:-6}))g"
fi
export OSLO_MEM_LIMIT

note() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE/queue.log"; }
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }
running() { docker ps -q --filter name=oslo-raft-run; }

next_job() {
  while IFS= read -r line; do
    line="$(trim "$line")"
    [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
    local name
    name="$(trim "${line%%|*}")"
    [ -e "$STATE/$name.done" ] || [ -e "$STATE/$name.failed" ] || { echo "$line"; return; }
  done < "$QUEUE"
}

# run_job <log> <script> <docker -e args...>; returns the job's exit status (124 = stalled).
run_job() {
  local log="$1" script="$2"
  shift 2
  OSLO_GIT_SHA="$(git rev-parse HEAD)" $COMPOSE run --build --rm "$@" \
    oslo-raft bash "$script" < /dev/null > "$log" 2>&1 &
  local pid=$! stalled=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ $(( $(date +%s) - $(mtime "$log") )) -gt $((STALL_MIN * 60)) ]; then
      note "no output for $STALL_MIN min, killing the container"
      running | xargs -r docker kill > /dev/null
      stalled=1
      break
    fi
  done
  wait "$pid"
  local status=$?
  [ $stalled -eq 1 ] && return 124
  return $status
}

note "queue start: memory cap $OSLO_MEM_LIMIT, stall limit $STALL_MIN min"
quick_fails=0
while true; do
  while [ -n "$(running)" ]; do
    note "an oslo-raft container is running, waiting"
    sleep 600
  done
  if [ "${PULL:-1}" = "1" ]; then
    git pull --ff-only >> "$STATE/queue.log" 2>&1 || note "git pull failed, using the current checkout"
  fi
  job="$(next_job)"
  if [ -z "$job" ]; then
    [ "${WAIT:-0}" = "1" ] || { note "queue empty, exiting"; break; }
    sleep 600
    continue
  fi

  IFS='|' read -r name script envs <<< "$job"
  name="$(trim "$name")"
  script="$(trim "$script")"
  args=()
  IFS=';' read -ra pairs <<< "${envs:-}"
  for pair in "${pairs[@]}"; do
    pair="$(trim "$pair")"
    [ -n "$pair" ] && args+=(-e "${pair//\\n/$'\n'}")
  done
  log="$STATE/$name.log"

  note "start $name ($script) at git $(git rev-parse --short HEAD); $(mem_gb MemAvailable) GB RAM available"
  started=$(date +%s)
  run_job "$log" "$script" "${args[@]}"
  status=$?
  if [ $status -ne 0 ] && [ $status -ne 124 ] \
      && { [ $status -eq 137 ] || grep -q -E "Killed|out of memory|OutOfMemory" "$log"; }; then
    mv "$log" "$STATE/$name.attempt1.log"
    note "$name ran out of memory (exit $status), retrying once with NUM_WORKERS=2"
    run_job "$log" "$script" "${args[@]}" -e NUM_WORKERS=2
    status=$?
  fi
  if [ $status -eq 0 ]; then
    touch "$STATE/$name.done"
    note "done $name"
    quick_fails=0
  else
    echo "exit $status" > "$STATE/$name.failed"
    note "FAILED $name (exit $status): $(tail -1 "$log")"
    if [ $(( $(date +%s) - started )) -lt 600 ]; then
      quick_fails=$((quick_fails + 1))
    else
      quick_fails=0
    fi
  fi
  bash scripts/box_queue_report.sh > /dev/null 2>&1 || note "report failed"
  if [ $quick_fails -ge 2 ]; then
    note "two jobs in a row failed within 10 minutes: setup error, stopping the queue"
    note "fix it, delete the .failed markers in $STATE and start the queue again"
    exit 1
  fi
done
bash scripts/box_queue_report.sh > /dev/null 2>&1
note "report: $STATE/report.txt and $STATE/report.tar.gz"
