"""Re-emit a previous run's command from its result JSON, with overrides.

Every runner writes its full ``args`` dict into the result JSON. Re-typing those
flags by hand to change one setting is how protocol drift happens: the SLOF rows
run at ``--iters 64 --infer-size 320x640`` (their published protocol) while the
PanoFlow row runs at native resolution, and silently mixing them would break
row-to-row comparability without any error.

This replays the recorded args through the *real* parser of the script that
produced them, applies the requested overrides, and prints the command. Values
equal to the parser default are dropped so the output stays readable, and
``store_true`` flags are emitted as bare flags.

    # what changed, without running anything
    python rerun_from_json.py /outputs/universality_slof_singlerotation_test \
        --set geodesic_metric=haversine \
        --set motion_bands_deg=0,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,inf \
        --output-suffix _hav

    # the whole table in one go
    python rerun_from_json.py /outputs/universality_* /outputs/a4_* --set ... --run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

# Discriminated by args keys that only one runner defines.
RUNNERS = [
    ("run_grid_floor_probe.py", "estimation_resolutions"),
    ("run_oslo_raft.py", "retina_resolution"),
    ("run_raft_shard_baseline.py", "predictor"),
]


def find_json(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no result JSON under {path}")
    return candidates[0]


def pick_runner(args: Dict) -> str:
    for script, key in RUNNERS:
        if key in args:
            return script
    raise ValueError(f"cannot tell which runner wrote these args: {sorted(args)[:8]}")


def build_command(script: str, saved: Dict, overrides: Dict[str, str],
                  output_suffix: str, explicit: bool = False) -> Optional[list[str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    module = __import__(script[:-3])
    # Run the script's own parser on an empty argv to learn its defaults.
    argv_backup = sys.argv
    sys.argv = [script]
    try:
        defaults = vars(module.parse_args())
    except SystemExit:  # a required arg exists; fall back to no default filtering
        defaults = {}
    finally:
        sys.argv = argv_backup

    # Validate overrides against the PARSER, not against the saved args: the whole
    # point is to set flags that did not exist when the run was made (a JSON from
    # before --geodesic-metric was added has no such key, and adding it is the use
    # case). Conversely, drop saved keys the parser no longer accepts, so replaying
    # a run from before a rename does not emit a dead flag.
    merged = dict(saved)
    if defaults:
        for key in overrides:
            if key not in defaults:
                raise KeyError(
                    f"{script} has no arg '{key}'. Valid: {', '.join(sorted(defaults))}")
        for key in list(merged):
            if key not in defaults:
                print(f"# note: dropping '{key}' — {script} no longer accepts it", flush=True)
                del merged[key]
    # --set values arrive as strings; a store_true flag must become a real bool or it
    # would be emitted as "--eval-only true", which argparse rejects.
    for key, raw in overrides.items():
        if isinstance(defaults.get(key), bool):
            merged[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            merged[key] = raw
    if output_suffix and "output_dir" in merged:
        merged["output_dir"] = merged["output_dir"].rstrip("/") + output_suffix

    cmd = ["python", script]
    for key, value in merged.items():
        if value is None or value == "":
            continue
        if (not explicit and key not in overrides
                and key in defaults and defaults[key] == value):
            continue                      # unchanged from default -> keep it terse
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)          # store_true
            continue
        cmd += [flag, str(value)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="Result JSONs or the dirs containing them.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a saved arg (repeatable). Keys use underscores, "
                        "as stored in the JSON (e.g. geodesic_metric=haversine).")
    p.add_argument("--output-suffix", default="",
                   help="Appended to the recorded --output-dir so the replay does not "
                        "overwrite the original result.")
    p.add_argument("--explicit", action="store_true",
                   help="Emit every recorded arg, not just the ones that differ from the "
                        "current defaults. Use for archival commands: the terse form is only "
                        "reproducible while the parser defaults stay put.")
    p.add_argument("--run", action="store_true",
                   help="Execute each command instead of only printing it.")
    args = p.parse_args()

    overrides: Dict[str, str] = {}
    for item in args.set:
        if "=" not in item:
            p.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value

    for raw in args.paths:
        path = Path(raw)
        try:
            payload = json.loads(find_json(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"# SKIP {raw}: {exc}", flush=True)
            continue
        saved = payload.get("args")
        if not saved:
            print(f"# SKIP {raw}: JSON has no 'args' block", flush=True)
            continue
        try:
            script = pick_runner(saved)
            cmd = build_command(script, saved, overrides, args.output_suffix, args.explicit)
        except (KeyError, ValueError) as exc:
            p.error(f"{raw}: {exc}")       # a typo'd --set must stop the whole batch
        print(" ".join(shlex.quote(c) for c in cmd), flush=True)
        if args.run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"# FAILED ({result.returncode}): {raw}", flush=True)
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
