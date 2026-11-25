"""Compare freshly generated results with the committed ones (timings ignored).

Run after the eval_*.py scripts. Exit code 1 if any number differs.
Usage: .venv/bin/python report/check_results.py
"""
import json
from pathlib import Path
import subprocess
import sys

REPORT = Path(__file__).resolve().parent
TIMINGS = {"seconds", "seconds_per_prompt", "llm_seconds_per_prompt"}


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in TIMINGS}
    return [strip(v) for v in value] if isinstance(value, list) else value


def differences(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            yield from differences(old.get(key), new.get(key), f"{path}/{key}")
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, pair in enumerate(zip(old, new)):
            yield from differences(*pair, f"{path}/{index}")
    elif isinstance(old, float) and isinstance(new, (int, float)) and abs(old - new) < 1e-6:
        return
    elif old != new:
        yield f"{path}: committed {str(old)[:60]!r}, now {str(new)[:60]!r}"


bad = 0
for name in ("planner", "generation", "compat", "plot"):
    file = REPORT / f"results_{name}.json"
    committed = subprocess.run(["git", "show", f"HEAD:./{file.name}"], cwd=REPORT,
                               capture_output=True, text=True, check=True).stdout
    found = list(differences(strip(json.loads(committed)), strip(json.loads(file.read_text()))))
    bad += len(found)
    print(f"{file.name}: " + ("identical" if not found else f"{len(found)} differences"))
    for line in found[:8]:
        print("   ", line)
sys.exit(bool(bad))
