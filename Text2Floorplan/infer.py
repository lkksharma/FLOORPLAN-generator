"""Generate a floor-plan PNG and editable JSON from the command line.

Example: python infer.py --prompt "A 3-bedroom apartment with 2 bathrooms and 1 balcony."
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Free-form apartment description, e.g. \"a 4 bedroom flat with a dining room and a balcony garden\"")
    parser.add_argument("--checkpoint", type=Path, help="Full or inference-only Flow Matching PTH")
    parser.add_argument("--clip-dir", type=Path, help="Local CLIP text encoder directory")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto",
                        help="Automatically use a compatible GPU, or force CPU")
    parser.add_argument("--seed", type=int, help="Start seed: the same seed, prompt and plot repeat the run "
                        "(saved as start_seed in the JSON)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "inference")
    parser.add_argument("--planner", choices=("auto", "rules"), default="auto",
                        help="auto: local LLM reads the prompt (falls back to rules); rules: rule reader only")
    parser.add_argument("--plot", type=Path, help="Image of the plot; the plan will fill its outline")
    parser.add_argument("--lengths", help="Edge lengths in metres, comma separated, in the order "
                        "printed by: python plot_outline.py IMAGE")
    parser.add_argument("--language", choices=("en", "zh"), default="en",
                        help="Progress-message language; prompts remain English")
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.clip_dir is not None and not args.clip_dir.is_dir():
        parser.error(f"CLIP directory does not exist: {args.clip_dir}")

    if (args.plot is None) != (args.lengths is None):
        parser.error("--plot and --lengths go together")
    plot = None
    if args.plot is not None:
        from plot_outline import PlotError, detect_outline, measure_plot
        try:
            plot = {"points": detect_outline(args.plot.read_bytes())["points"],
                    "lengths": [float(v) for v in args.lengths.split(",")]}
            measure_plot(**plot)
        except (OSError, PlotError, ValueError) as error:
            parser.error(str(error))

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    response = output / f"floorplan_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                       TEXT2REVIT_BACKEND="1", MPLBACKEND="Agg")
    python_home = Path(sys.executable).resolve().parent
    library_paths = [python_home, python_home / "Library" / "bin", python_home / "DLLs"]
    environment["PATH"] = os.pathsep.join(str(p) for p in library_paths if p.is_dir()) + os.pathsep + environment.get("PATH", "")
    if python_home == ROOT / "runtime":
        # Match the portable launcher's isolated runtime when invoked directly.
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
    environment["T2F_PLANNER"] = args.planner
    if args.device == "cpu":
        environment["TEXT2REVIT_DEVICE"] = "cpu"
    else:
        environment.pop("TEXT2REVIT_DEVICE", None)
    for value, variable in ((args.checkpoint, "TEXT2REVIT_CHECKPOINT"),
                            (args.clip_dir, "TEXT2REVIT_CLIP_DIR")):
        if value is not None:
            environment[variable] = str(value.resolve())

    # Reuse the production backend's geometry checks, retries and GPU fallback.
    # The request is temporary; the result JSON retains the prompt and final seed.
    with tempfile.TemporaryDirectory(prefix=".request-", dir=output) as directory:
        request = Path(directory) / "request.json"
        request.write_text(json.dumps({"prompt": args.prompt, "seed": args.seed,
                                       "language": args.language,
                                       **({"plot": plot} if plot else {})}, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", str(ROOT / "backend_cli.py"),
                                 "--request", str(request), "--response", str(response)],
                                cwd=ROOT, env=environment, check=False)
    if result.returncode:
        return result.returncode
    print(f"JSON: {response}")
    print(f"PNG:  {response.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
