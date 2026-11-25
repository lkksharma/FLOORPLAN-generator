"""Measure plot-constrained generation (no retries, rule reader only).

  spill  raw room area outside the plot / plot area, with and without the
         in-sampling guidance (L and T plots, 4 seeds each)
  fit    samples that the exact-fit step accepts (L, rectangle, T; 14 seeds each)
  valid  samples that also pass every door, wall and area check
Usage: .venv/bin/python report/eval_plot.py   -> report/results_plot.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("TEXT2REVIT_DEVICE", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["T2F_PLANNER"] = "rules"

from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import backend_cli as backend  # noqa: E402
import plot_layout  # noqa: E402
from plot_outline import measure_plot  # noqa: E402

pipeline = backend.pipeline
pipeline.INFERENCE_STEPS = backend.PLOT_INFERENCE_STEPS

PLOTS = {
    "L": ([(0, 0), (9, 0), (9, 6), (15, 6), (15, 14), (0, 14)], [9, 6, 6, 8, 15, 14]),
    "rectangle": ([(0, 0), (14, 0), (14, 15), (0, 15)], [14, 15, 14, 15]),
    "T": ([(0, 6), (5, 6), (5, 0), (11, 0), (11, 6), (16, 6), (16, 14), (0, 14)],
          [5, 6, 6, 6, 5, 8, 16, 8]),
}
SMALL = "A home with a living room, a kitchen, 3 bedrooms, 2 bathrooms and a balcony"
BUSY = ("A home with a living room, a kitchen, a master bedroom with an attached bathroom, "
        "a kids room, a guest room, a study, a pooja room, 1 bathroom and a balcony garden")


class Captured(Exception):
    pass


def raw_spill(outline, seed, guided):
    """Area the raw model output (before any clean-up) puts outside the plot."""
    def capture(topology, records, pairs):
        capture.records = records
        raise Captured

    original, push = pipeline._record_raw_metrics, plot_layout.Guide.push
    pipeline._record_raw_metrics = capture
    if not guided:
        plot_layout.Guide.push = lambda self, clean: clean
    try:
        pipeline.generate_plan_data(SMALL, seed, 0, outline)
    except Captured:
        pass
    finally:
        pipeline._record_raw_metrics, plot_layout.Guide.push = original, push
    x0, y0, x1, y1 = outline.bounds
    unit = Polygon([((x - x0) / (x1 - x0) * 2 - 1, (y - y0) / (y1 - y0) * 2 - 1)
                    for x, y in outline.exterior.coords])
    drawn = unary_union([r.polygon.buffer(0) for r in capture.records
                         if r.type_name not in plot_layout.DOORS])
    return drawn.difference(unit).area / unit.area


def attempt(outline, seed):
    """(fits the plot, passes all checks, reason)"""
    fitted, original = [], pipeline.plot_layout.fit_rooms

    def fit(records, shape):
        result = original(records, shape)
        fitted.append(True)
        return result

    pipeline.plot_layout.fit_rooms = fit
    try:
        _, overlays, md = pipeline.generate_plan_data(BUSY, seed, 0, outline)
        openings = backend.flatten_openings(overlays)
        backend.validate_door_connections(md["rooms"], openings, md["edge_lists"])
        walls, _ = backend.build_revit_geometry(md["rooms"], openings, md["pixel_to_meter"], md["seed"])
        backend.validate_room_areas(md["rooms"], md["pixel_to_meter"], walls=walls)
        return True, True, ""
    except ValueError as error:
        return bool(fitted), False, str(error)[:80]
    finally:
        pipeline.plot_layout.fit_rooms = original


def main():
    out = {"spill": {}, "fit": {}}
    for name in ("L", "T"):
        outline = Polygon(measure_plot(*PLOTS[name])["outline"])
        values = {key: [raw_spill(outline, seed, key == "guided") for seed in range(4)]
                  for key in ("unguided", "guided")}
        out["spill"][name] = {k: round(statistics.mean(v), 4) for k, v in values.items()}
        print(f"{name}: spill {out['spill'][name]}", flush=True)
    for name, plot in PLOTS.items():
        outline = Polygon(measure_plot(*plot)["outline"])
        rows = [attempt(outline, seed) for seed in range(14)]
        out["fit"][name] = {"samples": len(rows), "fits": sum(r[0] for r in rows),
                            "valid": sum(r[1] for r in rows), "rejections": [r[2] for r in rows if r[2]]}
        print(f"{name}: fits {out['fit'][name]['fits']}/14, valid {out['fit'][name]['valid']}/14", flush=True)
    (ROOT / "report" / "results_plot.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
