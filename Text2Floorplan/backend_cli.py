"""Stable JSON backend used by the Revit C# add-in."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path


os.environ["TEXT2REVIT_BACKEND"] = "1"

FLOW_ROOT = Path(__file__).resolve().parent
if str(FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOW_ROOT))

import pipeline_generate as pipeline  # noqa: E402
from pipeline_generate import generate_plan_data  # noqa: E402
from revit_geometry import build_revit_geometry
from plan_quality import load_physical_scale, validate_room_areas
from plot_layout import to_canvas
from plot_outline import measure_plot
from shapely.geometry import Polygon

MAX_GENERATION_ATTEMPTS = 10
# With an uploaded plot: stop after this many valid layouts, or as soon as there
# is one once PLOT_PATIENCE samples were drawn.
PLOT_ATTEMPTS, PLOT_CANDIDATES, PLOT_PATIENCE = 36, 3, 14
PLOT_INFERENCE_STEPS = 100


def plot_document(plot, outline, pixel_to_meter):
    """The plot as it is stored with a plan, in editor canvas coordinates."""
    _, point = to_canvas([], outline)
    return {
        "polygon": [point(p) for p in plot["plot"]],
        "edge_lengths_m": plot["lengths"],
        "plot_area_m2": plot["plot_area_m2"],
        "footprint_m2": plot["footprint_m2"],
        "coverage": plot["coverage"],
    }


def default_pixel_to_meter(size="medium"):
    return load_physical_scale(size)["pixel_to_meter"]


def flatten_openings(door_window_rules):
    pieces = []
    front_door = door_window_rules.get("front_door")
    if front_door is not None:
        pieces.append(front_door)
    pieces.extend(door_window_rules.get("doors", []))
    pieces.extend(door_window_rules.get("windows", []))
    return [
        piece["opening"]
        for piece in pieces
        if piece is not None and "opening" in piece
    ]


def validate_response(response):
    if not response["rooms"]:
        raise ValueError("backend generated no rooms")
    for room in response["rooms"]:
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            raise ValueError(
                f"room {room.get('room_id')} has invalid polygon"
            )
    for index, opening in enumerate(response["openings"], start=1):
        line = opening.get("reference_line") or []
        if len(line) != 2:
            raise ValueError(
                f"opening #{index} has invalid reference_line"
            )
        if float(opening.get("width", 0.0)) <= 0.0:
            raise ValueError(
                f"opening #{index} has non-positive width"
            )


def validate_door_connections(rooms, openings, edges):
    functional = {r["room_id"] for r in rooms if r["type"] not in {"door", "front_door"}}
    required = {frozenset((a, b)) for a, b in edges if a in functional and b in functional}
    actual = {frozenset((o["owner_room_id"], o["target_room_id"])) for o in openings
              if o["type"] != "window" and o.get("target_room_id") is not None}
    missing = required - actual
    if missing:
        # A planned connection without a door is acceptable when every room is
        # still reachable from the entrance through doors that were generated
        # (busy plans from prompt_planner.py often cannot fit every door).
        entrance = {room for o in openings
                    if o["type"] != "window" and o["host"] == "exterior_wall"
                    for room in (o["owner_room_id"], o.get("target_room_id")) if room is not None}
        doors = [tuple(pair) for pair in actual if len(pair) == 2]
        seen, queue = set(entrance), list(entrance)
        while queue:
            node = queue.pop()
            for a, b in doors:
                other = b if a == node else a if b == node else None
                if other is not None and other not in seen:
                    seen.add(other)
                    queue.append(other)
        if not functional <= seen:
            raise ValueError(f"connected rooms have no generated doorway: {sorted(tuple(sorted(pair)) for pair in missing)}")
    if not any(o["type"] != "window" and o["host"] == "exterior_wall" for o in openings):
        raise ValueError("the plan has no entrance door")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--status")
    args = parser.parse_args()

    request = json.loads(
        Path(args.request).read_text(encoding="utf-8")
    )
    prompt = request["prompt"]
    # Every attempt uses start_seed + attempt, so one number repeats the whole run.
    start_seed = request.get("seed")
    if start_seed is None:
        start_seed = secrets.randbits(32)
    start_seed = int(start_seed) & 0xFFFFFFFF
    plot = request.get("plot")
    if plot is not None:
        plot = measure_plot(plot["points"], plot["lengths"])
    def message(english, chinese):
        return chinese if request.get("language", "en") == "zh" else english
    def status(message):
        if args.status:
            path = Path(args.status)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"message": message}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        print(message, flush=True)

    device = "NVIDIA GPU" if pipeline.train.DEVICE.type == "cuda" else "CPU"
    status(message(f"Loading models and generating the floor plan ({device})…",f"正在加载模型并生成户型（{device}）…"))
    last_error = None
    # Read the prompt once (the LLM answer is cached for the retries). Busy
    # programs get more attempts, split into three phases: the exact request,
    # then open plan, then counts capped at what the model draws reliably.
    status(message("Reading your description…", "正在理解描述…"))
    room_total = sum(g.count for g in pipeline.prompt_planner.plan_prompt(prompt).rooms)
    attempts = MAX_GENERATION_ATTEMPTS if room_total <= 9 else 3 * MAX_GENERATION_ATTEMPTS
    outline, best, accepted = None, None, 0
    if plot is not None:
        # Filling a given outline rejects more samples, and the first valid one
        # is not always the nicest: keep the best of a few.
        outline = Polygon(plot["outline"])
        attempts = PLOT_ATTEMPTS
        pipeline.INFERENCE_STEPS = PLOT_INFERENCE_STEPS
    for attempt in range(attempts):
        if best is not None and (accepted >= PLOT_CANDIDATES or attempt >= PLOT_PATIENCE):
            break
        relax_level = attempt * 3 // attempts
        seed = (start_seed + attempt) & 0xFFFFFFFF
        try:
            processed, overlays, metadata = generate_plan_data(prompt, seed, relax_level, outline)
        except ValueError as error:
            last_error = error
            if attempt + 1 < attempts:
                status(message(f"Geometry check failed: {error}. Generating another layout (attempt {attempt + 2}/{attempts})…", f"几何检查未通过：{error}。正在重新生成（尝试 {attempt + 2}/{attempts}）…"))
            continue
        except RuntimeError as error:
            # CUDA availability does not guarantee enough VRAM or a supported GPU.
            error_text = str(error).lower()
            if pipeline.train.DEVICE.type != "cuda" or not any(term in error_text for term in ("cuda", "cudnn", "cublas", "out of memory", "nvidia")):
                raise
            pipeline.train.DEVICE = pipeline.torch.device("cpu")
            pipeline._cached_flow_model.cache_clear()
            status(message("GPU inference is unavailable. Switching to CPU…","显卡推理不可用，正在自动改用 CPU…"))
            processed, overlays, metadata = generate_plan_data(prompt, seed, relax_level, outline)
        openings = flatten_openings(overlays)
        try:
            # Repair diagnostics flag target shared-wall lengths, not missing doors.
            # The original placement code can fit a shorter door on such a wall.
            validate_door_connections(metadata["rooms"], openings, metadata["edge_lists"])
            pixel_to_meter = metadata["pixel_to_meter"]
            walls, railings = build_revit_geometry(metadata["rooms"], openings,
                pixel_to_meter, metadata["seed"])
            room_areas_m2 = validate_room_areas(metadata["rooms"], pixel_to_meter, walls=walls)
            if outline is None:
                break
            accepted += 1
            penalty = metadata["restore"]["plot_fit"]["penalty"]
            if best is None or penalty < best[0]:
                best = (penalty, processed, overlays, metadata, openings, pixel_to_meter,
                        walls, railings, room_areas_m2)
            if accepted < PLOT_CANDIDATES:
                status(message(f"Layout {accepted} fits the plot. Looking for a better one (attempt {attempt + 2}/{attempts})…",
                               f"已有 {accepted} 个方案符合地块，正在寻找更好的方案（尝试 {attempt + 2}/{attempts}）…"))
        except ValueError as error:
            last_error = error
            if attempt + 1 < attempts:
                status(message(f"Layout check failed: {error}. Generating another layout (attempt {attempt + 2}/{attempts})…",f"户型检查未通过：{error}。正在重新生成（尝试 {attempt + 2}/{attempts}）…"))
    else:
        if best is None and outline is not None:
            raise ValueError(f"No layout fitted this plot after {attempts} attempts ({last_error}). "
                             "Ask for fewer rooms or check the plot size.")
        if best is None:
            raise ValueError(f"Unable to generate a valid Revit plan after {attempts} attempts: {last_error}")
    if best is not None:
        (_, processed, overlays, metadata, openings, pixel_to_meter,
         walls, railings, room_areas_m2) = best
    status(message("Inference finished. Preparing Revit data…","推理完成，正在准备 Revit 数据…"))

    response = {
        "schema_version": "1.0",
        "coordinate_system": "raw_canvas_px",
        "pixel_to_meter": pixel_to_meter,
        "seed": metadata["seed"],
        "start_seed": start_seed,
        "prompt": prompt,
        "clip_prompt": metadata["clip_prompt"],
        "room_areas_m2": room_areas_m2,
        "total_area_m2": sum(room_areas_m2.values()),
        "area_definition": "Estimated clear floor area including balconies; wall footprints excluded",
        "rooms": metadata["rooms"],
        "planner": metadata["planner"],
        "walls": walls,
        "railings": railings,
        "openings": openings,
        "debug": {
            "edge_lists": metadata["edge_lists"],
            "node_type_ids": metadata["node_type_ids"],
            "restore": metadata["restore"],
        },
    }
    if plot is not None:
        metadata["planner"]["notes"].extend(plot["notes"])
        response["plot"] = plot_document(plot, outline, pixel_to_meter)
    validate_response(response)
    output_path = Path(args.response)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pipeline.plt.subplots(figsize=(9, 9))
    import textwrap
    title = "\n".join([textwrap.fill(prompt, 90)] + [textwrap.fill("Note: " + n, 100) for n in metadata["planner"]["notes"]])
    pipeline.draw_records(axis, processed, title, overlays=overlays,
                          room_areas_m2=room_areas_m2, room_labels=metadata["room_labels"])
    if plot is not None:
        # Same way up as the uploaded drawing and the editor.
        ring = response["plot"]["polygon"]
        axis.add_patch(pipeline.PolygonPatch(ring, closed=True, fill=False, edgecolor="#b3472f",
                                             linewidth=1.6, linestyle="--"))
        xs, ys = zip(*ring)
        pad = 0.03 * max(max(xs) - min(xs), max(ys) - min(ys))
        axis.set_xlim(min(0, min(xs)) - pad, max(255, max(xs)) + pad)
        axis.set_ylim(max(255, max(ys)) + pad, min(0, min(ys)) - pad)
    figure.tight_layout()
    preview_path = output_path.with_suffix(".png")
    preview_temporary = preview_path.with_suffix(".tmp")
    figure.savefig(preview_temporary, format="png", dpi=180, facecolor="white")
    pipeline.plt.close(figure)
    preview_temporary.replace(preview_path)
    response["preview_image"] = str(preview_path)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    status(message("Generation complete","生成完成"))


if __name__ == "__main__":
    main()
