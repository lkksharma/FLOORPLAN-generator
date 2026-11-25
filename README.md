# FloorPlan AI

Text and an optional plot drawing in, an editable floor plan out (walls, doors, windows, room areas, 2D and 3D).
It runs offline on a laptop.

Generative AI (UCS748) project.

| Name | Roll no. | Group |
|---|---|---|
| Lakksh Sharma | 102303836 | 4C24 |
| Ishita Gupta | 102303377 | 4C24 |
| Divyam Jund | 102303445 | 4C26 |
| Akhil Gupta | 102315121 | 4O15 |

![Generated floor plan](Text2Floorplan/report/ui/ui_split.png)

## How it works

```text
plot image ──► OpenCV outline ──► edge lengths (m) ──► building outline ───────────┐
                                                                                   │
prompt ──► Qwen2.5-1.5B + rule reader ──► room list ──► room graph ──┐             │
                                              └──► short sentence ──► CLIP ──┐     │
                                                                     ▼       ▼     ▼
                                         random corners ──► flow-matching model + plot guidance
                                                                                   ▼
                              fit to outline ──► doors, windows ──► checks ──► plan.json + plan.png
```

We trained nothing. The flow model is the released Text2Floorplan checkpoint, and CLIP and Qwen are public weights.

| Ours | File |
|---|---|
| Prompt planner (LLM + rules, room graph, fallbacks) | `prompt_planner.py` |
| Plot detection and measuring | `plot_outline.py` |
| Plot guidance and exact fit | `plot_layout.py` |
| Plot page | `web_editor/static/plot.*` |
| Evaluation | `report/eval_*.py` |

Changed by us: `pipeline_generate.py`, `backend_cli.py`, `infer.py`, `topology_rules.py`, `flow_solver.py`,
`inference_runtime.py` and the editor. The rest is Text2Floorplan.

## 1. Requirements

- Python **3.12** and Git
- 16 GB RAM recommended, about 5 GB of disk
- Internet for setup only
- No GPU needed. Tested on an Apple M4 (macOS). The commands are for macOS and Linux;
  on Windows use `.venv\Scripts\python` instead of `.venv/bin/python`.

## 2. Setup

```bash
git clone https://github.com/lkksharma/FLOORPLAN-generator.git
cd FLOORPLAN-generator/Text2Floorplan
python3.12 -m venv .venv
export PIP_CACHE_DIR="$PWD/.cache/pip" HF_HOME="$PWD/.cache/huggingface" MPLCONFIGDIR="$PWD/.cache/matplotlib"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_models.py
```

**Run every command below from `Text2Floorplan/`.** Everything is stored in this folder, so deleting it removes everything.
If a package changes later, install the exact tested versions instead: `pip install -r requirements-lock.txt`
(frozen on macOS arm64).

| Model | Where it goes | Size | Source, pinned |
|---|---|---|---|
| Flow matching | `checkpoints/flow_matching_best.pth` | 193 MB | Text2Floorplan release v1.1.0, SHA-256 checked |
| CLIP ViT-L/14 text encoder | `models/clip/` | 500 MB | `openai/clip-vit-large-patch14` @ `32bd642` |
| Qwen2.5-1.5B-Instruct | `models/llm/qwen2.5-1.5b-instruct/` | 3.1 GB | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79`, SHA-256 checked |

Check the model files at any time:

```bash
.venv/bin/python scripts/download_models.py --check
```

## 3. Start Qwen

Qwen is not a separate server. It is loaded inside Python when a prompt is read and released afterwards, so there is
nothing to keep running. To run it on its own:

```bash
# Qwen + rules: the final room list (Qwen's raw reply is in "llm_reply")
.venv/bin/python prompt_planner.py "3 bedrooms, a pooja room, a study and 2 bathrooms"

# Only Qwen's raw reply
.venv/bin/python -c "import prompt_planner as p; print(p._ask_llm('3 bedrooms, a pooja room, a study and 2 bathrooms'))"
```

| Setting | Effect |
|---|---|
| `T2F_LLM_DEVICE=cpu` (or `mps`, `cuda`) | Where Qwen runs. Default: cuda, then mps, then cpu |
| `T2F_LLM_DIR=/path` | Use Qwen from another folder |
| `T2F_PLANNER=rules` | Skip Qwen, use the rule reader only |

Decoding is greedy, so the same prompt on the same device gives the same reply. If Qwen is missing or its reply is
broken, the rule reader takes over and the plan's notes say so.

## 4. Run the app

```bash
.venv/bin/python -m web_editor.server --port 8765
```

Open **http://127.0.0.1:8765/plot** (use `127.0.0.1`, not `localhost`).

1. Upload a plot image, for example `sample/plot_L.png`. Correct the corners if needed.
2. Enter every edge in metres. For `plot_L.png`: A 9, B 6, C 6, D 8, E 15, F 14.
3. Continue, describe the home and press **Generate layout**. It takes one to four minutes on a CPU.
4. Edit in 2D or 3D, then save the JSON.

Without a plot, open http://127.0.0.1:8765/en. Each job is saved in `outputs/web_editor/<job id>/`
(`request.json`, `plan.json`, `plan.png`, `inference.log`). Stop the server with Ctrl+C.

A prompt that shows most features:

```text
A spacious home with a living room, a kitchen, a master bedroom with an attached bathroom, a kids room, a guest room, a home office, a pooja room, 2 bathrooms and a balcony garden
```

## 5. Run from the command line

```bash
# Text only
.venv/bin/python infer.py --seed 42 --prompt "2 bedroom flat, master bedroom with en-suite, a study and a balcony garden"

# Without Qwen
.venv/bin/python infer.py --seed 42 --planner rules --prompt "3bhk with 2 bathrooms and a balcony"

# List the edges found in a plot drawing, in order
.venv/bin/python plot_outline.py sample/plot_L.png

# Fill that plot
.venv/bin/python infer.py --seed 42 --plot sample/plot_L.png --lengths 9,6,6,8,15,14 --prompt "3bhk with 2 bathrooms and a balcony"
```

Other sample plots: `plot_rectangle.png` (`14,15,14,15`) and `plot_trapezoid.png` (`12,14.3,15,14`).
Results go to `outputs/inference/` as PNG and JSON. `--device cpu` forces the CPU.

**Repeating a run:** attempt *n* uses seed `start_seed + n`. The start seed is saved as `start_seed` in the JSON, so
`--seed <start_seed>` with the same prompt and plot gives the same plan on the same machine. Different hardware
or library versions can change floating-point results.

## 6. Tests

```bash
.venv/bin/python -m unittest discover -s tests
node --test tests/*.mjs        # optional, needs Node.js
```

## 7. Reproduce the evaluation

Times are for an Apple M4. Each script writes `report/results_*.json`.

| Command | Measures | Time |
|---|---|---|
| `.venv/bin/python report/eval_planner.py` | Count accuracy, rooms covered, exact program (30 prompts) | 1 min |
| `T2F_PLANNER=rules .venv/bin/python report/eval_compat.py 12` | GED, overlap, % overlap, Room ID (5–8 rooms) | 10 min |
| `.venv/bin/python report/eval_generation.py 5` | Validity, connection recall, counts, GED (8 prompts × 5 seeds) | 10 min |
| `.venv/bin/python report/eval_plot.py` | Spill with and without guidance, fit and validity on plots | 15 min |

Then compare with the committed numbers (timings are ignored) and redraw the charts:

```bash
.venv/bin/python report/check_results.py
.venv/bin/python report/make_charts.py
```

`eval_planner.py` and `eval_generation.py` reuse Qwen's saved replies in `report/llm_cache.json`. To ask Qwen again (about 10 minutes; all 30 replies came back identical):

```bash
mv report/llm_cache.json report/llm_cache.backup.json
.venv/bin/python report/eval_planner.py
```

### Expected results

**Reading the prompt**

| Reader | Count accuracy | Rooms covered | Exact program |
|---|---|---|---|
| Original Text2Floorplan parser | 45.7% | 46.1% | 13.3% |
| Qwen alone | 79.8% | 85.2% | 56.7% |
| Rules alone | 96.1% | 96.9% | 83.3% |
| **LLM + rules** | **100%** | **100%** | **100%** |

The prompts are ours and two bugs were fixed after seeing failures on them, so 100% is optimistic.

**Generated plans:** 21 of 40 valid on the first sample, 92% connection recall, mean GED 0.29 on accepted plans.
GED for 5, 6, 7 and 8 rooms: 0, 0.33, 0.33 and 1.0 (12 samples each, our own prompts, one-sided GED).

**Plots**

| | L | Rectangle | T |
|---|---|---|---|
| Raw spill outside the plot, no guidance (4 seeds) | 14.7% | – | 28.5% |
| Raw spill with guidance (4 seeds) | 1.4% | – | 5.2% |
| Samples that fit the plot (14 seeds, 10 rooms) | 9 | 14 | 9 |
| Samples that pass every check, no retry (14 seeds) | 3 | 6 | 2 |

The app retries and keeps the best valid plan. The fit step was tuned on these prompts, so these numbers are optimistic.
A final plan on a right-angled plot covers it exactly, because the fit step enforces it.

FID and IoU are not measured: they need the RPLAN and Tell2Design test sets.

## 8. Rebuild the report

```bash
cd report/latex && tectonic main.tex      # or: pdflatex main.tex, twice
```

Screenshots are optional and need Google Chrome on macOS, with the app running on port 8765:

```bash
.venv/bin/python -m pip install websocket-client
.venv/bin/python report/capture_ui.py sample/plan.json "A large 3-bedroom apartment with 2 bathrooms and 2 balconies."
```

## 9. Repository map

| Path (in `Text2Floorplan/`) | Contents |
|---|---|
| `infer.py`, `backend_cli.py`, `pipeline_generate.py` | Command line, retry loop and checks, pipeline |
| `prompt_planner.py`, `topology_rules.py` | Prompt to rooms to room graph |
| `plot_outline.py`, `plot_layout.py` | Plot image to outline, guidance and fit |
| `model.py`, `flow_solver.py`, `inference_runtime.py`, `data_generation.py`, `data_utils.py`, `edge_utils.py`, `runtime_utils.py` | Flow model, sampler and its inputs |
| `plan_postprocess.py`, `restore_canvas_aspect.py`, `geometry_cleanup.py`, `door_window_rules.py`, `plan_quality.py`, `revit_geometry.py` | Clean-up, scale, doors, windows, walls and checks |
| `web_editor/` | Local server, plot page and editor |
| `data/` | Small statistics needed at inference |
| `scripts/` | Model downloads |
| `sample/` | Example plan and plot drawings |
| `tests/`, `report/` | Tests, evaluation, results and report |

## 10. Problems

| Problem | Fix |
|---|---|
| `Checkpoint does not exist` or CLIP not found | Run `scripts/download_models.py` |
| Address already in use | Another server is running: use `--port 8766` |
| "Use the localhost editor URL" | Open `127.0.0.1`, not `localhost` |
| Qwen is slow or runs out of memory | `T2F_PLANNER=rules`, or `T2F_LLM_DEVICE=cpu` |
| "No layout fitted this plot" | Fewer rooms, or check the lengths |
| Plot not detected | Use a clean line drawing, or place the corners by hand |

## Limits

- The model has no plot input. We guide its output and fit it, so walls are right-angled, and a slanted plot gets a
  right-angled building inside it.
- Plot detection was tested on clean drawings, not photos.
- Setbacks and building rules are not applied.

## Credits

Generator: [Text2Floorplan](https://github.com/Simon777hh/Text2Floorplan) (MIT), see `Text2Floorplan/LICENSE` and
`THIRD_PARTY_NOTICES.md`. CLIP: OpenAI. Qwen2.5: Alibaba Cloud (Apache 2.0).

The code this project does not use (Raster-to-Graph, the earlier FloorPlan-GenAI app, training, Revit and Windows
packaging) was removed. It is still in the history: `git checkout 4c248f0`.
