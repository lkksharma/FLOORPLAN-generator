"""Download the three model files FloorPlan AI needs (about 3.8 GB in total).

1. Flow-matching checkpoint -> checkpoints/flow_matching_best.pth (193 MB)
2. CLIP ViT-L/14 text encoder -> models/clip/ (about 500 MB)
3. Qwen2.5-1.5B-Instruct for the prompt planner -> models/llm/qwen2.5-1.5b-instruct/ (3.1 GB)

Every source is pinned to one version, and the checkpoint and all Qwen files are
checked by SHA-256. All downloads and caches stay inside the project folder.
Usage: python scripts/download_models.py            download what is missing, then check
       python scripts/download_models.py --check    only check the files on disk
"""
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

CHECKPOINT = ROOT / "checkpoints" / "flow_matching_best.pth"
CHECKPOINT_URL = ("https://github.com/Simon777hh/Text2Floorplan/releases/download/"
                  "v1.1.0/flow_matching_best.pth")
CHECKPOINT_SHA256 = "11fe79209c433ec189b1d19b8626c20ff7319b71aac79915b54e570e89a3c6cb"
CLIP_DIR = ROOT / "models" / "clip"
QWEN_DIR = ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
QWEN_URL = f"https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/{QWEN_REVISION}/"
QWEN_SHA256 = {
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
}


def fetch(url: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.replace(target)


def verify(path: Path, expected: str) -> bool:
    name = path.relative_to(ROOT)
    if not path.is_file():
        print(f"  MISSING  {name}")
        return False
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    good = h.hexdigest() == expected
    print(f"  {'OK' if good else 'WRONG'}  {name}", flush=True)
    return good


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    print("1/3 flow-matching checkpoint")
    if not check_only:
        fetch(CHECKPOINT_URL, CHECKPOINT)
    good = verify(CHECKPOINT, CHECKPOINT_SHA256)

    print("2/3 CLIP text encoder")
    if not check_only and not (CLIP_DIR / "model.safetensors").is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "setup_models.py")], check=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        present = (CLIP_DIR / name).is_file()
        print(f"  {'OK' if present else 'MISSING'}  models/clip/{name}")
        good &= present

    print("3/3 Qwen2.5-1.5B-Instruct (prompt planner)")
    for name, expected in QWEN_SHA256.items():
        if not check_only:
            fetch(QWEN_URL + name, QWEN_DIR / name)
        good &= verify(QWEN_DIR / name, expected)
    print("all models ready" if good else "Some files are missing or wrong: delete them and run again.")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
