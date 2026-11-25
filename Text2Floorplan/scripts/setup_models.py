"""Download the public CLIP text encoder for subsequent offline inference."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "clip")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    from transformers import CLIPTextModel, CLIPTokenizer
    args.output.mkdir(parents=True, exist_ok=True)
    CLIPTextModel.from_pretrained(NAME, revision=REVISION).save_pretrained(args.output, safe_serialization=True)
    CLIPTokenizer.from_pretrained(NAME, revision=REVISION).save_pretrained(args.output)
    print(f"CLIP text encoder saved to {args.output}")


if __name__ == "__main__":
    main()
