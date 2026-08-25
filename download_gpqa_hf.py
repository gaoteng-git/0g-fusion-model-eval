#!/usr/bin/env python3
"""Download the official GPQA dataset from HuggingFace and convert it to the
JSONL schema eval/gpqa_tasks.py expects (Question, Correct Answer,
Incorrect Answer 1/2/3).

GPQA (https://huggingface.co/datasets/Idavidrein/gpqa) is a GATED dataset.
Its own terms forbid publishing examples in plain text online (anti-leakage,
to reduce risk of the answer key ending up in a future model's training
data) -- so this script only writes the output file to your local disk; it
never uploads or prints question text. Do NOT commit the output file to any
git repo. eval/data/* is already gitignored except the fake sample file, as
a safety net, but the safest thing is to just keep the output outside this
repo entirely (the default output path below does that).

One-time setup before running this script:
  1. pip install datasets huggingface_hub
  2. Open https://huggingface.co/datasets/Idavidrein/gpqa in a browser while
     logged in, and click "Agree and access repository" -- the download
     below will 403 until you've done this once.
  3. huggingface-cli login
     (paste an access token from https://huggingface.co/settings/tokens)

Usage:
  python3 download_gpqa_hf.py                       # writes ../gpqa_diamond.jsonl
  python3 download_gpqa_hf.py --config gpqa_main --out /path/to/gpqa_main.jsonl
  python3 download_gpqa_hf.py --limit 5              # smoke-test without pulling everything
"""
import argparse
import json
import os
import sys

REQUIRED_COLUMNS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="gpqa_diamond",
                    help="HF dataset config name, e.g. gpqa_diamond (198q) or gpqa_main (448q). Default: gpqa_diamond")
    p.add_argument("--split", default="train", help="HF split name. GPQA only ships 'train'. Default: train")
    p.add_argument("--out", default=None,
                    help="Output JSONL path. Default: one directory above this repo, "
                         "so it can never be accidentally git-added from inside it.")
    p.add_argument("--limit", type=int, default=None, help="Only convert the first N rows (for a quick smoke test).")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency. Run: pip install datasets huggingface_hub")

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", f"{args.config}.jsonl")
    out_path = os.path.abspath(out_path)

    print(f"Loading Idavidrein/gpqa config={args.config!r} split={args.split!r} from HuggingFace...")
    try:
        ds = load_dataset("Idavidrein/gpqa", args.config)[args.split]
    except Exception as e:
        sys.exit(
            "Failed to load the dataset. Most likely cause: you haven't agreed to the "
            "dataset's terms yet, or aren't logged in.\n"
            "  1. Visit https://huggingface.co/datasets/Idavidrein/gpqa and click "
            "'Agree and access repository'.\n"
            "  2. Run: huggingface-cli login\n"
            f"Original error: {e}"
        )

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        sys.exit(f"Dataset is missing expected column(s) {missing}; got columns: {ds.column_names}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({col: row[col] for col in REQUIRED_COLUMNS}) + "\n")

    print(f"Wrote {len(ds)} rows to {out_path}")
    print("Reminder: this file contains real GPQA questions -- keep it out of version control.")
    print(f'Use it via: load_tasks(path="{out_path}")')


if __name__ == "__main__":
    main()
