"""Extract a subset of a question file by row index into a new file, tagged
with each row's ORIGINAL absolute index (as `question_id`) so results computed
from different, possibly disjoint, sampled files stay keyed consistently and
can be merged later -- e.g. sample 0-4 to debug with, then once that's paying
off, sample the rest, run everything again, and `cat`/merge the two output
sets by question_id.

--indices takes a comma-separated list of integers and/or "start-end" ranges
(inclusive on both ends), e.g. "0-4,10,12-15". Duplicates are collapsed and
the output is always written in ascending question_id order, regardless of
the order --indices was typed in.

Run:
  python3 -m eval.sample --input eval/data/gpqa_sample.jsonl --indices 0-4 \\
      --out eval/samples/first5.jsonl
"""
import argparse
import json
import os
import sys

from .gpqa_tasks import load_questions


def parse_indices(raw):
    indices = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start, end = (int(x) for x in part.split("-", 1))
                if end < start:
                    # range(start, end+1) is silently EMPTY when end < start --
                    # a typo'd/inverted range (e.g. typing the second half of a
                    # split-the-remaining-questions batch backwards) would
                    # otherwise contribute nothing at all, with no error, and
                    # the resulting sample would just be quietly short.
                    raise ValueError(f"range {part!r} has end before start")
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
        except ValueError as e:
            sys.exit(f"--indices: can't parse {part!r}: {e}")
    if not indices:
        sys.exit("--indices must name at least one index")
    return sorted(indices)


def run(input_path, indices, out_path):
    questions = load_questions(input_path)
    by_id = dict(questions)
    missing = [i for i in indices if i not in by_id]
    if missing:
        raise ValueError(f"{input_path!r} has {len(questions)} question(s) (valid question_ids: "
                          f"{min(by_id)}-{max(by_id)}); requested index/indices not present: {missing}")
    # eval/samples/ and eval/results/ are both gitignored, so neither exists
    # on a fresh clone -- a bare open() would fail before writing anything.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in indices:
            f.write(json.dumps({**by_id[i], "question_id": i}) + "\n")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="A question file (CSV or JSONL, the official GPQA columns).")
    p.add_argument("--indices", required=True, help='Comma-separated indices/ranges, e.g. "0-4,10,12-15".')
    p.add_argument("--out", required=True, help="Always overwritten.")
    args = p.parse_args()
    print(run(args.input, parse_indices(args.indices), args.out))
