"""Replay-file plumbing shared by eval/panel.py, eval/fuse.py, eval/baseline.py.

All three write the same kind of file (one JSON row per question, keyed by
question_id), resume into it the same way, and share most of their CLI
shape -- so all of that lives here once. Everything task-specific -- what a
row contains, what gets called, what counts as a failure -- stays in each
script, behind run_replay()'s `process` callback.
"""
import argparse
import json
import os
import re
import sys

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def parse_models(raw):
    """Comma-separated --models -> deduped, order-preserving, non-empty list.
    A repeated name must cost one call, not two; an empty list must fail
    loud, not silently "succeed" at doing nothing."""
    models = list(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    if not models:
        sys.exit("--models must name at least one model")
    return models


def _positive_int(raw):
    n = int(raw)
    if n <= 0:
        # `if limit:` (every caller's actual limit check) treats 0 the same
        # as "no limit at all" -- silently running the FULL question set
        # instead of the empty/dry-run someone typing --limit 0 almost
        # certainly meant. A negative --limit would slice from the end
        # instead of narrowing anything. Reject both at the argparse layer.
        raise argparse.ArgumentTypeError("--limit must be a positive integer (omit it for no limit)")
    return n


def add_out_args(p):
    """--out/--limit/--no-resume, identical in all three writers' argparse
    setup. --models/--experiment/--*-url stay in each script -- tool-specific."""
    p.add_argument("--out", default=None, help="Defaults to results/<experiment>.jsonl.")
    p.add_argument("--limit", type=_positive_int, default=None)
    p.add_argument("--no-resume", action="store_true", help="Ignore any existing rows at --out.")


def default_out_path(experiment):
    """results/<experiment>.jsonl next to this file -- keyed by experiment
    name, not a timestamp, so re-running the same --experiment resumes into
    the same file instead of scattering one run across many."""
    safe = _UNSAFE_FILENAME_CHARS.sub("-", experiment or "unnamed")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{safe}.jsonl")


def ensure_out_dir(out_path):
    """Called from run(), not just from __main__: eval/results/ is gitignored,
    so it doesn't exist in a fresh clone and any non-CLI caller (tests.py)
    would otherwise fail on the open()."""
    directory = os.path.dirname(out_path)  # "" for a bare filename in cwd
    if directory:
        os.makedirs(directory, exist_ok=True)


class ResumeMismatchError(Exception):
    """A row already in --out is for a DIFFERENT question than the one this run
    has under the same question_id. A setup mistake, not a per-question
    failure: it always aborts the whole run."""


def load_existing(out_path, expected, expected_schema=None):
    """question_id -> row, for whatever's already at out_path (empty if the
    file doesn't exist yet -- a brand-new experiment name).

    `expected` maps question_id -> (instruction, correct_letter) for THIS run,
    and every reusable prior row is checked against it. question_id is only a
    positional index into the dataset, so it is not stable across datasets:
    load_tasks() silently switches from gpqa_sample.jsonl to the real
    gpqa_diamond.jsonl the moment that download appears (see gpqa_tasks.py),
    and a variant/baseline run can equally be pointed at a --base-replay built
    from different questions. Either way, resuming across that switch would
    reuse rows answering question A while grading them against question B's
    correct_letter -- silently wrong accuracy, no crash. Refuse instead.
    Called before --out is opened for writing, so aborting leaves it intact.

    `expected_schema`, if given, is checked against each row's own `schema`
    field: pointing --out (or eval/panel.py's --reuse) at a file written by a
    DIFFERENT tool -- e.g. `eval.baseline --out` aimed at an `eval.fuse`
    result by mistake -- must not be silently treated as "already have it".
    Every writer here rebuilds its row from scratch, so that would silently
    drop whatever the other tool had already paid for."""
    if not out_path or not os.path.exists(out_path):
        return {}
    existing = {}
    with open(out_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                # This fires before --out is ever opened for writing, so a
                # corrupt/hand-edited file can be fixed and nothing is lost
                # -- but only if the operator can tell WHICH file and WHERE;
                # the bare JSONDecodeError says neither.
                raise ValueError(f"{out_path!r} line {line_num}: {e}") from e
            existing[row["question_id"]] = row
    for qid, row in existing.items():
        # Rows this run won't reuse anyway can't corrupt it: a failed row gets
        # re-called, and a qid outside `expected` is never read.
        if row.get("failed") or qid not in expected:
            continue
        if (row.get("instruction"), row.get("correct_letter")) != expected[qid]:
            raise ResumeMismatchError(
                f"{out_path!r} already has a row for question_id={qid!r}, but it is a DIFFERENT "
                f"question than the one loaded now -- the questions changed under this --experiment "
                f"name. Reusing it would grade an old answer against the new question's "
                f"correct_letter. Prior instruction starts: {(row.get('instruction') or '')[:80]!r} "
                f"(correct_letter={row.get('correct_letter')!r}); now: {expected[qid][0][:80]!r} "
                f"(correct_letter={expected[qid][1]!r}). Use a new --experiment name, or "
                f"--no-resume to discard the old rows and recompute."
            )
        if expected_schema and row.get("schema") != expected_schema:
            raise ResumeMismatchError(
                f"{out_path!r} already has a row for question_id={qid!r} written by a DIFFERENT tool "
                f"(schema={row['schema']!r}, this tool writes {expected_schema!r}) -- probably the "
                f"wrong --out/--reuse path. Reusing it would silently drop whatever that other tool's "
                f"row already had (fusion/panel/baselines...), which likely already cost real money. "
                f"Point --out at the right file, or --no-resume if you really mean to overwrite it."
            )
    return existing


def run_replay(items, get_qid, process, out_path, existing):
    """The write-loop shared by all three writers: open --out, write one row
    per item, then carry forward every `existing` row this attempt never
    actually wrote a fresh line for -- and return counts.

    That covers two distinct cases with one rule: rows outside this run's
    window entirely (a smaller `--limit` than a prior run, or the underlying
    dataset/--panel file having shrunk for reasons that have nothing to do
    with THIS run's own --limit), AND rows still inside the window but not
    yet reached because the run was interrupted (a real Ctrl-C, or anything
    else escaping `process` -- a per-item try/except inside a tool's own
    `process` can't catch a raw KeyboardInterrupt). The carry-over runs in a
    `finally`, so an interruption mid-loop still preserves every row that
    hasn't been freshly rewritten yet -- otherwise the already-paid rows
    this exists to protect are the exact ones lost, right when a run is
    being interrupted. Without --limit and running to completion, "not
    written this attempt" and "outside the window" are the same set, so
    normal full runs are unaffected.

    `process(item, prior)` -- prior = existing.get(get_qid(item)) -- owns
    everything task-specific (skip/call/fail, build the row, log it) and
    returns `(row, deltas)`; `deltas` is e.g. `{"skipped": 1}` or
    `{"failed": 2}` (a count, not just a label -- one baseline row can fail
    more than one of its N models) added into the running totals, or `{}`
    for the ordinary case."""
    counts = {}
    written = set()
    with open(out_path, "w", encoding="utf-8") as f:
        try:
            for item in items:
                qid = get_qid(item)
                row, deltas = process(item, existing.get(qid))
                for key, delta in (deltas or {}).items():
                    # A typo'd key here (e.g. "skiped") would otherwise
                    # accumulate silently under the wrong name -- the
                    # printed summary is what an operator reads to decide
                    # whether a paid run did what they expected, so a silent
                    # miscount there is exactly the wrong place to be quiet.
                    # "carried" is computed below, not by `process`.
                    assert key in ("skipped", "failed"), f"run_replay: unknown delta key {key!r}"
                    counts[key] = counts.get(key, 0) + delta
                f.write(json.dumps(row) + "\n")
                written.add(qid)
        finally:
            carried = 0
            for qid, row in existing.items():
                if qid not in written:
                    f.write(json.dumps(row) + "\n")
                    carried += 1
            counts["carried"] = counts.get("carried", 0) + carried
    return counts
