"""Replay-file plumbing shared by eval/panel.py, eval/fuse.py, eval/baseline.py.

All three write the same kind of file (one JSON row per question, keyed by
question_id) and all three resume into it the same way, so the naming, the
directory creation and the resume-safety check live here once instead of
three times. Everything task-specific -- what a row contains, what gets
called, what counts as a failure -- stays in each script.
"""
import json
import os
import re

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


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
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
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
        if expected_schema and row.get("schema") and row["schema"] != expected_schema:
            raise ResumeMismatchError(
                f"{out_path!r} already has a row for question_id={qid!r} written by a DIFFERENT tool "
                f"(schema={row['schema']!r}, this tool writes {expected_schema!r}) -- probably the "
                f"wrong --out/--reuse path. Reusing it would silently drop whatever that other tool's "
                f"row already had (fusion/panel/baselines...), which likely already cost real money. "
                f"Point --out at the right file, or --no-resume if you really mean to overwrite it."
            )
    return existing


def carry_over_unprocessed(f, existing, expected):
    """Copy prior rows for questions OUTSIDE this run's window into the
    freshly rewritten --out, and return how many.

    Callers only invoke this when --limit is set. Every run truncates --out
    and rewrites it, so without this, re-running a `--limit 5` smoke test
    against an --out that already holds a finished 198-question run would
    delete the other 193 -- rows that cost real API spend. --limit scopes what
    this run *calls*, not what the file is allowed to keep. (--no-resume is the
    way to genuinely shrink a file: it starts from no prior rows at all, so
    there is nothing here to carry.) Without --limit, `expected` already covers
    the whole dataset / base file, so anything else in --out is stale rather
    than something worth preserving.

    Written in the prior file's own order, after the rows this run handled,
    which for --limit's leading window keeps question_id order."""
    carried = 0
    for qid, row in existing.items():
        if qid not in expected:
            f.write(json.dumps(row) + "\n")
            carried += 1
    return carried
