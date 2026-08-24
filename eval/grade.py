"""AlpacaEval-style pairwise judge over a replay file: fusion (A) vs baseline (B).
NOTE: the prompt below is a good-faith stand-in modeled on AlpacaEval's public pairwise
methodology, not verified byte-identical to the official annotator template -- swap in
the official prompt before treating scores as comparable to the public leaderboard.
Run: python3 -m eval.grade <replay.jsonl>
"""
import json
import os
import sys

from llm_client import call_llm

JUDGE_MODEL = os.environ.get("ZG_GRADE_JUDGE_MODEL", "judge-model")


def judge_pair(instruction, output_a, output_b):
    prompt = (
        f"Instruction:\n{instruction}\n\nResponse A:\n{output_a}\n\nResponse B:\n{output_b}\n\n"
        "Which response better follows the instruction? Reply with exactly one letter: A or B."
    )
    msg = call_llm(JUDGE_MODEL, [{"role": "user", "content": prompt}])
    return "A" if (msg.get("content") or "").strip().upper().startswith("A") else "B"


def grade_replay(path):
    wins = total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            if judge_pair(row["instruction"], row["fusion"]["content"], row["baseline"]["content"]) == "A":
                wins += 1
    return {"win_rate": wins / total if total else 0.0, "n": total}


if __name__ == "__main__":
    print(json.dumps(grade_replay(sys.argv[1])))
