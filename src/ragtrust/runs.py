"""The final question selection: how it was drawn (notebook 00) and how to load it.

400 questions known by at least one model and 400 known by neither, sampled separately
(stratified), saved in data/poison/final_selection.json.
"""

import json
import random
from pathlib import Path

from ragtrust.data import Example, load_hotpotqa

SELECTION_PATH = Path(__file__).resolve().parents[2] / "data" / "poison" / "final_selection.json"


def load_selection() -> list[Example]:
    """The 800 selected questions, in the saved order (the known stratum first)."""
    selection = json.load(open(SELECTION_PATH))
    by_id = {ex.id: ex for ex in load_hotpotqa(n=None)}
    return [by_id[i] for i in selection["ids"]]


def stratified_sample(candidates: list[str], known: set[str], n_known: int, n_unknown: int, seed: int = 0) -> dict:
    """Sample n_known ids from the known candidates and n_unknown from the others.

    Sampling the two groups separately guarantees enough known questions, which a
    plain random sample does not (about 1 in 4 questions is known). Consequence:
    results must be reported per group, or weighted, never pooled.
    """
    rng = random.Random(seed)
    known_ids = sorted(i for i in candidates if i in known)        # sorted: same input order every time
    unknown_ids = sorted(i for i in candidates if i not in known)
    pick_known = rng.sample(known_ids, min(n_known, len(known_ids)))
    pick_unknown = rng.sample(unknown_ids, min(n_unknown, len(unknown_ids)))
    return {
        "seed": seed,
        "available": {"known": len(known_ids), "unknown": len(unknown_ids)},
        "ids": pick_known + pick_unknown,
        "group": {**{i: "known" for i in pick_known}, **{i: "unknown" for i in pick_unknown}},
    }
