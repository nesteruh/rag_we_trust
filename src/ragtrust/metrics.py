"""Answer-correctness metrics: Exact Match (EM) and token F1.

Same definitions as the official SQuAD / HotpotQA evaluation scripts, so our
numbers are comparable with the literature.
"""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, remove articles (a, an, the), fix spaces.

    Example: "The Chief of Protocol." -> "chief of protocol"
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalized strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Word-overlap F1 between prediction and gold, after normalization."""
    pred = normalize_answer(prediction)
    ref = normalize_answer(gold)

    # Official HotpotQA rule: yes/no answers get no partial credit.
    if pred in {"yes", "no"} or ref in {"yes", "no"}:
        return float(pred == ref)

    pred_tokens = pred.split()
    ref_tokens = ref.split()
    shared = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)
    recall = shared / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, answer: str) -> bool:
    """True if the normalized answer appears in the normalized prediction
    as a sequence of whole words.

    Whole words, so "no" is not found inside "not".
    Example: contains_answer("They are both American.", "American") -> True
    """
    pred_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    n = len(answer_tokens)
    if n == 0:
        return False
    return any(pred_tokens[i:i + n] == answer_tokens for i in range(len(pred_tokens) - n + 1))


def classify_answer(prediction: str, gold: str, poison: str) -> str:
    """Put a prediction into one of four categories, given the true and the false answer.

    correct  -- contains the gold answer only
    poisoned -- contains the poison answer only
    hedged   -- contains both
    other    -- contains neither (a different answer, or no answer)
    """
    has_gold = contains_answer(prediction, gold)
    has_poison = contains_answer(prediction, poison)
    if has_gold and has_poison:
        return "hedged"
    if has_gold:
        return "correct"
    if has_poison:
        return "poisoned"
    return "other"
