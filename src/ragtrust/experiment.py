"""Run one experimental setting over a list of questions and collect the results."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from ragtrust.data import Example, Passage
from ragtrust.llm import OllamaLLM
from ragtrust.metrics import classify_answer, contains_answer, exact_match, f1_score, normalize_answer
from ragtrust.poison import N_VERSIONS, PoisonItem, build_poisoned_context
from ragtrust.prompts import build_messages
from ragtrust.retrieval import build_corpus, make_retriever

SETTINGS = ("closed_book", "gold_only", "distractor")


def select_passages(ex: Example, setting: str) -> list[Passage] | None:
    """Which passages the model sees in each setting.

    closed_book: none
    gold_only:   the 2 gold paragraphs
    distractor:  all 10 paragraphs (2 gold + 8 distractors), in dataset order
    """
    if setting == "closed_book":
        return None
    if setting == "gold_only":
        return [p for p in ex.passages if p.is_gold]
    if setting == "distractor":
        return ex.passages
    raise ValueError(f"Unknown setting: {setting}")


def run_setting(
    llm: OllamaLLM,
    examples: list[Example],
    setting: str,
    fewshot: list[tuple[str, str]] = (),
) -> pd.DataFrame:
    """Ask every question under one setting. One row per question.

    The question id is kept so that settings can be compared question by
    question (paired comparison), not only by their averages.
    """
    rows = []
    for ex in tqdm(examples, desc=setting):
        passages = select_passages(ex, setting)
        pred = llm.chat(build_messages(ex.question, fewshot, passages))
        rows.append({
            "id": ex.id,
            "model": llm.model,
            "setting": setting,
            "qtype": ex.qtype,
            "question": ex.question,
            "gold": ex.answer,
            "pred": pred,
            "em": exact_match(pred, ex.answer),
            "f1": f1_score(pred, ex.answer),
            "match": contains_answer(pred, ex.answer),
        })
    return pd.DataFrame(rows)


def run_poisoned(
    llm: OllamaLLM,
    examples: list[Example],
    items: list[PoisonItem],
    fewshot: list[tuple[str, str]] = (),
    seed: int = 0,
    prompt: str = "baseline",
    credibility: str = "none",
) -> pd.DataFrame:
    """Ask every poisoned question at every level m = 0..N_VERSIONS. One row per (question, m).

    category: correct / poisoned / hedged / other (see metrics.classify_answer).
    m = 0 is the clean control: 4 reworded true versions, nothing false.
    """
    by_id = {ex.id: ex for ex in examples}
    rows = []
    for item in tqdm(items, desc=f"poisoned ({prompt}, {credibility})"):
        ex = by_id[item.id]
        for m in range(N_VERSIONS + 1):
            passages, poison_positions = build_poisoned_context(ex, item, m, seed, credibility)
            pred = llm.chat(build_messages(ex.question, fewshot, passages, prompt))
            rows.append({
                "id": item.id,
                "model": llm.model,
                "prompt": prompt,
                "credibility": credibility,
                "m": m,
                "rate": m / N_VERSIONS,
                "question": item.question,
                "gold": item.gold,
                "fake": item.fake,
                "answer_type": item.answer_type,
                "pred": pred,
                "category": classify_answer(pred, item.gold, item.fake),
                "em": exact_match(pred, item.gold),
                "f1": f1_score(pred, item.gold),
                "poison_positions": poison_positions,
            })
    return pd.DataFrame(rows)


def cached_run(path: Path, llm: OllamaLLM, run: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load the results from path if they exist; otherwise call run(), save the
    results there, and save the generation settings next to them (.json)."""
    if path.exists():
        return pd.read_csv(path)
    df = run()
    df.to_csv(path, index=False)
    json.dump(llm.settings(), open(path.with_suffix(".json"), "w"), indent=2)
    return df


def run_retrieval(
    llm: OllamaLLM | None,
    examples: list[Example],
    items: list[PoisonItem],
    retriever: str,
    embedder=None,
    k: int = 5,
    fewshot: list[tuple[str, str]] = (),
    prompt: str = "baseline",
) -> pd.DataFrame:
    """For every level m: build the collection, retrieve the top k passages for every
    poisoned question, and (if llm is given) ask the model. One row per (question, m).

    With llm=None only the retrieval is measured (fast, no generation).
    Passages are given to the model in rank order, as a real RAG system would.
    """
    by_id = {ex.id: ex for ex in examples}
    rows = []
    for m in range(N_VERSIONS + 1):
        corpus = build_corpus(examples, items, m)
        index = make_retriever(retriever, corpus, embedder)
        for item in tqdm(items, desc=f"{retriever}, m={m}"):
            ex = by_id[item.id]
            hits = [corpus[i] for i in index.search(ex.question, k)]
            texts = {p.text for p in hits}
            bridge = next(p.text for p in ex.passages if p.is_gold and p.title != item.title)
            row = {
                "id": item.id,
                "retriever": retriever,
                "m": m,
                "rate": m / N_VERSIONS,
                "n_false": len(texts & set(item.false_versions[:m])),
                "n_true": len(texts & set(item.true_versions[m:])),
                "bridge_retrieved": bridge in texts,
                "gold": item.gold,
                "fake": item.fake,
            }
            if llm is not None:
                pred = llm.chat(build_messages(ex.question, fewshot, hits, prompt))
                row.update({
                    "model": llm.model,
                    "pred": pred,
                    "category": classify_answer(pred, item.gold, item.fake),
                    "em": exact_match(pred, item.gold),
                    "f1": f1_score(pred, item.gold),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def run_self_consistency(
    llm: OllamaLLM,
    examples: list[Example],
    items: list[PoisonItem],
    n_samples: int = 5,
    temperature: float = 0.7,
    fewshot: list[tuple[str, str]] = (),
    seed: int = 0,
) -> pd.DataFrame:
    """Sampling-based self-consistency (Wang et al., 2022), controlled setting, baseline prompt.

    Every (question, m) is answered n_samples times at temperature > 0, each sample with
    its own seed. If the answers agree, the model is consistent; if they spread over the
    true and the false answer, the conflict in the documents shows up as uncertainty.
    One row per (question, m, sample).
    """
    by_id = {ex.id: ex for ex in examples}
    samplers = [replace(llm, temperature=temperature, seed=seed + 1 + s) for s in range(n_samples)]
    rows = []
    for item in tqdm(items, desc=f"self-consistency ({llm.model})"):
        ex = by_id[item.id]
        for m in range(N_VERSIONS + 1):
            passages, _ = build_poisoned_context(ex, item, m, seed)
            messages = build_messages(ex.question, fewshot, passages)
            for s, sampler in enumerate(samplers):
                pred = sampler.chat(messages)
                rows.append({
                    "id": item.id,
                    "model": llm.model,
                    "m": m,
                    "sample": s,
                    "pred": pred,
                    "normalized": normalize_answer(pred),
                    "category": classify_answer(pred, item.gold, item.fake),
                })
    return pd.DataFrame(rows)
