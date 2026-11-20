"""Load HotpotQA and convert each row into simple Python objects."""

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Passage:
    """One Wikipedia paragraph that can be shown to the model."""
    title: str
    text: str
    is_gold: bool  # True if the paragraph is needed to answer the question
    source: str = ""  # optional credibility label shown to the model, e.g. "Wikipedia"


@dataclass
class Example:
    """One HotpotQA question with its answer and its 10 paragraphs."""
    id: str
    question: str
    answer: str
    qtype: str  # "bridge" or "comparison"
    passages: list[Passage]


def row_to_example(row: dict) -> Example:
    """Convert one raw HotpotQA row (as given by Hugging Face) into an Example."""
    # The titles of the paragraphs that contain supporting facts are the gold ones.
    gold_titles = set(row["supporting_facts"]["title"])

    passages = []
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        passages.append(Passage(title=title, text="".join(sentences), is_gold=title in gold_titles))

    return Example(
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        qtype=row["type"],
        passages=passages,
    )


def load_hotpotqa(n: int | None = 200, seed: int = 0) -> list[Example]:
    """Load n random questions from the HotpotQA validation set (all of them if n is None).
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [row_to_example(row) for row in ds]

def pick_fewshot_examples(seed: int = 0) -> list[tuple[str, str]]:
    """Pick 3 (question, answer) demonstrations from the TRAIN split.

    One of each kind, so the demonstrations do not push the model towards one
    answer type: a bridge question, a comparison question, and a yes/no question.
    Only "hard" questions (like the validation set) with answers of at most
    4 words (to show the short format we want).
    """
    # streaming=True reads rows one by one instead of downloading the whole train split
    train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)

    picked = {}
    for row in train.shuffle(seed=seed, buffer_size=1000):
        answer = row["answer"]
        if answer.lower() in ("yes", "no"):
            kind = "yes/no"
        else:
            kind = row["type"]  # "bridge" or "comparison"
        if kind not in picked and row["level"] == "hard" and len(answer.split()) <= 4:
            picked[kind] = (row["question"], answer)
        if len(picked) == 3:
            break
    return [picked["bridge"], picked["comparison"], picked["yes/no"]]