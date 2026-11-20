"""Retrieval: a document collection with poisoned versions, and two retrievers.

Every retriever has the same method, search(query, k) -> indices of the k best
passages, so another retriever can be added later without changing the experiment.

    BM25Retriever   -- sparse: word overlap, weighted by rarity (the TF-IDF family)
    DenseRetriever  -- dense: cosine similarity of embeddings (nomic-embed-text via Ollama)
"""

import pickle
import re
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

from ragtrust.data import Example, Passage
from ragtrust.poison import N_VERSIONS, PoisonItem


def build_corpus(examples: list[Example], items: list[PoisonItem], m: int) -> list[Passage]:
    """All paragraphs of all questions (each paragraph once). For every poisoned question,
    its original answer paragraph is replaced by its N_VERSIONS versions at level m:
    m false and N_VERSIONS - m true.

    The paragraph is replaced everywhere, also where it appears as a distractor of
    another question: in this collection, the original no longer exists.
    """
    replaced = {item.original for item in items}
    corpus, seen = [], set()
    for ex in examples:
        for p in ex.passages:
            if p.text not in replaced and p.text not in seen:
                seen.add(p.text)
                corpus.append(Passage(p.title, p.text, is_gold=False))
    for item in items:
        for i in range(N_VERSIONS):
            text = item.false_versions[i] if i < m else item.true_versions[i]
            corpus.append(Passage(item.title, text, is_gold=False))
    return corpus


def _document_text(p: Passage) -> str:
    """What is indexed: title and text, as in Wikipedia search."""
    return f"{p.title}: {p.text}"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """BM25 (Okapi): score(d, q) = sum over query words of IDF(w) * a saturated term
    frequency of w in d, normalised by the length of d."""

    def __init__(self, passages: list[Passage]):
        self.bm25 = BM25Okapi([_tokenize(_document_text(p)) for p in passages])

    def search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return list(np.argsort(-scores)[:k])


class OllamaEmbedder:
    """Embeds texts with an Ollama embedding model. Vectors are cached (in memory and,
    if cache_path is given, on disk), so a text is embedded only once."""

    def __init__(self, model: str = "nomic-embed-text", cache_path: Path | None = None):
        self.model = model
        self.cache_path = cache_path
        self.cache = {}
        if cache_path is not None and cache_path.exists():
            self.cache = pickle.load(open(cache_path, "rb"))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one unit-length vector per text (so dot product = cosine similarity)."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for start in range(0, len(missing), 64):
            batch = missing[start:start + 64]
            self.cache.update(zip(batch, ollama.embed(model=self.model, input=batch)["embeddings"]))
        if missing and self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.cache, open(self.cache_path, "wb"))
        vectors = np.array([self.cache[t] for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class DenseRetriever:
    """Ranks passages by cosine similarity between query and passage embeddings.

    nomic-embed-text expects a task prefix: "search_query: " for queries and
    "search_document: " for the documents.
    """

    def __init__(self, passages: list[Passage], embedder):
        self.embedder = embedder
        self.doc_vectors = embedder.embed([f"search_document: {_document_text(p)}" for p in passages])

    def search(self, query: str, k: int) -> list[int]:
        query_vector = self.embedder.embed([f"search_query: {query}"])[0]
        scores = self.doc_vectors @ query_vector
        return list(np.argsort(-scores)[:k])


def make_retriever(name: str, passages: list[Passage], embedder=None):
    if name == "bm25":
        return BM25Retriever(passages)
    if name == "dense":
        return DenseRetriever(passages, embedder)
    raise ValueError(f"Unknown retriever: {name}")
