"""ragtrust: measuring the reliability of Retrieval-Augmented Generation (RAG)
under controlled evidence poisoning.

Modules:
    data       -- load HotpotQA into Example and Passage objects
    prompts    -- build chat messages (system prompts, few-shot demonstrations, source labels)
    llm        -- OllamaLLM: one place for model name and generation settings
    metrics    -- Exact Match, token F1, and the strict answer classification
    runs       -- the final question selection (stratified sample) and how to load it
    poison     -- generate false and true versions of the answer paragraph; build poisoned contexts
    retrieval  -- the document collection with poison, BM25 and dense retrievers
    experiment -- run baselines, controlled poisoning, retrieval poisoning and self-consistency
    analysis   -- from saved answers to reported numbers: lenient scoring, knowledge groups, weights, tables
    stats      -- confidence intervals and tests (Wilson, McNemar, Fisher, Holm, bootstrap)
"""

__version__ = "0.1.0"
