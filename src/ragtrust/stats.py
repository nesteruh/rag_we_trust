"""Small statistics toolbox for the final analysis (standard library + numpy only).

Everything here is simple enough to explain at the oral exam:

    wilson          -- 95% confidence interval for one proportion
    mcnemar_exact   -- paired test: the same questions under two conditions
    fisher_exact    -- unpaired test: two different groups of questions
    holm            -- correction when several hypotheses are tested together
    bootstrap_ci    -- confidence interval by resampling QUESTIONS (not rows)
"""

from math import comb, sqrt

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion k / n. Better than the textbook
    p +- 1.96 * se when n is small or p is close to 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test (two-sided p-value).

    For paired yes/no outcomes only the DISCORDANT questions carry information:
    b = yes under condition 1 and no under condition 2, c = the opposite.
    If the conditions do not differ, each discordant question is a fair coin,
    so b ~ Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Fisher's exact test (two-sided p-value) for the 2x2 table [[a, b], [c, d]]
    (rows = two independent groups, columns = outcome yes / no).
    Sums the probabilities of all tables that are at most as likely as the observed one."""
    row1, row2, col1, n = a + b, c + d, a + c, a + b + c + d
    if n == 0 or row1 == 0 or row2 == 0:
        return 1.0

    def prob(x):
        return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values. Sort the p-values; the smallest is multiplied
    by the number of tests, the next by one less, and so on (kept monotone)."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[name] = running
    return adjusted


def bootstrap_ci(statistic, n_units: int, n_boot: int = 2000, seed: int = 0, strata=None) -> tuple[float, float]:
    """95% percentile bootstrap interval for statistic(index_array).

    The unit that is resampled is the QUESTION: statistic receives the indices of the
    resampled questions. strata (one label per question, optional): resample inside
    each stratum separately, as the questions were sampled (stratified sample).
    """
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(n_units)]
    else:
        strata = np.asarray(strata)
        groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        values.append(statistic(idx))
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return (float(lo), float(hi))
