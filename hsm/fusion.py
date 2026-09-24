"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

score(d) = sum over input rankings r of 1 / (k + rank_r(d)), rank starting at 1.
A document missing from a ranking contributes nothing from that ranking.
Ties are broken by document id so the output is deterministic.
"""
from __future__ import annotations

Ranking = dict[str, float]  # doc_id -> score, higher is better


def ranked_ids(ranking: Ranking) -> list[str]:
    """Doc ids ordered best first; ties broken by doc id."""
    return [d for d, _ in sorted(ranking.items(), key=lambda x: (-x[1], x[0]))]


def rrf(rankings: list[Ranking], k: int = 60, top_n: int | None = None) -> Ranking:
    """Fuse several rankings for one query with reciprocal rank fusion."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranked_ids(ranking), start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    if top_n is not None:
        ordered = ordered[:top_n]
    return dict(ordered)


def rrf_runs(runs: list[dict[str, Ranking]], k: int = 60, top_n: int | None = None):
    """Fuse whole runs (query_id -> ranking) query by query."""
    qids = set().union(*(r.keys() for r in runs))
    return {q: rrf([r.get(q, {}) for r in runs], k=k, top_n=top_n) for q in sorted(qids)}
