"""Metrics (ranx) and paired significance tests."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

METRICS = ["ndcg@10", "recall@5", "recall@10", "recall@100", "mrr@10"]


def score_runs(qrels_dict, runs_dict, metrics=METRICS):
    """Mean and per-query scores for every arm.

    Returns (means, per_query) where means[arm][metric] is a float and
    per_query[arm][metric] is a dict query_id -> score over every judged query
    (a query an arm returned nothing for scores 0)."""
    from ranx import Qrels, Run, evaluate

    qrels = Qrels(qrels_dict)
    qids = sorted(qrels_dict)
    means, per_query = {}, {}
    for arm, run_dict in runs_dict.items():
        full = {q: run_dict.get(q, {}) for q in qids}
        # ranx needs at least one document per query; a sentinel id that is never
        # relevant stands in for an empty result list and scores 0.
        full = {q: (r if r else {"__none__": 0.0}) for q, r in full.items()}
        run = Run(full, name=arm)
        m = evaluate(qrels, run, metrics, return_mean=True, make_comparable=True)
        means[arm] = {k: float(v) for k, v in m.items()}
        per_query[arm] = {
            k: {q: float(run.scores[k][q]) for q in qids} for k in metrics
        }
    return means, per_query


def paired_test(per_query, a: str, b: str, metric: str):
    """Two-sided paired t-test of arm a against arm b on one metric."""
    qids = sorted(per_query[a][metric])
    xa = np.array([per_query[a][metric][q] for q in qids])
    xb = np.array([per_query[b][metric][q] for q in qids])
    d = xa - xb
    n = len(d)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    wins = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    ties = n - wins - losses
    if sd == 0.0:
        p, lo, hi = 1.0, mean_d, mean_d
        note = "identical per query" if mean_d == 0.0 else "constant difference"
    else:
        p = float(stats.ttest_rel(xa, xb).pvalue)
        half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        lo, hi = mean_d - half, mean_d + half
        note = ""
    return {
        "a": a, "b": b, "metric": metric, "n": n,
        "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
        "diff": mean_d, "ci95": [lo, hi],
        "rel_change_pct": (100.0 * mean_d / float(xb.mean())) if xb.mean() else None,
        "p_value": p, "wins": wins, "ties": ties, "losses": losses, "note": note,
    }


def ranx_pvalues(qrels_dict, runs_dict, pairs, metrics=METRICS):
    """ranx's built-in paired Student's t-test, as a cross-check on scipy."""
    from ranx import Qrels, Run, compare

    qids = sorted(qrels_dict)
    runs = []
    for arm, run_dict in runs_dict.items():
        full = {q: (run_dict.get(q) or {"__none__": 0.0}) for q in qids}
        runs.append(Run(full, name=arm))
    rep = compare(Qrels(qrels_dict), runs, metrics, stat_test="student",
                  max_p=0.05, make_comparable=True).to_dict()
    out = {}
    for a, b in pairs:
        for m in metrics:
            out[(a, b, m)] = float(rep[a]["comparisons"][b][m])
    return out
