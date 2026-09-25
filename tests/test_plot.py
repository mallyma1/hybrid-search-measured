import json
from pathlib import Path

import pytest

from hsm import plot

RESULTS = Path(__file__).resolve().parent.parent / "results"


def toy_results():
    return {
        "significance": {"comparisons": [
            {"a": "rrf", "b": "bm25", "metric": "ndcg@10", "diff": 0.1, "p_value": 1e-9},
            {"a": "rrf", "b": "dense", "metric": "ndcg@10", "diff": -0.04, "p_value": 0.003},
            {"a": "rrf_rerank", "b": "rrf", "metric": "ndcg@10", "diff": 0.006, "p_value": 0.4429},
        ]}
    }


def test_verdict_follows_sign_and_p_value():
    res = toy_results()
    assert plot.verdict(res, "rrf", "bm25", "ndcg@10") == "hybrid beat BM25 (p < 0.0001)"
    assert plot.verdict(res, "rrf", "dense", "ndcg@10") == "dense beat hybrid (p = 0.0030)"
    assert plot.verdict(res, "rrf_rerank", "rrf", "ndcg@10") == (
        "rerank vs hybrid not significant (p = 0.4429)")


def test_intervals_rebuild_the_saved_means(tmp_path):
    csv_path = tmp_path / "per_query.csv"
    csv_path.write_text("query_id,arm,ndcg@10\nq1,a,0.2\nq2,a,0.4\nq3,a,0.6\n")
    ci = plot.arm_intervals(csv_path, {"a": {"ndcg@10": 0.4}})
    lo, hi = ci["a"]["ndcg@10"]
    assert lo < 0.4 < hi
    assert abs((lo + hi) / 2 - 0.4) < 1e-12
    with pytest.raises(ValueError):
        plot.arm_intervals(csv_path, {"a": {"ndcg@10": 0.5}})  # stale file is refused
    assert plot.arm_intervals(tmp_path / "missing.csv", {}) == {}


@pytest.mark.skipif(not (RESULTS / "results.json").exists(), reason="no saved results")
def test_charts_render_from_saved_results(tmp_path):
    assert plot.main(["--results-dir", str(RESULTS), "--out-dir", str(tmp_path)]) == 0
    for name in ("ndcg10.png", "metrics.png"):
        data = (tmp_path / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
    res = json.loads((RESULTS / "results.json").read_text())
    assert set(plot.ARMS) == set(res["metrics"])
