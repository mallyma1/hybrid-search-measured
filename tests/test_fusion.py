import math

import pytest

from hsm.fusion import ranked_ids, rrf, rrf_runs


def test_rrf_hand_computed():
    # BM25 ranks a > b > c; dense ranks c > a > d.
    bm25 = {"a": 9.0, "b": 5.0, "c": 1.0}
    dense = {"c": 0.9, "a": 0.8, "d": 0.1}
    fused = rrf([bm25, dense], k=60)
    expected = {
        "a": 1 / 61 + 1 / 62,
        "c": 1 / 63 + 1 / 61,
        "b": 1 / 62,
        "d": 1 / 63,
    }
    assert set(fused) == set(expected)
    for d, s in expected.items():
        assert math.isclose(fused[d], s, rel_tol=1e-12)
    # a and c appear in both lists, so they outrank the single-list documents.
    assert ranked_ids(fused) == ["a", "c", "b", "d"]


def test_rrf_uses_ranks_not_raw_scores():
    # Wildly different score scales must not matter, only order does.
    small = {"x": 0.002, "y": 0.001}
    huge = {"y": 5000.0, "x": 10.0}
    fused = rrf([small, huge], k=60)
    assert math.isclose(fused["x"], fused["y"])


def test_rrf_top_n_and_tie_break_is_deterministic():
    fused = rrf([{"b": 1.0, "a": 1.0, "c": 0.5}], k=60, top_n=2)
    assert list(fused) == ["a", "b"]  # equal scores ordered by id


def test_rrf_smaller_k_rewards_top_ranks_more():
    r1 = {"top": 2.0, "mid": 1.0}
    r2 = {"mid": 2.0, "other": 1.0}
    gap_small_k = rrf([r1, r2], k=1)["top"] - rrf([r1, r2], k=1)["other"]
    gap_big_k = rrf([r1, r2], k=60)["top"] - rrf([r1, r2], k=60)["other"]
    assert gap_small_k > gap_big_k > 0


def test_rrf_runs_handles_query_missing_from_one_arm():
    run_a = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    run_b = {"q1": {"d3": 1.0}}
    fused = rrf_runs([run_a, run_b], k=60)
    assert set(fused) == {"q1", "q2"}
    assert list(fused["q2"]) == ["d2"]


def test_matches_ranx_rrf():
    ranx = pytest.importorskip("ranx")
    a = {"q1": {"d1": 3.0, "d2": 2.0, "d3": 1.0}, "q2": {"d1": 2.0, "d2": 1.0}}
    b = {"q1": {"d3": 3.0, "d1": 2.0}, "q2": {"d2": 2.0, "d4": 1.0}}
    ours = rrf_runs([a, b], k=60)
    theirs = ranx.fuse([ranx.Run(a, name="a"), ranx.Run(b, name="b")],
                       method="rrf", params={"k": 60}).to_dict()
    for q in ours:
        for d, s in ours[q].items():
            assert math.isclose(s, theirs[q][d], rel_tol=1e-9)
