import math

from hsm.evaluation import paired_test, score_runs

QRELS = {
    "q1": {"d1": 1},            # one relevant document
    "q2": {"d2": 1, "d3": 1},   # two relevant documents
}


def test_perfect_ranking_scores_one():
    run = {"q1": {"d1": 3.0, "x": 1.0}, "q2": {"d2": 3.0, "d3": 2.0, "x": 1.0}}
    means, _ = score_runs(QRELS, {"perfect": run})
    for m in ["ndcg@10", "recall@10", "recall@100", "mrr@10"]:
        assert math.isclose(means["perfect"][m], 1.0), m


def test_hand_computed_values():
    # q1: relevant at rank 2. q2: relevant at ranks 1 and 3.
    run = {
        "q1": {"x": 3.0, "d1": 2.0, "y": 1.0},
        "q2": {"d2": 3.0, "y": 2.0, "d3": 1.0},
    }
    means, pq = score_runs(QRELS, {"arm": run})
    # MRR@10: q1 = 1/2, q2 = 1/1
    assert math.isclose(pq["arm"]["mrr@10"]["q1"], 0.5)
    assert math.isclose(pq["arm"]["mrr@10"]["q2"], 1.0)
    assert math.isclose(means["arm"]["mrr@10"], 0.75)
    # nDCG@10 with binary gains: q1 = (1/log2(3)) / 1
    assert math.isclose(pq["arm"]["ndcg@10"]["q1"], 1 / math.log2(3))
    # q2 = (1 + 1/log2(4)) / (1 + 1/log2(3))
    assert math.isclose(pq["arm"]["ndcg@10"]["q2"], (1 + 0.5) / (1 + 1 / math.log2(3)))
    # Recall@1 style cut: recall@5 finds everything here
    assert math.isclose(means["arm"]["recall@5"], 1.0)


def test_recall_cutoff():
    # q2's second relevant document sits at rank 11, outside the top 10.
    q2 = {"d2": 100.0}
    q2.update({f"n{i}": 50.0 - i for i in range(9)})
    q2["d3"] = 1.0
    means, pq = score_runs(QRELS, {"arm": {"q1": {"d1": 1.0}, "q2": q2}})
    assert math.isclose(pq["arm"]["recall@10"]["q2"], 0.5)
    assert math.isclose(pq["arm"]["recall@100"]["q2"], 1.0)


def test_empty_result_scores_zero():
    means, pq = score_runs(QRELS, {"arm": {"q1": {"d1": 1.0}}})  # nothing for q2
    assert pq["arm"]["ndcg@10"]["q2"] == 0.0
    assert math.isclose(means["arm"]["mrr@10"], 0.5)


def test_paired_test_identical_arms():
    run = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    _, pq = score_runs(QRELS, {"a": run, "b": run})
    res = paired_test(pq, "a", "b", "ndcg@10")
    assert res["diff"] == 0.0 and res["p_value"] == 1.0 and res["ties"] == 2
