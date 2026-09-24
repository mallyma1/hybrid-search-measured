"""Run every arm on BEIR FiQA-2018 (test) and write results/.

    python -m hsm.run [--device auto|mps|cpu|cuda] [--no-rerank]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from hsm import retrievers as R
from hsm.data import ensure_fiqa, load_fiqa
from hsm.evaluation import METRICS, paired_test, ranx_pvalues, score_runs
from hsm.fusion import rrf_runs

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 100          # depth retrieved by every arm (needed for recall@100)
RRF_K = 60           # the constant from Cormack et al. (2009); not tuned here
RERANK_DEPTH = 50    # fused candidates rescored by the cross-encoder

ARM_LABELS = {
    "bm25": "BM25 full-text only",
    "dense": "Dense vectors only (bge-small-en-v1.5)",
    "rrf": "Hybrid: RRF of BM25 + dense (k=60)",
    "rrf_rerank": "Experiment: hybrid + cross-encoder rerank of top 50",
}
COMPARISONS = [("rrf", "bm25"), ("rrf", "dense"), ("rrf_rerank", "rrf"), ("rrf_rerank", "dense"), ("dense", "bm25")]


def write_trec(run, path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for qid in sorted(run):
            ranking = run[qid]
            ids = sorted(ranking, key=lambda d: (-ranking[d], d))
            for rank, d in enumerate(ids, start=1):
                f.write(f"{qid} Q0 {d} {rank} {ranking[d]:.6f} {tag}\n")


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def hardware() -> dict:
    info = {"machine": platform.machine(), "python": platform.python_version()}
    if platform.system() == "Darwin":
        info["os"] = "macOS " + platform.mac_ver()[0]
        try:
            info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          capture_output=True, text=True).stdout.strip()
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                     capture_output=True, text=True).stdout.strip())
            info["ram_gb"] = round(mem / 2**30)
        except Exception:  # noqa: BLE001
            pass
    else:
        info["os"] = platform.platform()
    return info


def _log_invocation(cache_dir: Path, started: float, status: str) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / "run_log.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps({"started_unix": started, "wall_s": time.time() - started,
                            "status": status}) + "\n")
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop the resumable stages after this long; exit code 3 means rerun to resume")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    t_start = time.perf_counter()
    wall_start = time.time()
    deadline = wall_start + args.max_seconds if args.max_seconds else None
    started = dt.datetime.now(dt.timezone.utc)
    data_dir, res_dir = Path(args.data_dir), Path(args.results_dir)
    cache_dir = data_dir / "cache"
    device = R.pick_device(args.device)
    timings = {}

    t = time.perf_counter()
    prov = ensure_fiqa(data_dir)
    corpus, queries, qrels = load_fiqa(data_dir, "test")
    timings["load_data_s"] = time.perf_counter() - t
    n_qrels = sum(len(v) for v in qrels.values())
    print(f"[data] {len(corpus)} docs, {len(queries)} test queries, {n_qrels} judgements", flush=True)
    missing = set(qrels) - set(queries)
    assert not missing, f"{len(missing)} judged queries have no text"

    doc_ids = list(corpus)
    doc_texts = [corpus[d] for d in doc_ids]
    qids = sorted(queries)
    qtexts = [queries[q] for q in qids]

    try:
        print(f"[dense] corpus vectors on {device}", flush=True)
        doc_emb, enc_info = R.encode_corpus(doc_texts, device, cache_dir, deadline)

        print("[bm25] indexing and searching", flush=True)
        bm25_run, bm25_info = R.bm25_search(doc_ids, doc_texts, qids, qtexts, top_k=TOP_K)
        print(f"[bm25] {bm25_info}", flush=True)

        dense_run, dense_info = R.dense_search(doc_ids, doc_emb, qids, qtexts, device, top_k=TOP_K)
        del doc_emb
        print(f"[dense] {dense_info}", flush=True)

        t = time.perf_counter()
        rrf_run = rrf_runs([bm25_run, dense_run], k=RRF_K, top_n=TOP_K)
        fuse_s = time.perf_counter() - t

        runs = {"bm25": bm25_run, "dense": dense_run, "rrf": rrf_run}
        rerank_info = None
        if not args.no_rerank:
            print(f"[rerank] cross-encoder over top {RERANK_DEPTH} fused, on {device}", flush=True)
            runs["rrf_rerank"], rerank_info = R.cross_encoder_rerank(
                rrf_run, corpus, queries, device, cache_dir, depth=RERANK_DEPTH, deadline=deadline)
            print(f"[rerank] {rerank_info}", flush=True)
    except R.Incomplete as exc:
        _log_invocation(cache_dir, wall_start, f"incomplete: {exc}")
        print(f"[stop] time budget reached, {exc}. Run again to resume.", flush=True)
        sys.exit(3)

    print("[eval] scoring", flush=True)
    t = time.perf_counter()
    means, per_query = score_runs(qrels, runs)
    pairs = [(a, b) for a, b in COMPARISONS if a in runs and b in runs]
    ranx_p = ranx_pvalues(qrels, runs, pairs)
    tests = []
    for a, b in pairs:
        for m in METRICS:
            row = paired_test(per_query, a, b, m)
            row["p_value_ranx"] = ranx_p[(a, b, m)]
            tests.append(row)
    timings["evaluate_s"] = time.perf_counter() - t

    n = len(qids)
    timings.update({
        "dense_encode_corpus_s": enc_info["encode_corpus_s"],
        "bm25_index_s": bm25_info["index_s"],
        "bm25_search_s": bm25_info["search_s"],
        "dense_search_s": dense_info["search_s"],
        "rrf_fuse_s": fuse_s,
    })
    per_query_ms = {
        "bm25_search": 1000 * bm25_info["search_s"] / n,
        "dense_query_encode_and_search": 1000 * dense_info["search_s"] / n,
        "rrf_fusion_step": 1000 * fuse_s / n,
    }
    if rerank_info:
        timings["rerank_s"] = rerank_info["rerank_s"]
        per_query_ms["cross_encoder_rerank_step"] = 1000 * rerank_info["rerank_s"] / n
    final_invocation_s = time.perf_counter() - t_start
    invocations = _log_invocation(cache_dir, wall_start, "complete")
    wall_all = sum(i["wall_s"] for i in invocations)
    compute_total = sum(timings.values())

    # ------------------------------------------------------------------ outputs
    res_dir.mkdir(parents=True, exist_ok=True)
    for arm, run in runs.items():
        write_trec(run, res_dir / "runs" / f"{arm}.trec.gz", arm)
    with open(res_dir / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "arm"] + METRICS)
        for q in qids:
            for arm in runs:
                w.writerow([q, arm] + [f"{per_query[arm][m][q]:.6f}" for m in METRICS])

    pkgs = ["numpy", "scipy", "ranx", "bm25s", "PyStemmer", "torch", "transformers",
            "sentence-transformers", "pyarrow"]
    result = {
        "date_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "name": "BEIR FiQA-2018", "split": "test", "n_queries": n,
            "n_docs": len(corpus), "n_judgements": n_qrels, **prov,
        },
        "arms": ARM_LABELS if rerank_info else {k: v for k, v in ARM_LABELS.items() if k != "rrf_rerank"},
        "config": {
            "bm25": {"library": "bm25s", "method": "lucene", "k1": R.BM25_K1, "b": R.BM25_B,
                     "stopwords": "en", "stemmer": "Snowball English (PyStemmer)"},
            "dense": {"model": R.DENSE_MODEL, "revision": R.DENSE_REVISION,
                      "query_instruction": R.DENSE_QUERY_INSTRUCTION,
                      "similarity": "cosine (dot product of L2-normalised vectors), exact numpy search",
                      "dim": dense_info["dim"], "max_seq_length": dense_info["max_seq_length"],
                      "encode_batch": R.ENCODE_BATCH},
            "fusion": {"method": "reciprocal rank fusion", "k": RRF_K,
                       "inputs": "top 100 of BM25 and top 100 of dense", "output_depth": TOP_K},
            "rerank": ({"model": R.RERANK_MODEL, "revision": R.RERANK_REVISION,
                        "depth": RERANK_DEPTH, "max_length": 512,
                        "pairs_scored": rerank_info["pairs_scored"],
                        "note": "ranks 51-100 keep fused order below the reranked block"}
                       if rerank_info else None),
            "retrieval_depth": TOP_K, "device": device,
        },
        "metrics": means,
        "significance": {"test": "two-sided paired Student's t-test over queries (scipy; ranx cross-check)",
                         "comparisons": tests},
        "timings_s": timings,
        "latency_ms_per_query": per_query_ms,
        "total_compute_s": compute_total,
        "final_invocation_wall_s": final_invocation_s,
        "invocations": len(invocations),
        "wall_clock_all_invocations_s": wall_all,
        "hardware": hardware(),
        "versions": {p: version(p) for p in pkgs},
    }
    (res_dir / "results.json").write_text(json.dumps(result, indent=2))
    (res_dir / "results.md").write_text(render_markdown(result))
    print((res_dir / "results.md").read_text())


def render_markdown(r: dict) -> str:
    ds, cfg = r["dataset"], r["config"]
    lines = [
        "# Results",
        "",
        f"- Run date (UTC): {r['date_utc']}",
        f"- Dataset: {ds['name']}, {ds['split']} split, {ds['n_queries']} queries, "
        f"{ds['n_docs']:,} documents, {ds['n_judgements']:,} relevance judgements",
        f"- Data source: {ds['source']}",
        f"- Dense model: `{cfg['dense']['model']}` @ `{cfg['dense']['revision'][:12]}`",
    ]
    if cfg["rerank"]:
        lines.append(f"- Reranker (experiment): `{cfg['rerank']['model']}` @ `{cfg['rerank']['revision'][:12]}`")
    lines += [
        f"- BM25: bm25s {r['versions']['bm25s']}, Lucene variant, k1={cfg['bm25']['k1']}, b={cfg['bm25']['b']}",
        f"- Fusion: reciprocal rank fusion, k={cfg['fusion']['k']}",
        f"- Device: {cfg['device']}; hardware: {r['hardware'].get('chip', r['hardware']['machine'])}, "
        f"{r['hardware'].get('ram_gb', '?')} GB RAM, {r['hardware'].get('os', '')}",
        f"- Compute time, all stages: {r['total_compute_s']:.1f} s; wall clock {r['wall_clock_all_invocations_s']:.1f} s "
        f"over {r['invocations']} invocation(s) (the slow stages checkpoint and resume)",
        "",
        "## Scores (mean over queries)",
        "",
        "| Arm | nDCG@10 | Recall@5 | Recall@10 | Recall@100 | MRR@10 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, label in r["arms"].items():
        m = r["metrics"][arm]
        lines.append(f"| {label} | " + " | ".join(f"{m[k]:.4f}" for k in METRICS) + " |")
    lines += [
        "",
        "## Paired t-tests (two-sided, per query)",
        "",
        "| Comparison | Metric | A | B | A minus B | 95% CI | Change | p (scipy) | p (ranx) | Better / same / worse |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in r["significance"]["comparisons"]:
        rel = f"{t['rel_change_pct']:+.1f}%" if t["rel_change_pct"] is not None else "n/a"
        lines.append(
            f"| {t['a']} vs {t['b']} | {t['metric']} | {t['mean_a']:.4f} | {t['mean_b']:.4f} | "
            f"{t['diff']:+.4f} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | {rel} | "
            f"{fmt_p(t['p_value'])} | {fmt_p(t['p_value_ranx'])} | "
            f"{t['wins']} / {t['ties']} / {t['losses']} |"
        )
    lines += ["", "## Time", "", "| Stage | Seconds |", "|---|---|"]
    for k, v in r["timings_s"].items():
        lines.append(f"| {k} | {v:.1f} |")
    lines += ["", "| Per-query cost | ms |", "|---|---|"]
    for k, v in r["latency_ms_per_query"].items():
        lines.append(f"| {k} | {v:.2f} |")
    lines += ["", "## Versions", ""]
    lines += [f"- {k} {v}" for k, v in r["versions"].items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
