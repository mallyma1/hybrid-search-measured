"""The retrieval arms: BM25 full-text, dense vectors, cross-encoder rerank.

The two model-heavy stages (encoding the corpus, cross-encoder scoring) write
checkpoints in chunks under data/cache/, with the compute seconds of every
chunk. On a small laptop a run can stop at a time budget and resume later
without losing work, and the reported stage time is the sum of the chunk
times, not a guess.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

Run = dict[str, dict[str, float]]  # query_id -> {doc_id: score}

# Pinned model revisions (Hugging Face commit hashes) so a rerun uses the same weights.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# BGE's recommended instruction for short queries against longer passages.
DENSE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Published as cross-encoder/ms-marco-MiniLM-L-6-v2; the hub now serves it under this id.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

BM25_K1 = 1.5
BM25_B = 0.75

ENCODE_CHUNK = 2048      # documents per checkpoint
ENCODE_BATCH = 16
RERANK_CHUNK = 40        # queries per checkpoint
RERANK_BATCH = 16


class Incomplete(Exception):
    """Raised when a resumable stage hits its deadline; rerun to continue."""


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _free(device: str) -> None:
    gc.collect()
    if device == "mps":
        import torch

        torch.mps.empty_cache()


def _progress(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_progress(path: Path, prog: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog, indent=1))
    tmp.replace(path)


# --------------------------------------------------------------------------- BM25
def bm25_search(doc_ids, doc_texts, qids, qtexts, top_k: int = 100):
    """BM25 with bm25s (Lucene variant, k1=1.5, b=0.75), English stopwords and
    the Snowball English stemmer. Documents scoring 0 (no query term matched)
    are dropped rather than returned as filler."""
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    t = time.perf_counter()
    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B, method="lucene")
    retriever.index(corpus_tokens, show_progress=False)
    del corpus_tokens
    index_s = time.perf_counter() - t

    t = time.perf_counter()
    q_tokens = bm25s.tokenize(qtexts, stopwords="en", stemmer=stemmer, show_progress=False)
    idx, scores = retriever.retrieve(q_tokens, k=top_k, show_progress=False)
    run: Run = {}
    for qi, qid in enumerate(qids):
        run[qid] = {
            doc_ids[int(d)]: float(s) for d, s in zip(idx[qi], scores[qi]) if float(s) > 0.0
        }
    search_s = time.perf_counter() - t
    del retriever
    gc.collect()
    return run, {"index_s": index_s, "search_s": search_s}


# -------------------------------------------------------------------------- dense
def _load_dense(device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DENSE_MODEL, revision=DENSE_REVISION, device=device)


def encode_corpus(doc_texts, device: str, cache_dir: Path, deadline: float | None = None):
    """L2-normalised corpus vectors, encoded in length-sorted chunks (less
    padding, bounded memory) with a checkpoint per chunk. Returns
    (vectors, info). Raises Incomplete if the deadline passes first."""
    work = cache_dir / f"bge-small-{DENSE_REVISION[:8]}-{len(doc_texts)}"
    final = work / "corpus.npy"
    prog_path = work / "progress.json"
    work.mkdir(parents=True, exist_ok=True)
    prog = _progress(prog_path)
    n_chunks = -(-len(doc_texts) // ENCODE_CHUNK)
    if final.exists() and len(prog) == n_chunks:
        return np.load(final), {"encode_corpus_s": sum(prog.values()), "chunks": n_chunks}

    order = np.argsort([len(t) for t in doc_texts], kind="stable")
    model = _load_dense(device)
    for c in range(n_chunks):
        key = f"{c:04d}"
        if key in prog and (work / f"chunk_{key}.npy").exists():
            continue
        if deadline is not None and time.time() > deadline:
            del model
            _free(device)
            raise Incomplete(f"corpus encoding: {len(prog)}/{n_chunks} chunks done")
        t = time.perf_counter()
        sel = order[c * ENCODE_CHUNK : (c + 1) * ENCODE_CHUNK]
        vecs = model.encode([doc_texts[i] for i in sel], batch_size=ENCODE_BATCH,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
        np.save(work / f"chunk_{key}.npy", vecs)
        prog[key] = time.perf_counter() - t
        _save_progress(prog_path, prog)
        print(f"[dense] chunk {c + 1}/{n_chunks} ({len(sel)} docs) {prog[key]:.1f}s", flush=True)
        _free(device)
    max_len = int(model.max_seq_length)
    del model
    _free(device)

    dim = np.load(work / "chunk_0000.npy").shape[1]
    out = np.empty((len(doc_texts), dim), dtype=np.float32)
    for c in range(n_chunks):
        out[order[c * ENCODE_CHUNK : (c + 1) * ENCODE_CHUNK]] = np.load(work / f"chunk_{c:04d}.npy")
    np.save(final, out)
    return out, {"encode_corpus_s": sum(prog.values()), "chunks": n_chunks, "max_seq_length": max_len}


def dense_search(doc_ids, doc_emb, qids, qtexts, device: str, top_k: int = 100):
    """Cosine similarity (dot product of unit vectors), exact numpy search."""
    model = _load_dense(device)
    t = time.perf_counter()
    q_emb = model.encode(qtexts, batch_size=ENCODE_BATCH, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False,
                         prompt=DENSE_QUERY_INSTRUCTION).astype(np.float32)
    sims = q_emb @ doc_emb.T
    top = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
    run: Run = {}
    for qi, qid in enumerate(qids):
        cand = top[qi]
        cand = cand[np.argsort(-sims[qi, cand], kind="stable")]
        run[qid] = {doc_ids[int(d)]: float(sims[qi, d]) for d in cand}
    search_s = time.perf_counter() - t
    info = {"search_s": search_s, "dim": int(doc_emb.shape[1]),
            "max_seq_length": int(model.max_seq_length)}
    del model, sims
    _free(device)
    return run, info


# ------------------------------------------------------------------------- rerank
def cross_encoder_rerank(base_run: Run, corpus: dict, queries: dict, device: str,
                         cache_dir: Path, depth: int = 50, deadline: float | None = None):
    """Rescore the top `depth` documents of `base_run` with a cross-encoder.
    Documents below `depth` keep their original order underneath the reranked
    block, so recall@100 is unchanged by construction."""
    from sentence_transformers import CrossEncoder

    qids = sorted(base_run)
    ordered = {q: sorted(base_run[q], key=lambda d: (-base_run[q][d], d)) for q in qids}
    sig = hashlib.sha256(
        json.dumps([[q, ordered[q][:depth]] for q in qids]).encode()
    ).hexdigest()[:12]
    work = cache_dir / f"rerank-{RERANK_REVISION[:8]}-d{depth}-{sig}"
    work.mkdir(parents=True, exist_ok=True)
    prog_path = work / "progress.json"
    prog = _progress(prog_path)
    n_chunks = -(-len(qids) // RERANK_CHUNK)

    model = None
    for c in range(n_chunks):
        key = f"{c:04d}"
        if key in prog and (work / f"part_{key}.json").exists():
            continue
        if deadline is not None and time.time() > deadline:
            del model
            _free(device)
            raise Incomplete(f"rerank: {len(prog)}/{n_chunks} chunks done")
        if model is None:
            model = CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, device=device,
                                 max_length=512)
        t = time.perf_counter()
        part_q = qids[c * RERANK_CHUNK : (c + 1) * RERANK_CHUNK]
        pairs, owners = [], []
        for q in part_q:
            for d in ordered[q][:depth]:
                pairs.append((queries[q], corpus[d]))
                owners.append((q, d))
        scores = np.asarray(model.predict(pairs, batch_size=RERANK_BATCH,
                                          show_progress_bar=False), dtype=np.float64).reshape(-1)
        part: dict[str, dict[str, float]] = {q: {} for q in part_q}
        for (q, d), s in zip(owners, scores):
            part[q][d] = float(s)
        (work / f"part_{key}.json").write_text(json.dumps(part))
        prog[key] = time.perf_counter() - t
        _save_progress(prog_path, prog)
        print(f"[rerank] chunk {c + 1}/{n_chunks} ({len(pairs)} pairs) {prog[key]:.1f}s", flush=True)
        _free(device)
    if model is not None:
        del model
        _free(device)

    head_scores: dict[str, dict[str, float]] = {}
    for c in range(n_chunks):
        head_scores.update(json.loads((work / f"part_{c:04d}.json").read_text()))
    run: Run = {}
    pairs_scored = 0
    for q in qids:
        head = head_scores[q]
        pairs_scored += len(head)
        floor = min(head.values()) if head else 0.0
        out = dict(head)
        for i, d in enumerate(ordered[q][depth:], start=1):
            out[d] = floor - i  # strictly below every reranked score, fused order kept
        run[q] = out
    return run, {"rerank_s": sum(prog.values()), "pairs_scored": pairs_scored, "chunks": n_chunks}
