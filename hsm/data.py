"""Fetch and load BEIR FiQA-2018: the full corpus and the test split.

Primary source is the BEIR zip on the UKP server. If that host cannot be
reached, the loader falls back to the BEIR authors' Hugging Face mirror
(BeIR/fiqa and BeIR/fiqa-qrels), pinned to a fixed commit. Either way the
files are normalised to the standard BEIR layout under data/fiqa/:

    corpus.jsonl   {"_id", "title", "text"}
    queries.jsonl  {"_id", "text"}
    qrels/test.tsv query-id <tab> corpus-id <tab> score
"""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

BEIR_ZIP_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

HF_FIQA_REV = "979c07a7cb5ccc6ca009792241fa1250b98055dd"
HF_QRELS_REV = "252958f2d646e22cab6d0c72dd3f0d5de6d0655a"
HF_CORPUS_URL = (
    f"https://huggingface.co/datasets/BeIR/fiqa/resolve/{HF_FIQA_REV}"
    "/corpus/corpus-00000-of-00001.parquet"
)
HF_QUERIES_URL = (
    f"https://huggingface.co/datasets/BeIR/fiqa/resolve/{HF_FIQA_REV}"
    "/queries/queries-00000-of-00001.parquet"
)
HF_QRELS_TEST_URL = (
    f"https://huggingface.co/datasets/BeIR/fiqa-qrels/resolve/{HF_QRELS_REV}/test.tsv"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def _from_beir_zip(data_dir: Path) -> dict:
    zpath = data_dir / "fiqa.zip"
    if not zpath.exists():
        _download(BEIR_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data_dir)
    return {"source": BEIR_ZIP_URL, "files": {"fiqa.zip": _sha256(zpath)}}


def _from_hf_mirror(data_dir: Path) -> dict:
    import pyarrow.parquet as pq

    raw = data_dir / "hf-mirror"
    files = {
        "corpus.parquet": HF_CORPUS_URL,
        "queries.parquet": HF_QUERIES_URL,
        "qrels-test.tsv": HF_QRELS_TEST_URL,
    }
    for name, url in files.items():
        if not (raw / name).exists():
            _download(url, raw / name)

    out = data_dir / "fiqa"
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    corpus = pq.read_table(raw / "corpus.parquet").to_pylist()
    with open(out / "corpus.jsonl", "w") as f:
        for row in corpus:
            rec = {"_id": str(row["_id"]), "title": row.get("title") or "", "text": row["text"]}
            f.write(json.dumps(rec) + "\n")
    queries = pq.read_table(raw / "queries.parquet").to_pylist()
    with open(out / "queries.jsonl", "w") as f:
        for row in queries:
            f.write(json.dumps({"_id": str(row["_id"]), "text": row["text"]}) + "\n")
    (out / "qrels" / "test.tsv").write_bytes((raw / "qrels-test.tsv").read_bytes())
    return {
        "source": "Hugging Face mirror BeIR/fiqa@" + HF_FIQA_REV
        + " and BeIR/fiqa-qrels@" + HF_QRELS_REV
        + f" (primary {BEIR_ZIP_URL} unreachable)",
        "files": {name: _sha256(raw / name) for name in files},
    }


def ensure_fiqa(data_dir: Path) -> dict:
    """Make sure data/fiqa exists; return provenance (source URL, sha256s)."""
    data_dir = Path(data_dir)
    prov_path = data_dir / "provenance.json"
    fiqa = data_dir / "fiqa"
    if (fiqa / "corpus.jsonl").exists() and (fiqa / "qrels" / "test.tsv").exists():
        if prov_path.exists():
            return json.loads(prov_path.read_text())
    try:
        prov = _from_beir_zip(data_dir)
    except Exception as exc:  # noqa: BLE001 - any network or zip failure falls back
        print(f"[data] BEIR zip unavailable ({exc!r}); using the Hugging Face mirror")
        prov = _from_hf_mirror(data_dir)
    prov_path.write_text(json.dumps(prov, indent=2))
    return prov


def load_fiqa(data_dir: Path, split: str = "test"):
    """Return (corpus, queries, qrels).

    corpus: dict doc_id -> text (title and body joined, BEIR convention)
    queries: dict query_id -> text, restricted to queries judged in `split`
    qrels: dict query_id -> {doc_id: relevance}
    """
    fiqa = Path(data_dir) / "fiqa"
    corpus: dict[str, str] = {}
    with open(fiqa / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            corpus[str(d["_id"])] = (title + " " + d["text"]).strip() if title else d["text"]

    qrels: dict[str, dict[str, int]] = {}
    with open(fiqa / "qrels" / f"{split}.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, did, score in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(score)

    queries: dict[str, str] = {}
    with open(fiqa / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if str(q["_id"]) in qrels:
                queries[str(q["_id"])] = q["text"]
    return corpus, queries, qrels
