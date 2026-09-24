# Results

- Run date (UTC): 2026-09-24T07:29:13Z
- Dataset: BEIR FiQA-2018, test split, 648 queries, 57,638 documents, 1,706 relevance judgements
- Data source: Hugging Face mirror BeIR/fiqa@979c07a7cb5ccc6ca009792241fa1250b98055dd and BeIR/fiqa-qrels@252958f2d646e22cab6d0c72dd3f0d5de6d0655a (primary https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip unreachable)
- Dense model: `BAAI/bge-small-en-v1.5` @ `5c38ec7c405e`
- Reranker (experiment): `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d25c44`
- BM25: bm25s 0.3.11, Lucene variant, k1=1.5, b=0.75
- Fusion: reciprocal rank fusion, k=60
- Device: mps; hardware: Apple M3, 8 GB RAM, macOS 27.2
- Compute time, all stages: 1048.0 s; wall clock 1176.1 s over 4 invocation(s) (the slow stages checkpoint and resume)

## Scores (mean over queries)

| Arm | nDCG@10 | Recall@5 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---|---|---|
| BM25 full-text only | 0.2514 | 0.2454 | 0.3178 | 0.5593 | 0.3075 |
| Dense vectors only (bge-small-en-v1.5) | 0.4035 | 0.3806 | 0.4639 | 0.6963 | 0.4879 |
| Hybrid: RRF of BM25 + dense (k=60) | 0.3651 | 0.3573 | 0.4393 | 0.6931 | 0.4371 |
| Experiment: hybrid + cross-encoder rerank of top 50 | 0.3710 | 0.3784 | 0.4505 | 0.6931 | 0.4413 |

## Paired t-tests (two-sided, per query)

| Comparison | Metric | A | B | A minus B | 95% CI | Change | p (scipy) | p (ranx) | Better / same / worse |
|---|---|---|---|---|---|---|---|---|---|
| rrf vs bm25 | ndcg@10 | 0.3651 | 0.2514 | +0.1137 | [+0.0984, +0.1290] | +45.2% | <0.0001 | <0.0001 | 310 / 292 / 46 |
| rrf vs bm25 | recall@5 | 0.3573 | 0.2454 | +0.1119 | [+0.0906, +0.1332] | +45.6% | <0.0001 | <0.0001 | 168 / 468 / 12 |
| rrf vs bm25 | recall@10 | 0.4393 | 0.3178 | +0.1215 | [+0.0989, +0.1442] | +38.2% | <0.0001 | <0.0001 | 185 / 445 / 18 |
| rrf vs bm25 | recall@100 | 0.6931 | 0.5593 | +0.1338 | [+0.1118, +0.1557] | +23.9% | <0.0001 | <0.0001 | 195 / 436 / 17 |
| rrf vs bm25 | mrr@10 | 0.4371 | 0.3075 | +0.1296 | [+0.1083, +0.1509] | +42.1% | <0.0001 | <0.0001 | 252 / 359 / 37 |
| rrf vs dense | ndcg@10 | 0.3651 | 0.4035 | -0.0383 | [-0.0555, -0.0212] | -9.5% | <0.0001 | <0.0001 | 153 / 291 / 204 |
| rrf vs dense | recall@5 | 0.3573 | 0.3806 | -0.0232 | [-0.0461, -0.0004] | -6.1% | 0.0462 | 0.0462 | 76 / 460 / 112 |
| rrf vs dense | recall@10 | 0.4393 | 0.4639 | -0.0245 | [-0.0461, -0.0029] | -5.3% | 0.0260 | 0.0260 | 73 / 472 / 103 |
| rrf vs dense | recall@100 | 0.6931 | 0.6963 | -0.0032 | [-0.0184, +0.0120] | -0.5% | 0.6812 | 0.6812 | 43 / 549 / 56 |
| rrf vs dense | mrr@10 | 0.4371 | 0.4879 | -0.0508 | [-0.0736, -0.0280] | -10.4% | <0.0001 | <0.0001 | 104 / 396 / 148 |
| rrf_rerank vs rrf | ndcg@10 | 0.3710 | 0.3651 | +0.0059 | [-0.0092, +0.0210] | +1.6% | 0.4429 | 0.4429 | 177 / 301 / 170 |
| rrf_rerank vs rrf | recall@5 | 0.3784 | 0.3573 | +0.0211 | [+0.0003, +0.0419] | +5.9% | 0.0469 | 0.0469 | 89 / 489 / 70 |
| rrf_rerank vs rrf | recall@10 | 0.4505 | 0.4393 | +0.0111 | [-0.0081, +0.0304] | +2.5% | 0.2577 | 0.2577 | 75 / 498 / 75 |
| rrf_rerank vs rrf | recall@100 | 0.6931 | 0.6931 | +0.0000 | [+0.0000, +0.0000] | +0.0% | 1.0000 | n/a | 0 / 648 / 0 |
| rrf_rerank vs rrf | mrr@10 | 0.4413 | 0.4371 | +0.0042 | [-0.0182, +0.0265] | +1.0% | 0.7126 | 0.7126 | 134 / 385 / 129 |
| rrf_rerank vs dense | ndcg@10 | 0.3710 | 0.4035 | -0.0324 | [-0.0502, -0.0147] | -8.0% | 0.0003 | 0.0003 | 155 / 280 / 213 |
| rrf_rerank vs dense | recall@5 | 0.3784 | 0.3806 | -0.0021 | [-0.0234, +0.0191] | -0.6% | 0.8430 | 0.8430 | 83 / 467 / 98 |
| rrf_rerank vs dense | recall@10 | 0.4505 | 0.4639 | -0.0134 | [-0.0348, +0.0080] | -2.9% | 0.2185 | 0.2185 | 74 / 467 / 107 |
| rrf_rerank vs dense | recall@100 | 0.6931 | 0.6963 | -0.0032 | [-0.0184, +0.0120] | -0.5% | 0.6812 | 0.6812 | 43 / 549 / 56 |
| rrf_rerank vs dense | mrr@10 | 0.4413 | 0.4879 | -0.0466 | [-0.0713, -0.0218] | -9.5% | 0.0002 | 0.0002 | 118 / 370 / 160 |
| dense vs bm25 | ndcg@10 | 0.4035 | 0.2514 | +0.1521 | [+0.1284, +0.1758] | +60.5% | <0.0001 | <0.0001 | 317 / 246 / 85 |
| dense vs bm25 | recall@5 | 0.3806 | 0.2454 | +0.1351 | [+0.1074, +0.1629] | +55.1% | <0.0001 | <0.0001 | 216 / 386 / 46 |
| dense vs bm25 | recall@10 | 0.4639 | 0.3178 | +0.1461 | [+0.1171, +0.1751] | +46.0% | <0.0001 | <0.0001 | 230 / 368 / 50 |
| dense vs bm25 | recall@100 | 0.6963 | 0.5593 | +0.1370 | [+0.1100, +0.1639] | +24.5% | <0.0001 | <0.0001 | 223 / 380 / 45 |
| dense vs bm25 | mrr@10 | 0.4879 | 0.3075 | +0.1804 | [+0.1500, +0.2107] | +58.7% | <0.0001 | <0.0001 | 259 / 320 / 69 |

## Time

| Stage | Seconds |
|---|---|
| load_data_s | 0.1 |
| evaluate_s | 2.2 |
| dense_encode_corpus_s | 797.6 |
| bm25_index_s | 2.7 |
| bm25_search_s | 0.3 |
| dense_search_s | 2.8 |
| rrf_fuse_s | 0.0 |
| rerank_s | 242.1 |

| Per-query cost | ms |
|---|---|
| bm25_search | 0.46 |
| dense_query_encode_and_search | 4.36 |
| rrf_fusion_step | 0.07 |
| cross_encoder_rerank_step | 373.63 |

## Versions

- numpy 2.5.3
- scipy 1.18.1
- ranx 0.3.21
- bm25s 0.3.11
- PyStemmer 3.1.0
- torch 2.14.0
- transformers 5.17.0
- sentence-transformers 6.1.0
- pyarrow 25.0.1
