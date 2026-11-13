# EdgeCode Assistant: Code-Aware Multi-Representation RAG for Local Repositories

National University of Singapore course project.
Team: Visagan Sharavana Kumar (A0288566H), Vincent Ang Jun Wei (A0286413A), Zhang Zixiao (A0272525E).

## Research question
Does AST-based chunking combined with hybrid BM25 + dense retrieval improve code-snippet relevance and answer accuracy compared with fixed-size chunking?

## Four retrieval configurations
| Config | File | Description |
|---|---|---|
| Baseline | `files/baseline_rag.py` | 30-line windows with 5-line overlap, dense retrieval |
| AST | `files/ast_rag.py` | Python `ast` chunks (functions, methods, classes, module context), dense retrieval |
| AST + Hybrid | `files/hybrid_rag.py` | AST chunks, BM25 + dense fused with reciprocal rank fusion (k = 60) |
| AST + Parent | `files/parent_rag.py` | Hybrid hits widened to the enclosing class/file (up to 100 lines) |

Shared components: `all-MiniLM-L6-v2` embeddings, FAISS `IndexFlatIP`, top-3 retrieval, and `qwen2.5-coder:3b` served locally by Ollama (temperature 0).

## Repository layout
```
files/
  benchmark_oneclick.py   runs all four configurations on a question set
  one_click_gui.py        Tkinter front end for the benchmark
  evaluation.py           retrieval metrics (MRR, Recall@K, nDCG@K)
  evaluate_timed.py       timed evaluation runner
  hifirepoqa_cli.py       HiFiRepoQA question loader/exporter
  repository_reader.py    repository file reader and chunk extraction
hifirepoqa_qns/           HiFiRepoQA question banks for the five Python repositories used
```

## Setup
1. Install Python 3.11+ and [Ollama](https://ollama.com), then run `ollama pull qwen2.5-coder:3b`.
2. `pip install sentence-transformers faiss-cpu numpy ollama rank-bm25`
3. Download the target repository source (e.g., openfold, procrastinate, linkding, thingsboard-gateway, agentscope).
4. `python files/one_click_gui.py`, choose the repository folder and the matching questions JSON, then run all four architectures.

Results are written to `output/benchmark_runs/run_YYYYMMDD_HHMMSS/` (`comparison.csv`, `summary.json`, and one JSON per question per configuration).

## Dataset
Questions come from HiFiRepoQA: https://github.com/HiFiRepoQA/HiFiRepoQA/tree/main/datasets
Only repositories whose primary language is Python were selected, because the AST chunker parses Python only.
