PYPSA RAG BENCHMARK - WINDOWS

1. Install Python 3.11 and Ollama; run: ollama pull qwen2.5-coder:3b
2. Double-click INSTALL_ONCE.bat (only on first use).
3. Extract the PyPSA source repository into a folder (for example PyPSA-master).
4. Double-click START_ONE_CLICK.bat. Browse to your PyPSA-master folder.
5. Choose the questions JSON, starting index and number of questions, or specify indices like 0,2,4.
6. Click RUN ALL 4 ARCHITECTURES. Indexes are built on first run, then reused only if verified.
7. Inspect output/benchmark_runs/run_YYYYMMDD_HHMMSS/comparison.csv, summary.json, and each system/question_N.json.

IMPORTANT: The provided uploaded ZIP contains ONLY ONE exported PyPSA Q&A record at output/hifirepoqa/questions.json.
To benchmark more questions, supply the full HiFiRepoQA dataset JSON (raw entries with llm_analysis.correct_answer or exported entries with reference_answer) using the Browse button. This package cannot invent missing reference answers.

Ranking relevance scores (MRR, Recall@K, nDCG@K) are computed only when verified 'targets' with file and optional line ranges exist in the questions JSON. They are otherwise null, not fabricated. Reference answers are saved for independent/manual answer grading and are NOT supplied to the LLM.

Index building time is tracked separately from per-query retrieval and LLM times. Chunking and embedding indexing are combined in the index-building measurement; the package does not claim separately timed sub-stages for those operations. Ollama reports actual prompt/output tokens where supported.
