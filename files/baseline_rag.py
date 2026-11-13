"""Phase 2 baseline: fixed-size code chunks -> embeddings -> FAISS -> Ollama."""
from __future__ import annotations
import json
from pathlib import Path

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "qwen2.5-coder:3b"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}


def read_python_files(root: str | Path):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository directory not found: {root}")
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            yield path.relative_to(root).as_posix(), path.read_text(encoding="utf-8-sig")
        except (UnicodeError, OSError) as exc:
            print(f"Skipping {path}: {exc}")


def fixed_size_chunks(root: str | Path, chunk_lines: int = 30, overlap_lines: int = 5):
    """Line-based fixed-size baseline. No AST information is used."""
    if chunk_lines <= 0 or overlap_lines < 0 or overlap_lines >= chunk_lines:
        raise ValueError("Require chunk_lines > 0 and 0 <= overlap_lines < chunk_lines")
    chunks = []
    for relative_file, code in read_python_files(root):
        lines = code.splitlines()
        step = chunk_lines - overlap_lines
        for start in range(0, len(lines), step):
            end = min(start + chunk_lines, len(lines))
            body = "\n".join(lines[start:end]).strip()
            if body:
                chunks.append({"id": len(chunks), "file": relative_file,
                               "start_line": start + 1, "end_line": end,
                               "code": body})
            if end == len(lines):
                break
    return chunks


def build_index(repo: str, output: str, embed_model: str = DEFAULT_EMBED_MODEL,
                chunk_lines: int = 30, overlap_lines: int = 5):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    chunks = fixed_size_chunks(repo, chunk_lines, overlap_lines)
    if not chunks:
        raise ValueError("No nonempty Python code chunks found")
    model = SentenceTransformer(embed_model)
    vectors = np.asarray(model.encode([c["code"] for c in chunks],
                                      show_progress_bar=True), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity after normalization
    index.add(vectors)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(folder / "vectors.faiss"))
    (folder / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps({"embedding_model": embed_model,
       "chunk_lines": chunk_lines, "overlap_lines": overlap_lines}, indent=2), encoding="utf-8")
    return len(chunks)


def search(query: str, output: str, top_k: int = 3):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    folder = Path(output)
    chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(folder / "vectors.faiss"))
    model = SentenceTransformer(settings["embedding_model"])
    vector = np.asarray(model.encode([query]), dtype="float32")
    faiss.normalize_L2(vector)
    scores, ids = index.search(vector, min(top_k, len(chunks)))
    return [{**chunks[int(idx)], "score": float(score)}
            for idx, score in zip(ids[0], scores[0]) if idx >= 0]


def answer(query: str, matches: list[dict], llm: str = DEFAULT_LLM):
    import ollama
    context = "\n\n".join(
        f"SOURCE: {c['file']} lines {c['start_line']}-{c['end_line']}\n```python\n{c['code']}\n```"
        for c in matches
    )
    messages = [
        {"role": "system", "content": "You are a code Q&A assistant. Answer only using the provided repository snippets. Cite file names and line ranges. If the snippets are insufficient, say so. Do not invent code."},
        {"role": "user", "content": f"Question: {query}\n\nRetrieved repository snippets:\n{context}"},
    ]
    response = ollama.chat(model=llm, messages=messages)
    return response["message"]["content"]
