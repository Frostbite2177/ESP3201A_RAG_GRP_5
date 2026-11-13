"""Phase 3: Python AST-aware code chunks with the same dense search as Phase 2."""
from __future__ import annotations
import ast
import json
from pathlib import Path
from baseline_rag import read_python_files, DEFAULT_EMBED_MODEL, search, answer


def ast_chunks(repo: str | Path):
    """Split at module/class/function boundaries, retaining file/line provenance.

    Top-level imports/constants are grouped into module chunks. Class method
    definitions become separate chunks; class header and non-method statements
    are retained as a class-context chunk. Nested functions remain inside their
    enclosing function. No size cap is imposed on a single function yet.
    """
    result = []

    def add(file, lines, start, end, kind, name, parent=None):
        if start > end:
            return
        body = "\n".join(lines[start-1:end]).strip()
        if body:
            result.append({"id": len(result), "file": file, "start_line": start,
                           "end_line": end, "code": body, "kind": kind,
                           "name": name, "parent": parent})

    def start_line(node):
        decorators = getattr(node, "decorator_list", [])
        return min([node.lineno] + [d.lineno for d in decorators])

    for file, code in read_python_files(repo):
        lines = code.splitlines()
        try:
            tree = ast.parse(code, filename=file)
        except SyntaxError as exc:
            print(f"Skipping syntax-invalid file {file}: {exc}")
            continue
        cursor = 1
        for node in tree.body:
            start, end = start_line(node), node.end_lineno
            if cursor < start:
                add(file, lines, cursor, start-1, "module_context", "<module>")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(file, lines, start, end, "function", node.name)
            elif isinstance(node, ast.ClassDef):
                method_ranges = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ms, me = start_line(child), child.end_lineno
                        method_ranges.append((ms, me, child))
                if not method_ranges:
                    add(file, lines, start, end, "class", node.name)
                else:
                    cursor_class = start
                    for ms, me, method in method_ranges:
                        add(file, lines, cursor_class, ms-1, "class_context", node.name, node.name)
                        add(file, lines, ms, me, "method", f"{node.name}.{method.name}", node.name)
                        cursor_class = me+1
                    add(file, lines, cursor_class, end, "class_context", node.name, node.name)
            else:
                add(file, lines, start, end, "module_context", "<module>")
            cursor = end+1
        if cursor <= len(lines):
            add(file, lines, cursor, len(lines), "module_context", "<module>")
    return result


def build_ast_index(repo: str, output: str = "output/ast", embed_model: str = DEFAULT_EMBED_MODEL):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    chunks = ast_chunks(repo)
    if not chunks:
        raise ValueError("No valid Python code chunks found")
    model = SentenceTransformer(embed_model)
    vectors = np.asarray(model.encode([c["code"] for c in chunks],
                                      show_progress_bar=True), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(folder / "vectors.faiss"))
    (folder / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps({"embedding_model": embed_model,
        "chunking": "python_ast", "no_max_function_size": True}, indent=2), encoding="utf-8")
    return len(chunks)
