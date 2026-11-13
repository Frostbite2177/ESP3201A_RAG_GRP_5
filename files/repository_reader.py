"""Phase 1: extract function and class chunks from local Python files."""
import ast
from pathlib import Path


def extract_chunks(repository_path):
    chunks = []
    root = Path(repository_path)
    for file in sorted(root.rglob('*.py')):
        if any(part.startswith('.') or part in {'venv', '.venv', '__pycache__'} for part in file.relative_to(root).parts):
            continue
        code = file.read_text(encoding='utf-8-sig')
        tree = ast.parse(code, filename=str(file))
        lines = code.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list]) if hasattr(node, 'decorator_list') else node.lineno
                chunks.append({
                    'file': str(file),
                    'name': node.name,
                    'type': type(node).__name__,
                    'start_line': start,
                    'end_line': node.end_lineno,
                    'code': '\n'.join(lines[start - 1:node.end_lineno]),
                })
    return chunks
