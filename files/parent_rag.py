"""Phase 5: retrieve small AST chunks, then expand to their parent class or module.

The index still uses Phase 4 hybrid search. Expansion occurs AFTER retrieval.
No retrieved file content is executed.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
from hybrid_rag import build_hybrid_index, search_hybrid
from baseline_rag import read_python_files, answer, DEFAULT_EMBED_MODEL


def make_parent_records(repo: str):
    """Create per-file parent source spans and associate AST children by file/line."""
    records = {}
    for filename, source in read_python_files(repo):
        lines = source.splitlines()
        classes = []
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append({'name': node.name, 'start_line': min([node.lineno] + [d.lineno for d in node.decorator_list]), 'end_line': node.end_lineno})
        records[filename] = {'lines': lines, 'classes': classes}
    return records


def build_parent_index(repo: str, output: str = 'output/parent', embed_model: str = DEFAULT_EMBED_MODEL):
    count = build_hybrid_index(repo, output, embed_model)
    folder = Path(output)
    (folder/'parents.json').write_text(json.dumps(make_parent_records(repo), indent=2), encoding='utf-8')
    settings_file = folder/'settings.json'
    settings = json.loads(settings_file.read_text(encoding='utf-8'))
    settings['parent_retrieval'] = 'containing_class_or_module_window'
    settings_file.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    return count


def expand_match(match: dict, parents: dict, max_parent_lines: int = 100) -> dict:
    """Choose enclosing top-level class for a method, otherwise a file window.

    A bounded window includes the matched chunk and expands on both sides.
    If a matched chunk itself exceeds the budget, preserve the entire match;
    the budget is soft in this case.
    """
    if max_parent_lines <= 0:
        raise ValueError('max_parent_lines must be positive')
    file = match['file']
    if file not in parents:
        raise KeyError(f'No parent source for {file}')
    record = parents[file]
    lines = record['lines']
    start, end = int(match['start_line']), int(match['end_line'])
    if not 1 <= start <= end <= len(lines):
        raise ValueError('Invalid match line range')
    containing = [c for c in record['classes'] if c['start_line'] <= start and end <= c['end_line']]
    if containing:
        parent = min(containing, key=lambda c: c['end_line']-c['start_line'])
        lo, hi = parent['start_line'], parent['end_line']
        parent_kind, parent_name = 'class', parent['name']
    else:
        lo, hi = 1, len(lines)
        parent_kind, parent_name = 'module', file
    # Window always covers the retrieved match, and stays within its parent.
    budget = max(max_parent_lines, end-start+1)
    extra = max(0, budget-(end-start+1))
    left = min(start-lo, extra//2)
    right = min(hi-end, extra-left)
    left += min(start-lo-left, extra-left-right)
    lo_out, hi_out = start-left, end+right
    return {**match, 'matched_code': match['code'], 'matched_start_line': start,
            'matched_end_line': end, 'start_line': lo_out, 'end_line': hi_out,
            'code': '\n'.join(lines[lo_out-1:hi_out]),
            'parent_kind': parent_kind, 'parent_name': parent_name}


def search_with_parents(query: str, output: str = 'output/parent', top_k: int = 3,
                        candidate_k: int = 20, rrf_k: int = 60, mode: str = 'hybrid',
                        max_parent_lines: int = 100):
    folder = Path(output)
    matches = search_hybrid(query, output, top_k, candidate_k, rrf_k, mode)
    parents = json.loads((folder/'parents.json').read_text(encoding='utf-8'))
    return [expand_match(m, parents, max_parent_lines) for m in matches]


def answer_with_parents(query: str, matches: list[dict], llm: str):
    """Use the existing Ollama prompt and show which lines originally matched."""
    return answer(query, matches, llm)
