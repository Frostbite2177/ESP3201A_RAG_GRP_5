"""Phase 6 retrieval evaluation: annotated code locations, no LLM judge required."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path

SYSTEMS = ('baseline', 'ast', 'hybrid', 'parent')


def relevant(match: dict, targets: list[dict]) -> bool:
    """A hit must overlap an annotated relevant line range in the same file.

    For parent retrieval, evaluate the *original matched child* (not expanded
    parent), preventing inflated relevance from larger returned context.
    """
    lo = match.get('matched_start_line', match['start_line'])
    hi = match.get('matched_end_line', match['end_line'])
    return any(match['file'] == t['file'] and lo <= t['end_line'] and hi >= t['start_line'] for t in targets)


def retrieval_metrics(matches: list[dict], targets: list[dict], k: int) -> dict:
    """Precision@k, Hit@k and reciprocal rank for annotated relevant locations.

    Recall@k counts covered *target ranges*, not retrieved chunks.
    """
    if k <= 0 or not targets:
        raise ValueError('k must be positive and targets nonempty')
    selected = matches[:k]
    flags = [relevant(m, targets) for m in selected]
    covered = sum(any(relevant(m, [t]) for m in selected) for t in targets)
    first = next((i for i, ok in enumerate(flags, 1) if ok), None)
    return {'hit_at_k': int(any(flags)), 'precision_at_k': sum(flags)/k,
            'recall_at_k': covered/len(targets), 'reciprocal_rank': 1/first if first else 0.0}


def search_system(system: str, query: str, index_root: str, k: int, candidate_k: int,
                  rrf_k: int, max_parent_lines: int):
    folder = str(Path(index_root)/system)
    if system == 'baseline':
        from baseline_rag import search
        return search(query, folder, k)
    if system == 'ast':
        from baseline_rag import search
        return search(query, folder, k)
    if system == 'hybrid':
        from hybrid_rag import search_hybrid
        return search_hybrid(query, folder, k, candidate_k, rrf_k)
    if system == 'parent':
        from parent_rag import search_with_parents
        return search_with_parents(query, folder, k, candidate_k, rrf_k, 'hybrid', max_parent_lines)
    raise ValueError(f'Unknown system: {system}')


def load_questions(path: str) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(questions, list) or not questions:
        raise ValueError('Expected a nonempty JSON array')
    for q in questions:
        if not q.get('id') or not q.get('question') or not q.get('targets'):
            raise ValueError('Each question needs id, question, and nonempty targets')
        for t in q['targets']:
            if not t.get('file') or not 1 <= t.get('start_line', 0) <= t.get('end_line', 0):
                raise ValueError('Targets need file and valid line range')
    return questions


def evaluate(questions, systems=SYSTEMS, index_root='output', k=3,
             candidate_k=20, rrf_k=60, max_parent_lines=100, search_fn=search_system):
    rows = []
    for system in systems:
        if system not in SYSTEMS:
            raise ValueError(f'Unknown system: {system}')
        for q in questions:
            start = time.perf_counter()
            matches = search_fn(system, q['question'], index_root, k, candidate_k, rrf_k, max_parent_lines)
            elapsed = (time.perf_counter()-start)*1000
            metrics = retrieval_metrics(matches, q['targets'], k)
            rows.append({'system': system, 'question_id': q['id'], 'question': q['question'],
                         'latency_ms': round(elapsed, 2), **metrics,
                         'retrieved': [{'file': m['file'],
                                        'start_line': m.get('matched_start_line', m['start_line']),
                                        'end_line': m.get('matched_end_line', m['end_line']),
                                        'relevant': relevant(m, q['targets'])} for m in matches[:k]]})
    return rows


def summarize(rows):
    result = []
    for system in dict.fromkeys(r['system'] for r in rows):
        subset = [r for r in rows if r['system'] == system]
        result.append({'system': system, 'questions': len(subset), **{
            key: round(sum(r[key] for r in subset)/len(subset), 4)
            for key in ('hit_at_k', 'precision_at_k', 'recall_at_k', 'reciprocal_rank', 'latency_ms')
        }})
    return result


def write_results(rows, folder):
    path = Path(folder); path.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (path/'retrieval_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (path/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (path/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return summary
