"""HiFiRepoQA integration: export real Q&A and evaluate answers against references.

Does not use benchmark answers or rubrics in the prompt sent to the model.
"""
import argparse
import csv
import json
from pathlib import Path
from evaluation import SYSTEMS, search_system
from baseline_rag import answer, DEFAULT_LLM

ROOT = Path(__file__).parent
DATA = ROOT / 'evaluation_data' / 'hifirepoqa'

def load_dataset():
    records = []
    for path in sorted(DATA.glob('*.json')):
        for q in json.loads(path.read_text(encoding='utf-8')):
            a = q.get('llm_analysis') or {}
            if q.get('title') and a.get('correct_answer'):
                records.append({'id': q['question_id'], 'repo': q['repo'], 'language': q.get('language'),
                    'question': q['title'] + ('\n' + q['body'] if q.get('body') else ''),
                    'reference_answer': a['correct_answer'], 'rubric': a.get('scoring_rubric', []),
                    'url': q.get('url', '')})
    return records

def select(rows, repo, limit):
    rows = [r for r in rows if not repo or r['repo'].lower() == repo.lower()]
    return rows[:limit] if limit else rows

def main():
    p = argparse.ArgumentParser(description='HiFiRepoQA question-answer benchmark for EdgeCode')
    sub = p.add_subparsers(dest='command', required=True)
    l = sub.add_parser('list', help='Show repositories and question counts')
    e = sub.add_parser('export', help='Export question-answer records to JSON')
    r = sub.add_parser('run', help='Run one repository against indexed RAG systems')
    for s in (e,r):
        s.add_argument('--repo', required=True, help='Exact GitHub owner/repository')
        s.add_argument('--limit', type=int, default=10)
    e.add_argument('--output', default='output/hifirepoqa/questions.json')
    r.add_argument('--index-root', default='output/evaluation_indexes')
    r.add_argument('--output', default='output/hifirepoqa/answers.json')
    r.add_argument('--systems', nargs='+', choices=SYSTEMS, default=list(SYSTEMS))
    r.add_argument('--top-k', type=int, default=3)
    r.add_argument('--llm', default=DEFAULT_LLM)
    args = p.parse_args()
    rows = load_dataset()
    if args.command == 'list':
        from collections import Counter
        for repo, count in sorted(Counter(x['repo'] for x in rows).items(), key=lambda x:(-x[1],x[0])):
            print(f'{count:3}  {repo}')
        print(f'Total: {len(rows)} questions')
        return
    if args.limit < 0:
        p.error('--limit cannot be negative')
    rows = select(rows, args.repo, args.limit)
    if not rows:
        p.error('No questions for that repository. Run list to see available repositories.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'export':
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Exported {len(rows)} Q&A records to {output}')
        return
    print('IMPORTANT: Ensure all four indexes were built from the matching repository checkout.')
    results=[]
    for system in args.systems:
        for q in rows:
            matches=search_system(system,q['question'],args.index_root,args.top_k,20,60,100)
            prediction=answer(q['question'],matches,args.llm)
            results.append({**q,'system':system,'prediction':prediction,
                'retrieved':[{'file':m['file'],'start_line':m['start_line'],'end_line':m['end_line']} for m in matches]})
            print(f'{system}: completed question {q["id"]}')
            output.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Saved {len(results)} predictions and reference answers to {output}')
    print('NOTE: No automatic answer grade is claimed; use provided scoring rubrics for review.')

if __name__=='__main__':
    main()
