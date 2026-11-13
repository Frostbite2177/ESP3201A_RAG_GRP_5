"""Five-stage Phase 6 benchmark; original evaluator and retrieval modules unchanged."""
import csv
import json
import time
from pathlib import Path
from evaluation import SYSTEMS, load_questions, retrieval_metrics, relevant
from hybrid_rag import tokenize, reciprocal_rank_fusion
from parent_rag import expand_match


def now():
    return time.perf_counter()


def timed_search(system, question, folder, top_k=3, candidate_k=20, rrf_k=60, max_parent_lines=100):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    folder = Path(folder)
    times = {}
    t = now()
    chunks = json.loads((folder/'chunks.json').read_text(encoding='utf-8'))
    settings = json.loads((folder/'settings.json').read_text(encoding='utf-8'))
    index = faiss.read_index(str(folder/'vectors.faiss'))
    tokens = json.loads((folder/'tokens.json').read_text(encoding='utf-8')) if system in ('hybrid','parent') else None
    bm25 = BM25Okapi(tokens) if tokens is not None else None
    parents = json.loads((folder/'parents.json').read_text(encoding='utf-8')) if system == 'parent' else None
    model = SentenceTransformer(settings['embedding_model'])
    times['initialization_s'] = now()-t

    t=now()
    vector = np.asarray(model.encode([question]), dtype='float32')
    faiss.normalize_L2(vector)
    times['embedding_s'] = now()-t

    t=now()
    limit = min(len(chunks), max(top_k, candidate_k)) if bm25 is not None else min(top_k,len(chunks))
    dense_scores, dense_ids = index.search(vector,limit)
    dense_ranking = [int(i) for i in dense_ids[0] if i >= 0]
    bm25_ranking=[]
    if bm25 is not None:
        scores=bm25.get_scores(tokenize(question))
        bm25_ranking=sorted((i for i in range(len(chunks)) if scores[i]>0),key=lambda i:(-float(scores[i]),i))[:limit]
    times['retrieval_s'] = now()-t

    t=now()
    if bm25 is None:
        matches=[{**chunks[idx], 'score':float(score)} for idx,score in zip(dense_ids[0],dense_scores[0]) if idx>=0][:top_k]
    else:
        fused=reciprocal_rank_fusion([bm25_ranking,dense_ranking],rrf_k)[:top_k]
        matches=[{**chunks[idx], 'score':score,'retrieval_mode':'hybrid'} for idx,score in fused]
    if parents is not None:
        matches=[expand_match(m,parents,max_parent_lines) for m in matches]
    times['ranking_context_s']=now()-t
    return matches,times


def llm_answer(question,matches,model_name):
    import ollama
    context='\n\n'.join(f"SOURCE: {m['file']} lines {m['start_line']}-{m['end_line']}\n```python\n{m['code']}\n```" for m in matches)
    messages=[{'role':'system','content':'You are a code Q&A assistant. Answer only using the provided repository snippets. Cite file names and line ranges. If insufficient, say so. Do not invent code.'}, {'role':'user','content':f'Question: {question}\n\nRetrieved repository snippets:\n{context}'}]
    t=now()
    response=ollama.chat(model=model_name,messages=messages,options={'temperature':0})
    elapsed=now()-t
    def field(key,default=None):
        return response.get(key,default) if isinstance(response,dict) else getattr(response,key,default)
    msg=field('message',{})
    answer=msg.get('content','') if isinstance(msg,dict) else getattr(msg,'content','')
    inp=field('prompt_eval_count');out=field('eval_count')
    return answer,elapsed,inp,out


def run(questions,index_root,results_dir,top_k=3,candidate_k=20,rrf_k=60,max_parent_lines=100,run_llm=True,llm_model='qwen2.5-coder:3b'):
    out=Path(results_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    answer_dir = out / 'llm_answers'
    answer_dir.mkdir(parents=True, exist_ok=True)
    for system in SYSTEMS:
        for q in questions:
            matches,times=timed_search(system,q['question'],Path(index_root)/system,top_k,candidate_k,rrf_k,max_parent_lines)
            metrics=retrieval_metrics(matches,q['targets'],top_k)
            answer='';inp=None;output=None
            if run_llm:
                answer,times['llm_s'],inp,output=llm_answer(q['question'],matches,llm_model)
            else:
                times['llm_s']=None
            times['retrieval_total_s']=sum(times[key] for key in ('initialization_s','embedding_s','retrieval_s','ranking_context_s'))
            times['end_to_end_s']=times['retrieval_total_s']+times['llm_s'] if run_llm else None
            row={'system':system,'question_id':q['id'],'question':q['question'],**metrics,
                 **{key:round(val,5) if val is not None else None for key,val in times.items()},
                 'input_tokens':inp,'output_tokens':output,'total_tokens':inp+output if inp is not None and output is not None else None,
                 'retrieved':[{'file':m['file'],'start_line':m.get('matched_start_line',m['start_line']),'end_line':m.get('matched_end_line',m['end_line']),'relevant':relevant(m,q['targets'])} for m in matches], 'answer':answer}
            rows.append(row)
            # Save each answer immediately so partial results survive interrupted runs.
            safe_id = ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(q['id']))
            answer_path = answer_dir / f'{system}_{safe_id}.md'
            retrieved_lines = '\n'.join(
                f"{i}. `{m['file']}:{m.get('matched_start_line',m['start_line'])}-{m.get('matched_end_line',m['end_line'])}`"
                for i, m in enumerate(matches, 1)
            )
            report = (
                f"# {system.upper()} — {q['id']}\n\n"
                f"**Question:** {q['question']}\n\n"
                f"**LLM model:** {llm_model if run_llm else 'Not run'}\n\n"
                f"**LLM latency:** {times['llm_s'] if run_llm else 'N/A'} seconds\n\n"
                f"**Input tokens:** {inp if inp is not None else 'N/A'}  "
                f"**Output tokens:** {output if output is not None else 'N/A'}\n\n"
                f"## Retrieved code references\n\n{retrieved_lines}\n\n"
                f"## Qwen answer\n\n{answer if run_llm else 'LLM generation disabled.'}\n"
            )
            answer_path.write_text(report, encoding='utf-8')
            # Incremental machine-readable backup.
            (out/'timed_results.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
            print('Saved LLM answer:', answer_path, flush=True)
            print(f"{system:8} {q['id']} | init={times['initialization_s']:.3f}s embed={times['embedding_s']:.3f}s search={times['retrieval_s']:.3f}s rank={times['ranking_context_s']:.3f}s LLM={times['llm_s'] if run_llm else 'off'} | Hit@{top_k}={metrics['hit_at_k']}",flush=True)
    (out/'timed_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    cols=['system','question_id','hit_at_k','precision_at_k','recall_at_k','reciprocal_rank','initialization_s','embedding_s','retrieval_s','ranking_context_s','llm_s','retrieval_total_s','end_to_end_s','input_tokens','output_tokens','total_tokens']
    with (out/'latency_breakdown.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    averages=[]
    for system in SYSTEMS:
        subset=[r for r in rows if r['system']==system]
        d={'system':system,'questions':len(subset)}
        for col in cols[2:]:
            vals=[r[col] for r in subset if r[col] is not None]
            d[col]=round(sum(vals)/len(vals),5) if vals else None
        averages.append(d)
    (out/'timed_summary.json').write_text(json.dumps(averages,indent=2),encoding='utf-8')
    with (out/'timed_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(averages[0]));w.writeheader();w.writerows(averages)
    # One combined report, suitable for reviewing all 12 answers side by side.
    combined = ['# Phase 6 — Qwen Answers (Student Management System)', '']
    for r in rows:
        combined.extend([
            f"## {r['system'].upper()} — {r['question_id']}",
            '', f"**Question:** {r['question']}", '',
            f"**LLM latency:** {r['llm_s']} seconds; **Input tokens:** {r['input_tokens']}; **Output tokens:** {r['output_tokens']}",
            '', '**Answer:**', '', r['answer'] if run_llm else 'LLM generation disabled.', '', '---', ''
        ])
    (out/'llm_answers_all.md').write_text('\n'.join(combined), encoding='utf-8')
    print('Saved results to:',out.resolve())
    return rows
