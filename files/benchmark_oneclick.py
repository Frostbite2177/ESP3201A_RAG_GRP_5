"""One-click PyPSA RAG benchmark. Python 3.11+. No answer leakage into prompts."""
import argparse, csv, hashlib, json, re, time, traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
SYSTEMS=('baseline','ast','hybrid','parent')

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def questions(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data,dict): data=data.get('questions',[])
    rows=[]
    for item in data:
        analysis=item.get('llm_analysis') or {}
        q=item.get('question') or ((item.get('title') or '')+'\n'+(item.get('body') or '')).strip()
        ref=item.get('reference_answer') or analysis.get('correct_answer')
        if q and ref:
            rows.append({'id':item.get('id',item.get('question_id',len(rows))), 'repo':item.get('repo',''), 'question':q,'reference_answer':ref,'rubric':item.get('rubric',analysis.get('scoring_rubric',[])),'url':item.get('url',''), 'targets':item.get('targets',[])})
    return rows

def signature(repo):
    root=Path(repo).resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    py=sorted(p for p in root.rglob('*.py') if not any(x in {'.git','.venv','venv','__pycache__','node_modules'} for x in p.relative_to(root).parts))
    if not py:raise ValueError('Repository contains no Python files')
    h=hashlib.sha256()
    for p in py:
        st=p.stat();h.update(f'{p.relative_to(root).as_posix()}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
    return {'path':str(root),'file_count':len(py),'fingerprint':h.hexdigest()}

def prepare(repo,index_root,force=False):
    from baseline_rag import build_index
    from ast_rag import build_ast_index
    from hybrid_rag import build_hybrid_index
    from parent_rag import build_parent_index
    sig=signature(repo);index_root=Path(index_root)
    builds={'baseline':build_index,'ast':build_ast_index,'hybrid':build_hybrid_index,'parent':build_parent_index}
    meta={}
    for name,fn in builds.items():
        folder=index_root/name;manifest=folder/'repository_manifest.json'
        required=['chunks.json','settings.json','vectors.faiss']+(['tokens.json'] if name in ('hybrid','parent') else [])+(['parents.json'] if name=='parent' else [])
        valid=manifest.exists() and all((folder/x).exists() for x in required)
        if valid:
            try:valid=json.loads(manifest.read_text(encoding='utf-8'))==sig
            except Exception:valid=False
        if force or not valid:
            print(f'INDEX {name}: building from {repo}',flush=True)
            t=time.perf_counter();count=fn(str(repo),str(folder));elapsed=time.perf_counter()-t
            dump(manifest,sig)
            meta[name]={'chunks':count,'index_build_seconds':round(elapsed,3),'reused':False}
        else:
            count=len(json.loads((folder/'chunks.json').read_text(encoding='utf-8')))
            print(f'INDEX {name}: reusing verified index ({count} chunks)',flush=True)
            meta[name]={'chunks':count,'index_build_seconds':0,'reused':True}
    return sig,meta

def get_messages(question,matches):
    context='\n\n'.join(f"SOURCE: {m['file']} lines {m['start_line']}-{m['end_line']}\n```python\n{m['code']}\n```" for m in matches)
    return [{'role':'system','content':'You are a code Q&A assistant. Answer only using provided repository snippets. Cite file paths and line ranges. If evidence is insufficient, say so. Do not invent code.'},{'role':'user','content':f'Question: {question}\n\nRetrieved repository snippets:\n{context}'}]

def llm_call(messages,model):
    import ollama
    t=time.perf_counter();r=ollama.chat(model=model,messages=messages,options={'temperature':0,'num_ctx':8192})
    wall=time.perf_counter()-t
    def field(key,default=None):return r.get(key,default) if isinstance(r,dict) else getattr(r,key,default)
    msg=field('message',{})
    content=msg.get('content','') if isinstance(msg,dict) else getattr(msg,'content','')
    ns=lambda key:round((field(key,0) or 0)/1e9,5)
    return {'answer':content,'prompt_eval_count':field('prompt_eval_count'),'eval_count':field('eval_count'), 'total_tokens':((field('prompt_eval_count') or 0)+(field('eval_count') or 0)) or None,'wall_seconds':round(wall,4),'ollama_total_seconds':ns('total_duration'),'ollama_load_seconds':ns('load_duration'),'ollama_prompt_seconds':ns('prompt_eval_duration'),'ollama_generation_seconds':ns('eval_duration')}

def target_hit(match,target):
    a=str(match['file']).replace('\\','/');b=str(target.get('file','')).replace('\\','/')
    if not b or not (a==b or a.endswith('/'+b) or b.endswith('/'+a)):return False
    if 'start_line' not in target or 'end_line' not in target:return True
    start=match.get('matched_start_line',match['start_line']);end=match.get('matched_end_line',match['end_line'])
    return start<=target['end_line'] and end>=target['start_line']

def ranking_metrics(matches,targets,k):
    if not targets:return {'available':False,'reason':'No independently verified relevant source locations provided; ranking metrics not computed.'}
    flags=[any(target_hit(m,t) for t in targets) for m in matches]
    import math
    dcg=sum(int(v)/math.log2(i+2) for i,v in enumerate(flags))
    ideal=sum(1/math.log2(i+2) for i in range(min(len(targets),k)))
    return {'available':True,'hit_at_k':int(any(flags)),'precision_at_k':sum(flags)/k,'recall_at_k':sum(any(target_hit(m,t) for m in matches) for t in targets)/len(targets),'mrr':next((1/i for i,v in enumerate(flags,1) if v),0),'ndcg_at_k':dcg/ideal if ideal else 0,'relevant_flags':flags}

def run(args):
    all_q=questions(args.questions)
    if not all_q:raise ValueError('No Q&A records with reference answers in '+str(args.questions))
    indices=[int(x) for x in args.indices.split(',') if x.strip()] if args.indices else list(range(args.start,args.start+args.count))
    if not indices or any(i<0 or i>=len(all_q) for i in indices):raise ValueError(f'Question indices must be 0..{len(all_q)-1}. Available: {len(all_q)}')
    if args.count<=0 or args.top_k<=0:raise ValueError('count and top-k must be positive')
    selected=[all_q[i] for i in indices]
    
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=Path(args.output)/('run_'+stamp);out.mkdir(parents=True,exist_ok=True)
    dump(out/'selected_questions.json',[{'index':i,**q} for i,q in zip(indices,selected)])
    sig,index_meta=prepare(args.repo,args.index_root,args.rebuild)
    dump(out/'run_config.json',{'repo':sig,'index_metadata':index_meta,'llm':args.llm,'question_indices':indices,'top_k':args.top_k,'notes':'Reference answers and rubrics are never supplied to LLM. Answer correctness requires separate review.'})
    from evaluation import search_system
    rows=[];summaries=defaultdict(list)
    for system in SYSTEMS:
        chunks=json.loads((Path(args.index_root)/system/'chunks.json').read_text(encoding='utf-8'))
        dump(out/system/'chunking.json',{'strategy':system,'count':len(chunks),'chunks':chunks})
        for i,q in zip(indices,selected):
            print(f'RUN {system} question index {i}, id={q["id"]}',flush=True)
            try:
                t=time.perf_counter();matches=search_system(system,q['question'],args.index_root,args.top_k,20,60,100);retrieval=time.perf_counter()-t
                messages=get_messages(q['question'],matches)
                llm=llm_call(messages,args.llm)
                ranking=ranking_metrics(matches,q['targets'],args.top_k)
                record={'system':system,'question_index':i,'question_id':q['id'],'question':q['question'],'reference_answer':q['reference_answer'],'rubric':q['rubric'],'retrieved':[{'rank':j,'relevant':ranking.get('relevant_flags',[None]*len(matches))[j-1] if ranking['available'] else None,**m} for j,m in enumerate(matches,1)],'ranking_evaluation':ranking,'prompt_messages':messages,'llm_reply':llm['answer'],'token_usage':{k:llm[k] for k in ('prompt_eval_count','eval_count','total_tokens')},'timing_seconds':{'retrieval_and_ranking':round(retrieval,4),'llm_wall':llm['wall_seconds'],'total_query':round(retrieval+llm['wall_seconds'],4),'ollama_total':llm['ollama_total_seconds'],'ollama_load':llm['ollama_load_seconds'],'ollama_prompt':llm['ollama_prompt_seconds'],'ollama_generation':llm['ollama_generation_seconds']},'answer_evaluation':{'status':'not_automatically_graded','reason':'Rubric requires independent grading; reference answer is retained for manual evaluation.'}}
                dump(out/system/f'question_{i}.json',record)
                rows.append({'system':system,'question_index':i,'question_id':q['id'],'retrieval_seconds':round(retrieval,4),'llm_seconds':llm['wall_seconds'],'total_seconds':round(retrieval+llm['wall_seconds'],4),'input_tokens':llm['prompt_eval_count'],'output_tokens':llm['eval_count'],'total_tokens':llm['total_tokens'],'mrr':ranking.get('mrr'),'recall_at_k':ranking.get('recall_at_k'),'ndcg_at_k':ranking.get('ndcg_at_k'),'status':'ok'})
            except Exception as e:
                dump(out/system/f'question_{i}_error.json',{'error':str(e),'traceback':traceback.format_exc()})
                rows.append({'system':system,'question_index':i,'question_id':q['id'],'status':'error','error':str(e)})
                print(f'ERROR {system} q{i}: {e}',flush=True)
            dump(out/'results.json',rows)
            with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
                cols=['system','question_index','question_id','retrieval_seconds','llm_seconds','total_seconds','input_tokens','output_tokens','total_tokens','mrr','recall_at_k','ndcg_at_k','status','error'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    for name in SYSTEMS:
        ok=[r for r in rows if r['system']==name and r['status']=='ok']
        metrics=['retrieval_seconds','llm_seconds','total_seconds','input_tokens','output_tokens','total_tokens','mrr','recall_at_k','ndcg_at_k']
        summaries[name]={'completed':len(ok),'errors':len([r for r in rows if r['system']==name and r['status']=='error']),'index':index_meta[name], 'averages':{k:round(sum(r[k] for r in ok if r.get(k)is not None)/len([r for r in ok if r.get(k)is not None]),4) if any(r.get(k)is not None for r in ok) else None for k in metrics}}
    dump(out/'summary.json',summaries)
    print(f'FINISHED. Results: {out.resolve()}',flush=True)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True);p.add_argument('--questions',default=str(ROOT/'output/hifirepoqa/questions.json'))
    p.add_argument('--index-root',default=str(ROOT/'output/evaluation_indexes'));p.add_argument('--output',default=str(ROOT/'output/benchmark_runs'))
    p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=1);p.add_argument('--indices',default='')
    p.add_argument('--llm',default='qwen2.5-coder:3b');p.add_argument('--top-k',type=int,default=3);p.add_argument('--rebuild',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
