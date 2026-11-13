"""Phase 4: AST chunks indexed for both BM25 and dense vector retrieval."""
from __future__ import annotations
import json
import re
from pathlib import Path
from ast_rag import ast_chunks
from baseline_rag import DEFAULT_EMBED_MODEL, answer


def tokenize(text: str) -> list[str]:
    """Preserve identifiers and split snake_case/camelCase for keyword matching."""
    tokens = []
    for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+", text):
        tokens.append(word.lower())
        tokens.extend(part.lower() for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+", word.replace('_',' ')) if part.lower() != word.lower())
    return tokens


def reciprocal_rank_fusion(rankings: list[list[int]], rrf_k: int = 60) -> list[tuple[int,float]]:
    """Fuse rank lists without mixing incompatible BM25/cosine score scales."""
    if rrf_k <= 0: raise ValueError('rrf_k must be positive')
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(dict.fromkeys(ranking), start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0/(rrf_k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def build_hybrid_index(repo: str, output: str = 'output/hybrid', embed_model: str = DEFAULT_EMBED_MODEL):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    chunks = ast_chunks(repo)
    if not chunks: raise ValueError('No valid Python chunks found')
    folder = Path(output); folder.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(SentenceTransformer(embed_model).encode([c['code'] for c in chunks], show_progress_bar=True), dtype='float32')
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    faiss.write_index(index, str(folder/'vectors.faiss'))
    (folder/'chunks.json').write_text(json.dumps(chunks, indent=2), encoding='utf-8')
    (folder/'tokens.json').write_text(json.dumps([tokenize(c['code']) for c in chunks]), encoding='utf-8')
    (folder/'settings.json').write_text(json.dumps({'embedding_model': embed_model, 'chunking':'python_ast', 'fusion':'RRF', 'rrf_k':60}, indent=2), encoding='utf-8')
    return len(chunks)


def search_hybrid(query: str, output: str = 'output/hybrid', top_k: int = 3, candidate_k: int = 20, rrf_k: int = 60, mode: str = 'hybrid'):
    if top_k <= 0 or candidate_k <= 0: raise ValueError('top_k and candidate_k must be positive')
    if mode not in {'hybrid','bm25','dense'}: raise ValueError('Invalid mode')
    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    folder = Path(output)
    chunks = json.loads((folder/'chunks.json').read_text(encoding='utf-8'))
    settings = json.loads((folder/'settings.json').read_text(encoding='utf-8'))
    limit = min(len(chunks), max(top_k, candidate_k))
    rankings = []
    if mode in {'hybrid','bm25'}:
        tokens = json.loads((folder/'tokens.json').read_text(encoding='utf-8'))
        bm25 = BM25Okapi(tokens)
        scores = bm25.get_scores(tokenize(query))
        # Exclude zero-score candidates: no keyword evidence for these chunks.
        bm25_ids = sorted((i for i in range(len(chunks)) if scores[i] > 0), key=lambda i: (-float(scores[i]),i))[:limit]
        rankings.append(bm25_ids)
    if mode in {'hybrid','dense'}:
        index = faiss.read_index(str(folder/'vectors.faiss'))
        vector = np.asarray(SentenceTransformer(settings['embedding_model']).encode([query]), dtype='float32')
        faiss.normalize_L2(vector)
        _, ids = index.search(vector, limit)
        rankings.append([int(i) for i in ids[0] if i >= 0])
    fused = reciprocal_rank_fusion(rankings, rrf_k)[:top_k]
    return [{**chunks[i], 'score':score, 'retrieval_mode':mode} for i,score in fused]
