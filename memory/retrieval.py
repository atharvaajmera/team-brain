import math

import numpy as np

from memory.intent import analyze_query_intent
from memory.prf import run_prf_retrieval
from memory.storage import collection
from memory.ranking import mmr_sort

def build_chroma_filter(query):
    intent=analyze_query_intent(query)
    chroma_filter={}

    if intent['filter_timeline']:
        chroma_filter['ts']={"$gte":intent['filter_timeline']}

    if intent['aggregation']:
        print("Aggregation detected:", intent['aggregation'])

    if not chroma_filter:
        chroma_filter=None

    return chroma_filter


def _query_collection(query, chroma_filter=None, n_results=40):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=chroma_filter,
        include=['documents', 'metadatas', 'distances', 'embeddings']
    )

    if not results['documents'] or not results['documents'][0]:
        return []

    docs = results['documents'][0]
    metas = results['metadatas'][0]
    ids = results['ids'][0]
    dists = results['distances'][0]
    embs = results['embeddings'][0]

    candidates = []
    for doc, meta, id, dist, emb in zip(docs, metas, ids, dists, embs):
        candidates.append({
            "id": id,
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "embedding": emb
        })

    return candidates


def _softmax_entropy(values, temp=0.1):
    values = np.array(values)
    neg_over_t = -values / temp
    neg_over_t -= neg_over_t.max()
    exp_scores = np.exp(neg_over_t)
    probs = exp_scores / exp_scores.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_entropy = float(np.log2(len(values))) if len(values) > 1 else 1.0
    return entropy / max_entropy


def _compute_prf_gate_metrics(candidates):
    threads = {}
    for candidate in candidates:
        thread_id = candidate["metadata"]["thread_id"]
        threads.setdefault(thread_id, {"distances": []})
        threads[thread_id]["distances"].append(candidate["distance"])

    aggregates = []
    for thread_id, payload in threads.items():
        distances = payload["distances"]
        avg_distance = float(np.mean(distances))
        aggregates.append({
            "thread_id": thread_id,
            "thread_score": avg_distance - math.log(len(distances) + 1) * 0.25,
        })

    if not aggregates:
        return {"ent_score_T0.1": 0.0, "rel_gap": 1.0}

    sorted_threads = sorted(aggregates, key=lambda item: item["thread_score"])
    thread_scores = [item["thread_score"] for item in sorted_threads]
    entropy = _softmax_entropy(thread_scores, temp=0.1)

    if len(sorted_threads) >= 2:
        best = sorted_threads[0]
        second = sorted_threads[1]
        spread = sorted_threads[-1]["thread_score"] - best["thread_score"]
        rel_gap = (
            (second["thread_score"] - best["thread_score"]) / spread
            if spread > 0 else 1.0
        )
    else:
        rel_gap = 1.0

    return {
        "ent_score_T0.1": entropy,
        "rel_gap": rel_gap,
    }

def retrieve_candidates(query, intent, with_filter=True, use_prf=False):
    chroma_filter = build_chroma_filter(query) if with_filter else None
    n_results = 40

    first_pass = _query_collection(query, chroma_filter=chroma_filter, n_results=n_results)
    if not use_prf or not first_pass:
        return first_pass

    metrics = _compute_prf_gate_metrics(first_pass)
    entropy = metrics.get("ent_score_T0.1", 0.0)
    rel_gap = metrics.get("rel_gap", 1.0)
    if not isinstance(rel_gap, (int, float)):
        rel_gap = 1.0

    apply_prf = entropy > 0.6 or rel_gap < 0.15
    if not apply_prf:
        return first_pass

    def _retrieve_fn(expanded_query):
        return _query_collection(expanded_query, chroma_filter=chroma_filter, n_results=n_results)

    prf_result = run_prf_retrieval(
        query,
        first_pass,
        retrieve_fn=_retrieve_fn,
        limit=n_results,
    )
    merged = prf_result["merged_candidates"] or first_pass
    for candidate in merged:
        candidate.setdefault("prf_debug", {
            "original_query": query,
            "apply_prf": apply_prf,
            "trigger_entropy": entropy,
            "trigger_rel_gap": rel_gap,
            "expansion_terms": prf_result["expansion_terms"],
            "expanded_queries": prf_result["expanded_queries"],
        })
    return merged
