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

def retrieve_candidates(query, intent, with_filter=True, use_prf=False):
    chroma_filter = build_chroma_filter(query) if with_filter else None
    n_results = 40

    first_pass = _query_collection(query, chroma_filter=chroma_filter, n_results=n_results)
    if not use_prf or not first_pass:
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
            "expansion_terms": prf_result["expansion_terms"],
            "expanded_queries": prf_result["expanded_queries"],
        })
    return merged
