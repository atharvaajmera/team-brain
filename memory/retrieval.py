from memory.intent import analyze_query_intent
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

def retrieve_candidates(query, intent, with_filter=True):
    chroma_filter = build_chroma_filter(query) if with_filter else None
    n_results = 40 
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=chroma_filter,
        include=['documents', 'metadatas', 'distances', 'embeddings']
    )

    if not results['documents'] or not results['documents'][0]:
        print(f"[DEBUG] Phase 1 FAILED: No documents found for query='{query}' with filter={chroma_filter}")
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