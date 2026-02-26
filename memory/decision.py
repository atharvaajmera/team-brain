from memory.intent import analyze_query_intent
from memory.retrieval import retrieve_candidates
from memory.ranking import select_anchor
from memory.storage import collection

def get_top_convo_id(query):
    print(f"\n[DEBUG] === Starting get_top_convo_id for query: '{query}' ===")
    
    intent = analyze_query_intent(query)
    timeline_filter = intent.get('timeline')
    
    candidates = retrieve_candidates(query,intent, with_filter=True)
    print(f"[DEBUG] Phase 1 returned {len(candidates)} candidates")

    anchor = select_anchor(candidates, mode="NORMAL")

    if anchor:
        print(f"[DEBUG] get_top_convo_id SUCCESS: Found thread_id={anchor['metadata']['thread_id']}")
        return {
            "thread_id": anchor['metadata']['thread_id'],
            "is_fallback": False,
            "fallback_reason": None
        }

    print("[DEBUG] Phase 1 failed (No anchor or strict threshold). Attempting Fallback...")

    candidates = retrieve_candidates(query, intent, with_filter=False)
    print(f"[DEBUG] Fallback phase returned {len(candidates)} candidates")
    
    anchor = select_anchor(candidates, mode="FALLBACK")
    
    if anchor:
        fallback_reason = f"No results found for temporal filter '{timeline_filter}'" if timeline_filter else "No exact match found"
        print(f"[DEBUG] get_top_convo_id SUCCESS (FALLBACK): Found thread_id={anchor['metadata']['thread_id']}")
        return {
            "thread_id": anchor['metadata']['thread_id'],
            "is_fallback": True,
            "fallback_reason": fallback_reason
        }

    print(f"[DEBUG] get_top_convo_id FAILED: No conversation found even after fallback")
    return None


def query_text_phase_2(query):
    print(f"\n[DEBUG] === Phase 2: Full thread retrieval ===")
    result = get_top_convo_id(query)
    if not result:
        print(f"[DEBUG] Phase 2 FAILED: No thread_id found")
        return []
    
    top_convo_id = result['thread_id']
    is_fallback = result['is_fallback']
    fallback_reason = result['fallback_reason']
    
    print(f"[DEBUG] Phase 2: Fetching all messages for thread_id={top_convo_id}")
    results = collection.get(
        where={
            "thread_id": top_convo_id
        }
    )
    if not results['documents']:
        print(f"[DEBUG] Phase 2 FAILED: No documents found for thread_id={top_convo_id}")
        return []
    
    ids = results['ids']
    documents = results['documents']
    metadatas = results['metadatas']
    outputs = []
    for doc, metadata, id in zip(documents, metadatas, ids):
        output = {
            "id": id,
            "document": doc,
            "metadata": metadata
        }
        outputs.append(output)
    outputs.sort(key=lambda x: float(x['metadata']['ts']))
    return {
        "messages": outputs,
        "is_fallback": is_fallback,
        "fallback_reason": fallback_reason
    }