from memory.intent import analyze_query_intent
from memory.retrieval import retrieve_candidates
from memory.ranking import select_anchor
from memory.storage import collection

def get_top_convo_id(query):
    print(f"\n[DEBUG] === Starting get_top_convo_id for query: '{query}' ===")
    
    intent = analyze_query_intent(query)
    timeline_filter = intent.get('timeline')
    
    candidates = retrieve_candidates(query, intent, with_filter=True)
    print(f"[DEBUG] Phase 1 returned {len(candidates)} candidates")

    result = select_anchor(candidates, mode="NORMAL")

    if result:
        print(f"[DEBUG] get_top_convo_id SUCCESS ({result['type'].upper()}): thread_ids={result['thread_ids']}")
        result['is_fallback'] = False
        result['fallback_reason'] = None
        return result

    print("[DEBUG] Phase 1 failed. Attempting Fallback...")

    candidates = retrieve_candidates(query, intent, with_filter=False)
    print(f"[DEBUG] Fallback phase returned {len(candidates)} candidates")
    
    result = select_anchor(candidates, mode="FALLBACK")
    
    if result:
        fallback_reason = f"No results found for temporal filter '{timeline_filter}'" if timeline_filter else "No exact match found"
        print(f"[DEBUG] get_top_convo_id SUCCESS (FALLBACK {result['type'].upper()}): thread_ids={result['thread_ids']}")
        result['is_fallback'] = True
        result['fallback_reason'] = fallback_reason
        return result

    print(f"[DEBUG] get_top_convo_id FAILED: No conversation found even after fallback")
    return None


def query_text_phase_2(query):
    print(f"\n[DEBUG] === Phase 2: Full thread retrieval ===")
    result = get_top_convo_id(query)
    if not result:
        print(f"[DEBUG] Phase 2 FAILED: No thread_id found")
        return None
    
    all_thread_messages = []

    for thread_id in result['thread_ids']:
        print(f"[DEBUG] Phase 2: Fetching all messages for thread_id={thread_id}")
        results = collection.get(
            where={"thread_id": thread_id}
        )
        if not results['documents']:
            print(f"[DEBUG] Phase 2: No documents found for thread_id={thread_id}, skipping")
            continue
        
        thread_msgs = []
        for doc, metadata, id in zip(results['documents'], results['metadatas'], results['ids']):
            thread_msgs.append({
                "id": id,
                "document": doc,
                "metadata": metadata
            })
        thread_msgs.sort(key=lambda x: float(x['metadata']['ts']))
        all_thread_messages.append({
            "thread_id": thread_id,
            "messages": thread_msgs
        })

    return {
        "type": result['type'],
        "threads": all_thread_messages,
        "is_fallback": result['is_fallback'],
        "fallback_reason": result['fallback_reason']
    }