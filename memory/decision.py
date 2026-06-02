from memory.intent import analyze_query_intent
from memory.retrieval import retrieve_candidates
from memory.ranking import select_anchor
from memory.storage import collection

def get_top_convo_id(query):
    intent = analyze_query_intent(query)
    timeline_filter = intent.get('timeline')

    candidates = retrieve_candidates(query, intent, with_filter=True)

    result = select_anchor(candidates, mode="NORMAL")

    if result:
        result['is_fallback'] = False
        result['fallback_reason'] = None
        return result

    # Fallback: retry without temporal filter
    candidates = retrieve_candidates(query, intent, with_filter=False)

    result = select_anchor(candidates, mode="FALLBACK")

    if result:
        fallback_reason = f"No results found for temporal filter '{timeline_filter}'" if timeline_filter else "No exact match found"
        result['is_fallback'] = True
        result['fallback_reason'] = fallback_reason
        return result

    return None


def query_text_phase_2(query):
    result = get_top_convo_id(query)
    if not result:
        return None

    all_thread_messages = []

    for thread_id in result['thread_ids']:
        results = collection.get(
            where={"thread_id": thread_id}
        )
        if not results['documents']:
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
        "fallback_reason": result['fallback_reason'],
        "stats": result.get('stats', {}),
        "thread_debug": result.get('thread_debug', []),
    }
