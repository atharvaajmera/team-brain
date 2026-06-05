"""Main routing logic for user queries."""

from time import perf_counter
from memory.query_planner import plan_query
from memory.retrieval import retrieve_candidates
from memory.privacy import scan_threads, redact_threads
from memory.ranking import select_anchor

def process_query(query: str):
    """Orchestrate intent planning, retrieval, and privacy routing."""
    timings = {}
    t0 = perf_counter()

    t = perf_counter()
    plan = plan_query(query)
    timings["plan"] = perf_counter() - t

    action = plan["action"]

    if action == "reject":
        timings["total"] = perf_counter() - t0
        return {
            "action": "REJECT",
            "threads": [],
            "route": "local",
            "scan": None,
            "plan": plan,
            "timings": timings,
        }

    # Retrieve candidates based on the search query (or original query if none)
    search_query = plan.get("search_query") or query
    filters = plan.get("filters") or {}

    t = perf_counter()
    candidates = retrieve_candidates(search_query, filters)
    timings["retrieve"] = perf_counter() - t

    t = perf_counter()
    anchor = select_anchor(candidates, mode="NORMAL")
    timings["rank"] = perf_counter() - t

    t = perf_counter()
    threads = []
    if anchor and "thread_ids" in anchor:
        from memory.storage import collection
        for thread_id in anchor["thread_ids"]:
            results = collection.get(where={"thread_id": thread_id})
            if not results["documents"]:
                continue

            thread_msgs = []
            for doc, meta, doc_id in zip(results["documents"], results["metadatas"], results["ids"]):
                thread_msgs.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta
                })
            thread_msgs.sort(key=lambda x: float(x["metadata"]["ts"]))
            threads.append({
                "thread_id": thread_id,
                "messages": thread_msgs
            })
    timings["fetch_threads"] = perf_counter() - t

    # Privacy scan
    t = perf_counter()
    scan = scan_threads(query, threads)
    timings["privacy_scan"] = perf_counter() - t

    t = perf_counter()
    if scan.route == "cloud":
        final_threads = redact_threads(threads)
    else:
        final_threads = threads
    timings["redact"] = perf_counter() - t

    timings["total"] = perf_counter() - t0

    return {
        "action": action.upper(),
        "threads": final_threads,
        "route": scan.route,
        "scan": scan,
        "plan": plan,
        "timings": timings,
    }
