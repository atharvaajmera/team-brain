"""Main routing logic for user queries."""

from time import perf_counter
from memory.query_planner import plan_query
from memory.retrieval import execute_semantic_search, execute_recent_threads
from memory.privacy import scan_threads, redact_threads
from memory.shared import group_threads

def process_query(query: str):
    """Orchestrate tool-based planning, retrieval, and privacy routing."""
    timings = {}
    t0 = perf_counter()

    t = perf_counter()
    plan = plan_query(query)
    timings["plan"] = perf_counter() - t

    goal = plan.get("goal", "answer")

    if goal == "reject":
        timings["total"] = perf_counter() - t0
        return {
            "action": "REJECT",
            "threads": [],
            "route": "local",
            "scan": None,
            "plan": plan,
            "timings": timings,
        }

    t = perf_counter()
    all_candidates = []
    
    for step in plan.get("retrieval_steps", []):
        tool = step.get("tool")
        step_query = step.get("query")
        filters = step.get("filters", {})
        limit = step.get("limit", 40)
        
        if tool in ("semantic_search", "author_search"):
            candidates = execute_semantic_search(step_query or query, filters, limit=limit)
            all_candidates.extend(candidates)
        elif tool == "recent_threads":
            candidates = execute_recent_threads(filters, limit=limit)
            all_candidates.extend(candidates)
            
    timings["retrieve"] = perf_counter() - t

    t = perf_counter()
    
    seen = set()
    unique_candidates = []
    for c in all_candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique_candidates.append(c)
            
    sorted_threads = group_threads(unique_candidates)
    
    top_thread_ids = [t["thread_id"] for t in sorted_threads[:5]]
    timings["rank"] = perf_counter() - t

    t = perf_counter()
    threads = []
    if top_thread_ids:
        from memory.storage import collection
        for thread_id in top_thread_ids:
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
        "action": goal.upper(),
        "threads": final_threads,
        "route": scan.route,
        "scan": scan,
        "plan": plan,
        "timings": timings,
    }
