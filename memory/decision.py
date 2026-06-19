"""Main routing logic for user queries."""

from time import perf_counter
from memory.models import QueryPlan
from memory.retrieval import execute_semantic_search, execute_recent_threads
from memory.shared import group_threads


def _fetch_full_threads(thread_specs):
    """Expand thread IDs into full message lists from ChromaDB."""
    if not thread_specs:
        return []

    from memory.storage import collection

    threads = []
    for thread_id, channel_id in thread_specs:
        where_clause = {"thread_id": thread_id}
        if channel_id:
            where_clause = {"$and": [{"thread_id": thread_id}, {"channel_id": channel_id}]}
            
        results = collection.get(where=where_clause)
        if not results["documents"]:
            continue

        thread_msgs = []
        for doc, meta, doc_id in zip(
            results["documents"], results["metadatas"], results["ids"]
        ):
            thread_msgs.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta,
            })
        thread_msgs.sort(key=lambda x: float(x["metadata"]["ts"]))
        threads.append({
            "thread_id": thread_id,
            "messages": thread_msgs,
        })
    return threads


def _handle_search(query: str, plan: QueryPlan, timings: dict, allowed_channel_ids: list[str] | None = None):
    """Standard semantic search path."""
    t = perf_counter()
    all_candidates = []

    for step in plan.retrieval_steps:
        tool = step.tool
        step_query = step.query
        filters = step.filters.model_dump(exclude_none=True)
        limit = step.limit

        if tool in ("semantic_search", "author_search"):
            candidates = execute_semantic_search(
                step_query or query, filters, limit=limit, allowed_channel_ids=allowed_channel_ids
            )
            all_candidates.extend(candidates)
        elif tool == "recent_threads":
            candidates = execute_recent_threads(filters, limit=limit, allowed_channel_ids=allowed_channel_ids)
            all_candidates.extend(candidates)

    timings["retrieve"] = perf_counter() - t
    return all_candidates


def _handle_summarize(query: str, plan: QueryPlan, timings: dict, allowed_channel_ids: list[str] | None = None):
    """Decompose a broad query into sub-queries for wider topic coverage."""
    fmt = plan.answer_requirements.format
    if fmt in ("timeline", "comparison"):
        # Don't decompose for timeline (needs pure chronology) or comparison (planner provides steps)
        return _handle_search(query, plan, timings, allowed_channel_ids)

    from memory.decomposition import decompose_query

    t = perf_counter()

    # Extract filters from the planner's step (if any)
    steps = plan.retrieval_steps
    filters = steps[0].filters.model_dump(exclude_none=True) if steps else {}
    limit = steps[0].limit if steps else 40

    decomp = decompose_query(query)
    timings["decompose"] = perf_counter() - t

    timings["_decomposition"] = decomp

    t = perf_counter()
    all_candidates = []

    for sub_query in decomp["sub_queries"]:
        candidates = execute_semantic_search(sub_query, filters, limit=limit, allowed_channel_ids=allowed_channel_ids)
        all_candidates.extend(candidates)

    timings["retrieve"] = perf_counter() - t
    return all_candidates


def execute_plan(plan: QueryPlan, query: str, timings: dict, allowed_channel_ids: list[str] | None = None):
    """Execute the retrieval steps from a QueryPlan and return full threads."""
    goal = plan.goal

    if goal == "reject" or goal == "clarify":
        return []

    # --- Route by goal ---
    if goal == "summarize":
        all_candidates = _handle_summarize(query, plan, timings, allowed_channel_ids)
    else:
        all_candidates = _handle_search(query, plan, timings, allowed_channel_ids)

    # --- Deduplicate ---
    t = perf_counter()
    seen = set()
    unique_candidates = []
    for c in all_candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique_candidates.append(c)

    sorted_threads = group_threads(unique_candidates)
    
    # --- Format specific logic ---
    fmt = plan.answer_requirements.format
    if fmt == "decision":
        decision_markers = ["decided", "decision", "agreed", "going with", "approved", "resolution"]
        for th in sorted_threads:
            doc = (th.get("best_candidate", {}).get("document") or "").lower()
            if any(m in doc for m in decision_markers):
                th["thread_score"] += 0.2
        # Re-sort after boosting
        sorted_threads.sort(key=lambda x: x["thread_score"], reverse=True)

    top_5 = sorted_threads[:5]
    
    if fmt == "timeline":
        top_5.sort(key=lambda th: float(th.get("best_candidate", {}).get("metadata", {}).get("ts", 0)))
    
    top_thread_specs = []
    for th in top_5:
        cid = th.get("best_candidate", {}).get("metadata", {}).get("channel_id")
        top_thread_specs.append((th["thread_id"], cid))

    # Build a lookup of retrieval scores keyed by thread_id
    retrieval_scores = {}
    for th in top_5:
        retrieval_scores[th["thread_id"]] = {
            "thread_score": th["thread_score"],
            "min_distance": th["min_distance"],
            "avg_distance": th["avg_distance"],
            "message_count": th["message_count"],
        }

    timings["rank"] = perf_counter() - t

    # --- Expand full threads ---
    t = perf_counter()
    threads = _fetch_full_threads(top_thread_specs)
    timings["fetch_threads"] = perf_counter() - t

    # --- Attach retrieval scores to each thread so evidence.py can use them ---
    for thread in threads:
        tid = thread["thread_id"]
        thread["_retrieval_score"] = retrieval_scores.get(tid, {})

    return threads

