"""Main routing logic for user queries."""

from time import perf_counter

from memory.models import QueryPlan
from memory.retrieval import execute_recent_threads, execute_semantic_search
from memory.shared import group_threads


def _rank_recent_threads(candidates):
    """Group recent candidates by thread_id and rank by newest timestamp."""
    threads = {}
    for c in candidates:
        meta = c.get("metadata") or {}
        tid = meta.get("thread_id", meta.get("ts", c.get("id")))
        if tid is None:
            continue
        threads.setdefault(tid, {"candidates": [], "ts_list": []})
        threads[tid]["candidates"].append(c)
        ts = float(meta.get("ts", 0) or 0)
        threads[tid]["ts_list"].append(ts)

    aggregates = []
    for tid, td in threads.items():
        newest_ts = max(td["ts_list"])
        aggregates.append(
            {
                "thread_id": tid,
                "avg_distance": 0.0,
                "min_distance": 0.0,
                "message_count": len(td["candidates"]),
                "thread_score": -newest_ts,  # negative so that smaller (more negative) is better in ascending sort
                "best_candidate": max(
                    td["candidates"], key=lambda x: float(x["metadata"].get("ts", 0))
                ),
            }
        )

    # Sort ascending (most negative = newest = first)
    return sorted(aggregates, key=lambda x: x["thread_score"])


def _fetch_full_threads(thread_specs):
    """Expand thread IDs into full message lists from ChromaDB."""
    if not thread_specs:
        return []

    from memory.storage import collection

    threads = []
    for thread_id, channel_id in thread_specs:
        where_clause = {"thread_id": thread_id}
        if channel_id:
            where_clause = {
                "$and": [{"thread_id": thread_id}, {"channel_id": channel_id}]
            }

        results = collection.get(where=where_clause)
        if not results["documents"]:
            continue

        thread_msgs = []
        for doc, meta, doc_id in zip(
            results["documents"], results["metadatas"], results["ids"]
        ):
            thread_msgs.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                }
            )
        thread_msgs.sort(key=lambda x: float(x["metadata"]["ts"]))
        threads.append(
            {
                "thread_id": thread_id,
                "messages": thread_msgs,
            }
        )
    return threads


from datetime import datetime, timedelta


def _broaden_catch_up_filters(original_after_str: str | None) -> list[str | None]:
    """Return a sequence of progressively broader 'after' dates."""
    if not original_after_str:
        return [None]

    try:
        dt = datetime.strptime(original_after_str, "%Y-%m-%d")
    except ValueError:
        return [original_after_str, None]

    today = datetime.now()
    days_diff = (today - dt).days

    # If the original query is already older than a week, just fallback to None
    if days_diff > 7:
        return [original_after_str, None]

    three_days_ago = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    seven_days_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    windows = [original_after_str]
    if three_days_ago != original_after_str:
        windows.append(three_days_ago)
    if seven_days_ago != original_after_str and seven_days_ago != three_days_ago:
        windows.append(seven_days_ago)
    windows.append(None)

    return windows


def _handle_search(
    query: str,
    plan: QueryPlan,
    timings: dict,
    allowed_channel_ids: list[str] | None = None,
):
    """Standard semantic search path."""
    t = perf_counter()
    all_candidates = []

    is_catch_up = plan.goal == "catch_up"

    # Track if we broadened the search
    broadened = False
    original_after = None

    for step in plan.retrieval_steps:
        tool = step.tool
        step_query = step.query
        filters = step.filters.model_dump(exclude_none=True)
        limit = step.limit
        
        if original_after is None:
            original_after = filters.get("after")

        after_windows = (
            _broaden_catch_up_filters(filters.get("after"))
            if is_catch_up
            else [filters.get("after")]
        )

        for i, after_val in enumerate(after_windows):
            current_filters = dict(filters)
            if after_val is not None:
                current_filters["after"] = after_val
            elif "after" in current_filters:
                del current_filters["after"]

            if tool in ("semantic_search", "author_search"):
                candidates = execute_semantic_search(
                    step_query or query,
                    current_filters,
                    limit=limit,
                    allowed_channel_ids=allowed_channel_ids,
                )
            elif tool == "recent_threads":
                candidates = execute_recent_threads(
                    current_filters,
                    limit=limit,
                    allowed_channel_ids=allowed_channel_ids,
                )
            else:
                candidates = []

            if candidates:
                all_candidates.extend(candidates)
                if i > 0:
                    broadened = True
                break  # Stop broadening if we found results

    timings["retrieve"] = perf_counter() - t
    return all_candidates, broadened, original_after


def _handle_summarize(
    query: str,
    plan: QueryPlan,
    timings: dict,
    allowed_channel_ids: list[str] | None = None,
    allow_query_decomposition: bool = True,
):
    """Decompose a broad query into sub-queries for wider topic coverage."""
    fmt = plan.answer_requirements.format
    if (not allow_query_decomposition) or fmt in ("timeline", "comparison"):
        # Skip decomposition when cloud planning is disabled, or when chronology/comparison matters more.
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
        candidates = execute_semantic_search(
            sub_query, filters, limit=limit, allowed_channel_ids=allowed_channel_ids
        )
        all_candidates.extend(candidates)

    timings["retrieve"] = perf_counter() - t
    return all_candidates, False, None


def execute_plan(
    plan: QueryPlan,
    query: str,
    timings: dict,
    allowed_channel_ids: list[str] | None = None,
    allow_query_decomposition: bool = True,
):
    """Execute the retrieval steps from a QueryPlan and return full threads."""
    goal = plan.goal

    if goal == "reject" or goal == "clarify":
        return []

    # --- Route by goal ---
    broadened = False
    original_after = None
    if goal == "summarize":
        all_candidates, broadened, original_after = _handle_summarize(
            query,
            plan,
            timings,
            allowed_channel_ids,
            allow_query_decomposition=allow_query_decomposition,
        )
    else:
        all_candidates, broadened, original_after = _handle_search(
            query, plan, timings, allowed_channel_ids
        )

    timings["_broadened"] = original_after if broadened else None

    # --- Deduplicate ---
    t = perf_counter()
    seen = set()
    unique_candidates = []
    for c in all_candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique_candidates.append(c)

    is_recent = goal == "catch_up" or any(
        step.tool == "recent_threads" for step in plan.retrieval_steps
    )

    if is_recent:
        sorted_threads = _rank_recent_threads(unique_candidates)
    else:
        sorted_threads = group_threads(unique_candidates)

    # --- Format specific logic ---
    fmt = plan.answer_requirements.format
    if fmt == "decision":
        decision_markers = [
            "decided",
            "decision",
            "agreed",
            "going with",
            "approved",
            "resolution",
        ]
        for th in sorted_threads:
            doc = (th.get("best_candidate", {}).get("document") or "").lower()
            if any(m in doc for m in decision_markers):
                th["thread_score"] -= 0.2
        # Re-sort after boosting
        sorted_threads.sort(key=lambda x: x["thread_score"])

    top_5 = sorted_threads[:5]

    if fmt == "timeline":
        top_5.sort(
            key=lambda th: float(
                th.get("best_candidate", {}).get("metadata", {}).get("ts", 0)
            )
        )

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
