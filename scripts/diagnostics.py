import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.storage import collection
from memory.query_planner import plan_query
from memory.decision import execute_plan
from memory.evidence import evaluate_evidence

DEFAULT_TEST_QUERIES_FILE = REPO_ROOT / "config" / "diagnostics_queries.json"

def _load_thread_corpus():
    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    threads = defaultdict(list)
    for document, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        thread_id = metadata.get("thread_id")
        if thread_id is None:
            continue
        tid = str(int(float(thread_id)))
        text = (metadata.get("text") or document or "").strip()
        if text:
            threads[tid].append(text.lower())

    return {
        tid: {
            "messages": messages,
            "full_text": "\n".join(messages),
        }
        for tid, messages in threads.items()
    }

def _normalize_expected_term_groups(value):
    if not value:
        return []
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return [[item.strip().lower() for item in value if item.strip()]]
        groups = []
        for group in value:
            if isinstance(group, list):
                cleaned = [str(item).strip().lower() for item in group if str(item).strip()]
                if cleaned:
                    groups.append(cleaned)
        return groups
    return []

def _resolve_expected_threads(item, thread_corpus):
    expected_status = str(item.get("expected_status", "")).strip().lower()
    if expected_status == "reject":
        return []

    explicit_ids = item.get("expected_thread_ids", [])
    resolved_ids = []
    for tid in explicit_ids:
        try:
            resolved_ids.append(str(int(float(tid))))
        except (TypeError, ValueError):
            continue
    if resolved_ids:
        return resolved_ids

    term_groups = _normalize_expected_term_groups(item.get("expected_thread_terms"))
    if not term_groups:
        return []

    matched = []
    for group in term_groups:
        group_match = None
        best_partial = None
        best_score = -1

        for tid, payload in thread_corpus.items():
            full_text = payload["full_text"]
            score = sum(term in full_text for term in group)
            if score == len(group):
                group_match = tid
                break
            if score > best_score:
                best_score = score
                best_partial = tid

        if group_match:
            matched.append(group_match)
        elif best_partial and best_score > 0:
            matched.append(best_partial)

    deduped = []
    seen = set()
    for tid in matched:
        if tid not in seen:
            deduped.append(tid)
            seen.add(tid)
    return deduped

def _load_test_queries(path=DEFAULT_TEST_QUERIES_FILE):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
        
    queries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        expected_goal = str(item.get("expected_goal", "")).strip().lower()
        expected_status = str(item.get("expected_status", "")).strip().lower()
        if not query:
            continue
        queries.append({
            "query": query,
            "expected_goal": expected_goal,
            "expected_status": expected_status,
            "desc": str(item.get("desc", "")).strip(),
            "expected_thread_ids": item.get("expected_thread_ids", []),
            "expected_thread_terms": item.get("expected_thread_terms", []),
        })
    return queries

import argparse
import os
from pydantic import ValidationError
from memory.models import QueryPlan

def run_diagnostics():
    parser = argparse.ArgumentParser(description="Run system diagnostics.")
    parser.add_argument("--offline", action="store_true", help="Run without calling the LLM planner (uses cached plans).")
    args = parser.parse_args()

    test_queries = _load_test_queries()
    if not test_queries:
        print("No queries found.")
        return

    thread_corpus = _load_thread_corpus()
    
    results = []
    
    print(f"Running diagnostics on {len(test_queries)} queries (Offline: {args.offline})...")
    
    cache_path = REPO_ROOT / "config" / "diagnostics_cache.json"
    plan_cache = {}
    if args.offline or cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                plan_cache = json.load(f)
        except Exception:
            plan_cache = {}
            
    planner_failures = 0
    
    for item in test_queries:
        query = item["query"]
        expected_goal = item["expected_goal"]
        expected_status = item["expected_status"]
        
        expected_tids = _resolve_expected_threads(item, thread_corpus)
        
        # 1. Plan query
        plan = None
        if args.offline:
            cached = plan_cache.get(query)
            if cached:
                try:
                    plan = QueryPlan(**cached)
                except ValidationError:
                    pass
            if not plan:
                print(f"Skipping '{query}' (no valid cached plan in offline mode)")
                continue
        else:
            try:
                plan = plan_query(query)
                plan_cache[query] = plan.model_dump()
            except Exception as e:
                print(f"Planner failed for '{query}': {e}")
                planner_failures += 1
                # Try fallback to cache
                cached = plan_cache.get(query)
                if cached:
                    try:
                        plan = QueryPlan(**cached)
                    except ValidationError:
                        pass
                if not plan:
                    continue

        goal_correct = (plan.goal == expected_goal)
        
        # 2. Execute plan (Retrieval)
        threads = execute_plan(plan, query, {}, allowed_channel_ids=None)
        retrieved_tids = [str(int(float(t["thread_id"]))) for t in threads]
        
        retrieval_hit = False
        if expected_status == "reject":
            retrieval_hit = True # N/A basically
        elif not expected_tids:
            retrieval_hit = False # Could not find expected thread in corpus
        else:
            # We consider it a hit if ANY expected thread is in the top 5
            top_5 = retrieved_tids[:5]
            retrieval_hit = any(tid in top_5 for tid in expected_tids)
            
        # 3. Evaluate Evidence
        evidence = evaluate_evidence(plan, threads, query)
        
        # 4. End-to-end correctness
        if plan.goal == "reject":
            actual_status = "reject"
        elif not evidence.strong_enough:
            actual_status = "clarify"
        else:
            actual_status = "ok"
            
        status_correct = (actual_status == expected_status)
        
        results.append({
            "query": query,
            "expected_goal": expected_goal,
            "actual_goal": plan.goal,
            "goal_correct": goal_correct,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "status_correct": status_correct,
            "retrieval_hit": retrieval_hit,
        })
        
    # Summary
    if not args.offline:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(plan_cache, f, indent=2)
        except Exception as e:
            pass # ignore

    planner_correct = sum(1 for r in results if r["goal_correct"])
    retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
    e2e_correct = sum(1 for r in results if r["status_correct"])
    
    total = len(results)
    
    print("\n" + "="*50)
    print("DIAGNOSTICS SUMMARY")
    print("="*50)
    print(f"Total Queries:         {total}")
    if not args.offline:
        print(f"Planner API Failures:  {planner_failures}")
    print(f"Planner Accuracy:      {planner_correct}/{total} ({planner_correct/total*100:.1f}%)")
    print(f"Retrieval Hit Rate:    {retrieval_hits}/{total} ({retrieval_hits/total*100:.1f}%)")
    print(f"End-to-End Correct:    {e2e_correct}/{total} ({e2e_correct/total*100:.1f}%)")
    
    print("\nFailed End-to-End:")
    for r in results:
        if not r["status_correct"]:
            print(f"  Query: '{r['query']}'")
            print(f"    Expected: {r['expected_status']}, Actual: {r['actual_status']} (Goal: {r['actual_goal']}, Hit: {r['retrieval_hit']})")

if __name__ == "__main__":
    run_diagnostics()
