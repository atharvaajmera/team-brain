import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from memory.storage import collection
from memory.retrieval import retrieve_candidates
from memory.intent import analyze_query_intent
from scripts.benchmark import (
    compute_metrics, decide, compute_population_stats,
    _group_threads, run_logreg_loo, MIN_THREAD_SIZE,
)

DEFAULT_TEST_QUERIES_FILE = REPO_ROOT / "config" / "diagnostics_queries.json"


def _tid_label(ts):
    return str(int(float(ts)))


def _recall_at_k(ranked_thread_ids, expected_tids, k=5):
    if not expected_tids:
        return None
    top_k = ranked_thread_ids[:k]
    return 1 if any(t in top_k for t in expected_tids) else 0


def _mrr(ranked_thread_ids, expected_tids):
    if not expected_tids:
        return None
    for rank, tid in enumerate(ranked_thread_ids, start=1):
        if tid in expected_tids:
            return round(1.0 / rank, 4)
    return 0.0


def _rank_threads(candidates):
    agg = _group_threads(candidates)
    agg_sorted = sorted(agg, key=lambda x: x["thread_score"])
    multi = [t for t in agg_sorted if t["message_count"] >= MIN_THREAD_SIZE]
    if multi:
        agg_sorted = multi
    return [str(int(float(t["thread_id"]))) for t in agg_sorted]


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
    expected = str(item.get("expected", "")).strip().upper()
    if expected == "REJECT":
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
            print(
                f"  Warning: partial expected-thread match for query "
                f"'{item.get('query', '')[:30]}...' with terms {group} -> {best_partial}"
            )
            matched.append(best_partial)
        else:
            print(
                f"  Warning: no expected thread match for query "
                f"'{item.get('query', '')[:30]}...' with terms {group}"
            )

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
        expected = str(item.get("expected", "")).strip().upper()
        desc = str(item.get("desc", "")).strip()
        if not query or not expected:
            continue
        queries.append({
            "query": query,
            "expected": expected,
            "desc": desc,
            "expected_thread_ids": item.get("expected_thread_ids", []),
            "expected_thread_terms": item.get("expected_thread_terms", []),
        })
    return queries


def _evaluate_queries(test_queries, thread_corpus, use_prf=False):
    all_results = []

    for item in test_queries:
        query = item["query"]
        expected = item["expected"]
        desc = item["desc"]

        intent = analyze_query_intent(query)
        candidates = retrieve_candidates(query, intent, with_filter=False, use_prf=use_prf)
        if not candidates:
            continue

        m = compute_metrics(candidates)
        m["query"] = query
        m["expected"] = expected
        m["desc"] = desc

        expected_tids = _resolve_expected_threads(item, thread_corpus)
        ranked_tids = _rank_threads(candidates)

        if expected != "REJECT" and not expected_tids:
            print(f"  Warning: no expected threads resolved for '{query}'")

        m["expected_threads"] = expected_tids
        m["ranked_threads"] = ranked_tids
        m["recall_5"] = _recall_at_k(ranked_tids, expected_tids, k=5)
        m["mrr"] = _mrr(ranked_tids, expected_tids)
        m["prf_debug"] = candidates[0].get("prf_debug") if use_prf and candidates else None

        all_results.append(m)

    if not all_results:
        return []

    pop_stats = compute_population_stats(all_results)
    for m in all_results:
        m["predicted"] = decide(m, pop_stats)
        m["correct"] = m["predicted"] == m["expected"]

    lr_preds = run_logreg_loo(all_results)
    for i, m in enumerate(all_results):
        m["lr_predicted"] = lr_preds[i]
        m["lr_correct"] = lr_preds[i] == m["expected"]

    return all_results


def _summarize_results(all_results):
    has_threads = [m for m in all_results if m["recall_5"] is not None]
    avg_recall = np.mean([m["recall_5"] for m in has_threads]) if has_threads else 0.0
    avg_mrr = np.mean([m["mrr"] for m in has_threads]) if has_threads else 0.0
    rule_errors = sum(1 for m in all_results if m["predicted"] != m["expected"])
    lr_errors = sum(1 for m in all_results if m["lr_predicted"] != m["expected"])
    return {
        "queries": len(all_results),
        "recall@5": avg_recall,
        "mrr": avg_mrr,
        "rule_errors": rule_errors,
        "lr_errors": lr_errors,
    }


def _retrieval_hit_flag(result):
    recall = result.get("recall_5")
    if recall is None:
        return "N/A"
    return "Y" if recall == 1 else "N"


def _bucket(value, low_cut, high_cut):
    if value < low_cut:
        return "low"
    if value < high_cut:
        return "mid"
    return "high"


def _print_geometric_view(all_results, pred_key="predicted", name="RULE-BASED"):
    print(f"\n  GEOMETRIC VIEW - {name}")
    print("  (entropy x coherence for failure cases)")

    wrong = [m for m in all_results if m[pred_key] != m["expected"]]
    if not wrong:
        print("  (none)")
        return

    hdr = (
        f"  {'Query':<35} {'Entropy':>8} {'Coherence':>10} "
        f"{'Label':<10} {'Predicted':<10} {'Retr.Hit':<9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for m in wrong:
        entropy = m.get("ent_score_T0.1", 0.0)
        coherence = m.get("semantic_coherence_top5", 0.0)
        retrieval_hit = _retrieval_hit_flag(m)
        print(
            f"  {m['query'][:33]:<35} "
            f"{entropy:>8.4f} {coherence:>10.4f} "
            f"{m['expected']:<10} {m[pred_key]:<10} {retrieval_hit:<9}"
        )

    entropy_values = [m.get("ent_score_T0.1", 0.0) for m in all_results]
    coherence_values = [m.get("semantic_coherence_top5", 0.0) for m in all_results]
    ent_low, ent_high = np.percentile(entropy_values, [33, 66])
    coh_low, coh_high = np.percentile(coherence_values, [33, 66])

    print("\n  Region summary by expected label:")
    print(
        f"  entropy cuts: low<{ent_low:.3f}, mid<{ent_high:.3f}, high>= {ent_high:.3f}"
    )
    print(
        f"  coherence cuts: low<{coh_low:.3f}, mid<{coh_high:.3f}, high>= {coh_high:.3f}"
    )

    for label in ["NARROW", "AMBIGUOUS", "BROAD", "REJECT"]:
        subset = [m for m in all_results if m["expected"] == label]
        if not subset:
            continue

        regions = defaultdict(int)
        for m in subset:
            ent_bucket = _bucket(m.get("ent_score_T0.1", 0.0), ent_low, ent_high)
            coh_bucket = _bucket(m.get("semantic_coherence_top5", 0.0), coh_low, coh_high)
            regions[(ent_bucket, coh_bucket)] += 1

        avg_entropy = np.mean([m.get("ent_score_T0.1", 0.0) for m in subset])
        avg_coherence = np.mean([m.get("semantic_coherence_top5", 0.0) for m in subset])
        dominant = sorted(regions.items(), key=lambda item: (-item[1], item[0]))[:3]
        dominant_text = ", ".join(
            f"{ent}/{coh}:{count}" for (ent, coh), count in dominant
        ) or "-"
        print(
            f"    {label:<10} avg_entropy={avg_entropy:.4f}  "
            f"avg_coherence={avg_coherence:.4f}  top_regions=[{dominant_text}]"
        )


def run_error_analysis(use_prf=False, label="BASELINE"):
    print(f"  ERROR ANALYSIS - {label} - Recall@5 | MRR | failure categorisation")

    test_queries = _load_test_queries()
    if not test_queries:
        print(
            "  No diagnostics query set found. Add labeled queries to "
            f"{DEFAULT_TEST_QUERIES_FILE} and re-run."
        )
        return None

    thread_corpus = _load_thread_corpus()
    all_results = _evaluate_queries(test_queries, thread_corpus, use_prf=use_prf)
    if not all_results:
        print("  No diagnostics results produced.")
        return None

    print("  Training logistic regression (leave-one-out)...\n")

    hdr = (
        f"  {'Query':<35} {'Expected':<10} {'Rule':<10} {'LR':<10} "
        f"{'Recall@5':>8} {'MRR':>6}  {'Exp.Threads':<18} {'Top-5 Threads':<18}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for m in all_results:
        r5 = f"{m['recall_5']}" if m["recall_5"] is not None else "N/A"
        mrr = f"{m['mrr']:.2f}" if m["mrr"] is not None else "N/A"
        exp_t = ",".join(_tid_label(t) for t in m["expected_threads"]) or "-"
        top5 = ",".join(_tid_label(t) for t in m["ranked_threads"][:5])
        print(
            f"  {m['query'][:33]:<35} {m['expected']:<10} {m['predicted']:<10} "
            f"{m['lr_predicted']:<10} {r5:>8} {mrr:>6}  {exp_t:<18} {top5:<18}"
        )

    if use_prf:
        print("\n  PRF expansions:")
        for m in all_results:
            debug = m.get("prf_debug") or {}
            print(f"  Original query : {debug.get('original_query', m['query'])}")
            print(f"  Apply PRF      : {debug.get('apply_prf', False)}")
            print(f"  Trigger entropy: {debug.get('trigger_entropy', 'N/A')}")
            print(f"  Trigger rel_gap: {debug.get('trigger_rel_gap', 'N/A')}")
            print(f"  Domain conf    : {debug.get('domain_confidence', 'N/A')}")
            print(f"  Tech ratio     : {debug.get('tech_ratio', 'N/A')}")
            print(f"  Support ratio  : {debug.get('support_ratio', 'N/A')}")
            print(f"  OOD ratio      : {debug.get('ood_ratio', 'N/A')}")
            print(f"  Mixed domain   : {debug.get('mixed_domain', 'N/A')}")
            print(f"  Domain blocked : {debug.get('blocked_by_domain_gate', 'N/A')}")
            print(f"  Supported terms: {debug.get('supported_terms', [])}")
            print(f"  Unsupported terms: {debug.get('unsupported_terms', [])}")
            print(f"  Expansion terms: {debug.get('expansion_terms', [])}")
            print(f"  Expanded queries: {debug.get('expanded_queries', [])}")

    summary = _summarize_results(all_results)

    has_threads = [m for m in all_results if m["recall_5"] is not None]
    print(f"\n  AGGREGATE (queries with expected threads, n={len(has_threads)}):")
    print(f"    Recall@5 = {summary['recall@5']:.2%}")
    print(f"    MRR      = {summary['mrr']:.4f}")

    for label in ["NARROW", "AMBIGUOUS", "BROAD"]:
        subset = [m for m in all_results if m["expected"] == label and m["recall_5"] is not None]
        if not subset:
            continue
        r = np.mean([m["recall_5"] for m in subset])
        mrr_v = np.mean([m["mrr"] for m in subset])
        print(f"    {label:<10}  Recall@5={r:.2%}  MRR={mrr_v:.4f}  (n={len(subset)})")

    _print_error_cases(all_results, pred_key="predicted", name="RULE-BASED")
    _print_geometric_view(all_results, pred_key="predicted", name="RULE-BASED")
    _print_error_cases(all_results, pred_key="lr_predicted", name="LOGREG (LOO)")
    _print_geometric_view(all_results, pred_key="lr_predicted", name="LOGREG (LOO)")
    return summary


def _print_error_cases(all_results, pred_key, name):
    print(f"  ERROR CASES - {name}")

    wrong = [m for m in all_results if m[pred_key] != m["expected"]]
    case1, case2 = [], []

    for m in wrong:
        if m["recall_5"] is None:
            case2.append(m)
            continue
        if m["recall_5"] == 0:
            case1.append(m)
        else:
            case2.append(m)

    print(f"\n  CASE 1 - Retrieval failed ({len(case1)} queries)")
    print("  (expected thread NOT in top-5 -> classifier had no chance)")
    if case1:
        print(
            f"  {'Query':<35} {'Expected':<10} {'Predicted':<10} "
            f"{'MRR':>6}  {'Exp.Threads':<16} {'Top-5':<18}"
        )
        for m in case1:
            exp_t = ",".join(_tid_label(t) for t in m["expected_threads"]) or "-"
            top5 = ",".join(_tid_label(t) for t in m["ranked_threads"][:5])
            mrr = f"{m['mrr']:.2f}" if m["mrr"] is not None else "N/A"
            print(
                f"  {m['query'][:33]:<35} {m['expected']:<10} {m[pred_key]:<10} "
                f"{mrr:>6}  {exp_t:<16} {top5:<18}"
            )
    else:
        print("  (none)")

    print(f"\n  CASE 2 - Retrieval correct, classifier failed ({len(case2)} queries)")
    print("  (expected thread IS in top-5, but label was wrong)")
    if case2:
        print(
            f"  {'Query':<35} {'Expected':<10} {'Predicted':<10} "
            f"{'MRR':>6}  {'Exp.Threads':<16} {'Top-5':<18}"
        )
        for m in case2:
            exp_t = ",".join(_tid_label(t) for t in m["expected_threads"]) or "-"
            top5 = ",".join(_tid_label(t) for t in m["ranked_threads"][:5])
            mrr = f"{m['mrr']:.2f}" if m["mrr"] is not None else "N/A"
            print(
                f"  {m['query'][:33]:<35} {m['expected']:<10} {m[pred_key]:<10} "
                f"{mrr:>6}  {exp_t:<16} {top5:<18}"
            )
        print("\n  Feature details:")
        for m in case2:
            print(f"  Query: {m['query']}")
            print(
                "    "
                f"signal={m.get('signal', '?')}  "
                f"std_distance={m.get('std_distance', '?')}  "
                f"signal_norm={m.get('signal_norm', '?')}  "
                f"abs_ratio={m.get('abs_ratio', '?')}  "
                f"coherence={m.get('semantic_coherence_top5', '?')}"
            )
            print(
                "    "
                f"rel_gap={m.get('rel_gap', '?')}  "
                f"gap_score={m.get('gap_score', '?')}  "
                f"gap_dist={m.get('gap_dist', '?')}  "
                f"spread={m.get('spread', '?')}"
            )
            print(
                "    "
                f"entropy={m.get('ent_score_T0.1', '?')}  "
                f"n_threads={m.get('n_threads', '?')}  "
                f"top3={m.get('top3_threads', '?')}"
            )
    else:
        print("  (none)")

    total_wrong = len(wrong)
    print(f"\n  SUMMARY - {name}:")
    print(f"    Total errors:                 {total_wrong}/{len(all_results)}")
    if total_wrong:
        print(
            f"    CASE 1 (retrieval failed):    {len(case1)}/{total_wrong} "
            f"({len(case1) / total_wrong * 100:.0f}% of errors)"
        )
        print(
            f"    CASE 2 (classifier failed):   {len(case2)}/{total_wrong} "
            f"({len(case2) / total_wrong * 100:.0f}% of errors)"
        )


if __name__ == "__main__":
    baseline = run_error_analysis(use_prf=False, label="BASELINE")
    print("\n" + "  " + "=" * 74 + "\n")
    prf = run_error_analysis(use_prf=True, label="PRF")

    if baseline and prf:
        print("\n  COMPARISON")
        print(f"  {'Metric':<20} {'Baseline':<12} {'PRF':<12}")
        print("  " + "-" * 46)
        print(f"  {'Queries':<20} {baseline['queries']:<12} {prf['queries']:<12}")
        print(f"  {'Recall@5':<20} {baseline['recall@5']:.2%}       {prf['recall@5']:.2%}")
        print(f"  {'MRR':<20} {baseline['mrr']:.4f}       {prf['mrr']:.4f}")
        print(f"  {'Rule errors':<20} {baseline['rule_errors']:<12} {prf['rule_errors']:<12}")
        print(f"  {'LR errors':<20} {baseline['lr_errors']:<12} {prf['lr_errors']:<12}")
