import re, sys, math, json
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
    _group_threads, run_logreg_loo, extract_features, FEATURE_KEYS,
    MIN_THREAD_SIZE,
)

DEFAULT_TEST_QUERIES_FILE = REPO_ROOT / "config" / "diagnostics_queries.json"


THREAD_MAP = {
    1:  "1773000000",   # Google OAuth Bug
    2:  "1773100000",   # General API Issues
    3:  "1773200000",   # API Documentation
    4:  "1773300000",   # Deployment v3.0
    5:  "1773400000",   # Dashboard Performance
    6:  "1773500000",   # Multi-Bug Release
    7:  "1773600000",   # CI/CD Pipeline
    8:  "1773700000",   # Pagination
    9:  "1773800000",   # Caching / Redis
    10: "1773900000",   # Team Offsite
    11: "1774000000",   # Database Migration
    12: "1774100000",   # Security Vulnerability
}
REVERSE_MAP = {v: f"T{k}" for k, v in THREAD_MAP.items()}

def _tid_label(ts):
    """Convert a thread_ts like '1773000000' or 1773000000.0 to 'T1'."""
    key = str(int(float(ts)))            
    return REVERSE_MAP.get(key, key)


def _parse_expected_threads(desc, expected_label):
    if expected_label == "REJECT":
        return []                       
    nums = sorted(set(int(x) for x in re.findall(r'T(\d+)', desc)))
    return [THREAD_MAP[n] for n in nums if n in THREAD_MAP]


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
    agg_sorted = sorted(agg, key=lambda x: x['thread_score'])
    multi = [t for t in agg_sorted if t['message_count'] >= MIN_THREAD_SIZE]
    if multi:
        agg_sorted = multi
    return [str(int(float(t['thread_id']))) for t in agg_sorted]


def _load_test_queries(path=DEFAULT_TEST_QUERIES_FILE):
    if not path.exists():
        return []

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")

    queries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        query = str(item.get('query', '')).strip()
        expected = str(item.get('expected', '')).strip().upper()
        desc = str(item.get('desc', '')).strip()
        if not query or not expected:
            continue
        queries.append((query, expected, desc))
    return queries

def run_error_analysis():
    print("  ERROR ANALYSIS - Recall@5 | MRR | failure categorisation")

    test_queries = _load_test_queries()
    if not test_queries:
        print(
            "  No diagnostics query set found. Add labeled queries to "
            f"{DEFAULT_TEST_QUERIES_FILE} and re-run."
        )
        return

    all_results = []

    for query, expected, desc in test_queries:
        intent = analyze_query_intent(query)
        candidates = retrieve_candidates(query, intent, with_filter=False)
        if not candidates:
            continue

        m = compute_metrics(candidates)
        m['query']    = query
        m['expected'] = expected
        m['desc']     = desc

        expected_tids = _parse_expected_threads(desc, expected)
        ranked_tids   = _rank_threads(candidates)

        m['expected_threads'] = expected_tids
        m['ranked_threads']   = ranked_tids
        m['recall_5']         = _recall_at_k(ranked_tids, expected_tids, k=5)
        m['mrr']              = _mrr(ranked_tids, expected_tids)

        all_results.append(m)

    pop_stats = compute_population_stats(all_results)
    for m in all_results:
        m['predicted'] = decide(m, pop_stats)
        m['correct']   = m['predicted'] == m['expected']

    print("  Training logistic regression (leave-one-out)...\n")
    lr_preds = run_logreg_loo(all_results)
    for i, m in enumerate(all_results):
        m['lr_predicted'] = lr_preds[i]
        m['lr_correct']   = lr_preds[i] == m['expected']

    hdr = (f"  {'Query':<35} {'Expected':<10} {'Rule':<10} {'LR':<10} "
           f"{'Recall@5':>8} {'MRR':>6}  {'Exp.Threads':<18} {'Top-5 Threads':<18}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for m in all_results:
        r5  = f"{m['recall_5']}" if m['recall_5'] is not None else "N/A"
        mrr = f"{m['mrr']:.2f}" if m['mrr'] is not None else "N/A"
        exp_t = ",".join(_tid_label(t) for t in m['expected_threads']) or "-"
        top5  = ",".join(_tid_label(t) for t in m['ranked_threads'][:5])
        print(f"  {m['query'][:33]:<35} {m['expected']:<10} {m['predicted']:<10} "
              f"{m['lr_predicted']:<10} {r5:>8} {mrr:>6}  {exp_t:<18} {top5:<18}")

    has_threads = [m for m in all_results if m['recall_5'] is not None]
    avg_recall = np.mean([m['recall_5'] for m in has_threads])
    avg_mrr    = np.mean([m['mrr']      for m in has_threads])

    print(f"\n  AGGREGATE (queries with expected threads, n={len(has_threads)}):")
    print(f"    Recall@5 = {avg_recall:.2%}")
    print(f"    MRR      = {avg_mrr:.4f}")

    for label in ['NARROW', 'AMBIGUOUS', 'BROAD']:
        subset = [m for m in all_results if m['expected'] == label and m['recall_5'] is not None]
        if not subset:
            continue
        r = np.mean([m['recall_5'] for m in subset])
        mrr_v = np.mean([m['mrr'] for m in subset])
        print(f"    {label:<10}  Recall@5={r:.2%}  MRR={mrr_v:.4f}  (n={len(subset)})")

    _print_error_cases(all_results, pred_key='predicted', name='RULE-BASED')

    _print_error_cases(all_results, pred_key='lr_predicted', name='LOGREG (LOO)')


def _print_error_cases(all_results, pred_key, name):
    print(f"  ERROR CASES - {name}")

    wrong = [m for m in all_results if m[pred_key] != m['expected']]
    case1, case2 = [], []

    for m in wrong:
        if m['recall_5'] is None:
            case2.append(m)
            continue
        if m['recall_5'] == 0:
            case1.append(m)
        else:
            case2.append(m)

    print(f"\n  CASE 1 - Retrieval failed ({len(case1)} queries)")
    print("  (expected thread NOT in top-5 -> classifier had no chance)")
    if case1:
        print(f"  {'Query':<35} {'Expected':<10} {'Predicted':<10} {'MRR':>6}  {'Exp.Threads':<16} {'Top-5':<18}")
        for m in case1:
            exp_t = ",".join(_tid_label(t) for t in m['expected_threads']) or "-"
            top5  = ",".join(_tid_label(t) for t in m['ranked_threads'][:5])
            mrr   = f"{m['mrr']:.2f}" if m['mrr'] is not None else "N/A"
            print(f"  {m['query'][:33]:<35} {m['expected']:<10} {m[pred_key]:<10} "
                  f"{mrr:>6}  {exp_t:<16} {top5:<18}")
    else:
        print("  (none)")

    print(f"\n  CASE 2 - Retrieval correct, classifier failed ({len(case2)} queries)")
    print("  (expected thread IS in top-5, but label was wrong)")
    if case2:
        print(f"  {'Query':<35} {'Expected':<10} {'Predicted':<10} {'MRR':>6}  {'Exp.Threads':<16} {'Top-5':<18}")
        for m in case2:
            exp_t = ",".join(_tid_label(t) for t in m['expected_threads']) or "-"
            top5  = ",".join(_tid_label(t) for t in m['ranked_threads'][:5])
            mrr   = f"{m['mrr']:.2f}" if m['mrr'] is not None else "N/A"
            print(f"  {m['query'][:33]:<35} {m['expected']:<10} {m[pred_key]:<10} "
                  f"{mrr:>6}  {exp_t:<16} {top5:<18}")
    else:
        print("  (none)")

    total_wrong = len(wrong)
    print(f"\n  SUMMARY - {name}:")
    print(f"    Total errors:                 {total_wrong}/{len(all_results)}")
    if total_wrong:
        print(f"    CASE 1 (retrieval failed):    {len(case1)}/{total_wrong} "
              f"({len(case1)/total_wrong*100:.0f}% of errors)")
        print(f"    CASE 2 (classifier failed):   {len(case2)}/{total_wrong} "
              f"({len(case2)/total_wrong*100:.0f}% of errors)")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        print("Re-seeding database...")
        from seed_db import seed
        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])
        seed()
        print("  Seeding complete.\n")

    run_error_analysis()
