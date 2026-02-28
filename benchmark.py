import sys, math
import numpy as np
from memory.storage import collection
from memory.retrieval import retrieve_candidates
from memory.intent import analyze_query_intent


def _group_threads(candidates):
    threads = {}
    for c in candidates:
        tid = c['metadata']['thread_id']
        threads.setdefault(tid, {'candidates': [], 'distances': []})
        threads[tid]['candidates'].append(c)
        threads[tid]['distances'].append(c['distance'])

    aggregates = []
    for tid, td in threads.items():
        aggregates.append({
            'thread_id': tid,
            'avg_distance': np.mean(td['distances']),
            'min_distance': np.min(td['distances']),
            'message_count': len(td['candidates']),
        })
    aggregates.sort(key=lambda x: x['min_distance'])
    return aggregates


def _all_distances(candidates):
    return np.array([c['distance'] for c in candidates])


# Formula A — Score + Gap + Signal + Absolute Distance Check
ALPHA_A           = 0.25
REL_GAP_THRESH_A  = 0.20
MIN_SIGNAL_A      = 0.20
ABS_REJECT_A      = 0.80
MAX_BROAD_A       = 3
MIN_THREAD_SIZE_A = 2

def formula_a(candidates):
    agg = _group_threads(candidates)
    dists = _all_distances(candidates)
    mean_d = float(np.mean(dists))
    std_d  = float(np.std(dists))

    for t in agg:
        t['thread_score'] = t['avg_distance'] - math.log(t['message_count'] + 1) * ALPHA_A
    agg.sort(key=lambda x: x['thread_score'])

    multi = [t for t in agg if t['message_count'] >= MIN_THREAD_SIZE_A]
    if multi:
        agg = multi

    best = agg[0]
    signal = mean_d - best['min_distance']
    abs_ratio = best['min_distance'] / mean_d if mean_d > 0 else 1.0

    diag = {
        'best_distance': round(best['min_distance'], 4),
        'mean_distance': round(mean_d, 4),
        'std_distance':  round(std_d, 4),
        'z_score':       round(signal / std_d, 4) if std_d > 0 else 0.0,
        'gap':           None,
        'signal':        round(signal, 4),
        'abs_ratio':     round(abs_ratio, 4),
    }

    if abs_ratio > ABS_REJECT_A:
        diag['gap'] = '-'
        diag['decision'] = 'REJECT'
        diag['reason'] = f'abs_ratio {abs_ratio:.3f} > {ABS_REJECT_A}'
        diag['thread_ids'] = []
        return diag

    if len(agg) < 2:
        diag['gap'] = '-'
        if signal < MIN_SIGNAL_A:
            diag['decision'] = 'REJECT'
            diag['reason'] = f'signal {signal:.3f} < {MIN_SIGNAL_A}'
        else:
            diag['decision'] = 'NARROW'
            diag['reason'] = 'single thread, passes signal'
        diag['thread_ids'] = [best['thread_id']]
        return diag

    second = agg[1]
    gap    = second['thread_score'] - best['thread_score']
    spread = agg[-1]['thread_score'] - best['thread_score']
    rel_gap = gap / spread if spread > 0 else 1.0
    diag['gap'] = round(gap, 4)
    diag['rel_gap'] = round(rel_gap, 4)

    if signal < MIN_SIGNAL_A:
        diag['decision'] = 'REJECT'
        diag['reason'] = f'signal {signal:.3f} < {MIN_SIGNAL_A}'
        diag['thread_ids'] = []
    elif rel_gap < REL_GAP_THRESH_A:
        diag['decision'] = 'BROAD'
        diag['reason'] = f'rel_gap {rel_gap:.3f} < {REL_GAP_THRESH_A}'
        diag['thread_ids'] = [t['thread_id'] for t in agg[:MAX_BROAD_A]]
    else:
        diag['decision'] = 'NARROW'
        diag['reason'] = f'rel_gap={rel_gap:.3f}, signal={signal:.3f}'
        diag['thread_ids'] = [best['thread_id']]
    return diag


# Formula B — Z-Score + Absolute Distance Quality
ABS_REJECT_B = 0.80
Z_REJECT_B   = 1.5
Z_BROAD_B    = 0.5
MAX_BROAD_B  = 3

def formula_b(candidates):
    agg = _group_threads(candidates)
    dists = _all_distances(candidates)
    mean_d = float(np.mean(dists))
    std_d  = float(np.std(dists))

    best = agg[0]
    z_score   = (mean_d - best['min_distance']) / std_d if std_d > 0 else 0.0
    abs_ratio = best['min_distance'] / mean_d if mean_d > 0 else 1.0

    diag = {
        'best_distance': round(best['min_distance'], 4),
        'mean_distance': round(mean_d, 4),
        'std_distance':  round(std_d, 4),
        'z_score':       round(z_score, 4),
        'gap':           None,
        'signal':        round(mean_d - best['min_distance'], 4),
        'abs_ratio':     round(abs_ratio, 4),
    }

    if abs_ratio > ABS_REJECT_B:
        diag['gap'] = '-'
        diag['decision'] = 'REJECT'
        diag['reason'] = f'abs_ratio {abs_ratio:.3f} > {ABS_REJECT_B}'
        diag['thread_ids'] = []
        return diag

    if len(agg) < 2:
        diag['gap'] = '-'
        if z_score < Z_REJECT_B:
            diag['decision'] = 'REJECT'
            diag['reason'] = f'z {z_score:.2f} < {Z_REJECT_B}'
        else:
            diag['decision'] = 'NARROW'
            diag['reason'] = f'z={z_score:.2f}'
        diag['thread_ids'] = [best['thread_id']]
        return diag

    second = agg[1]
    gap_dist = second['min_distance'] - best['min_distance']
    gap_z    = gap_dist / std_d if std_d > 0 else 0.0
    diag['gap']   = round(gap_dist, 4)
    diag['gap_z'] = round(gap_z, 4)

    if z_score < Z_REJECT_B:
        diag['decision'] = 'REJECT'
        diag['reason'] = f'z {z_score:.2f} < {Z_REJECT_B}'
        diag['thread_ids'] = []
    elif gap_z < Z_BROAD_B:
        diag['decision'] = 'BROAD'
        diag['reason'] = f'gap_z {gap_z:.3f} < {Z_BROAD_B}'
        diag['thread_ids'] = [t['thread_id'] for t in agg[:MAX_BROAD_B]]
    else:
        diag['decision'] = 'NARROW'
        diag['reason'] = f'z={z_score:.2f}, gap_z={gap_z:.3f}'
        diag['thread_ids'] = [best['thread_id']]
    return diag


# Formula C — Percentile-Ratio (best/p75)
RATIO_REJECT_C = 0.85
RATIO_BROAD_C  = 0.45
MAX_BROAD_C    = 3

def formula_c(candidates):
    agg = _group_threads(candidates)
    dists = _all_distances(candidates)
    mean_d = float(np.mean(dists))
    std_d  = float(np.std(dists))
    p75    = float(np.percentile(dists, 75))

    best = agg[0]
    ratio     = best['min_distance'] / p75 if p75 > 0 else 0.0
    abs_ratio = best['min_distance'] / mean_d if mean_d > 0 else 1.0

    diag = {
        'best_distance': round(best['min_distance'], 4),
        'mean_distance': round(mean_d, 4),
        'std_distance':  round(std_d, 4),
        'z_score':       round((mean_d - best['min_distance']) / std_d, 4) if std_d > 0 else 0.0,
        'gap':           None,
        'signal':        round(mean_d - best['min_distance'], 4),
        'p75':           round(p75, 4),
        'ratio':         round(ratio, 4),
        'abs_ratio':     round(abs_ratio, 4),
    }

    if ratio > RATIO_REJECT_C:
        diag['gap'] = '-'
        diag['decision'] = 'REJECT'
        diag['reason'] = f'ratio {ratio:.3f} > {RATIO_REJECT_C}'
        diag['thread_ids'] = []
        return diag

    if len(agg) < 2:
        diag['gap'] = '-'
        diag['decision'] = 'NARROW'
        diag['reason'] = f'ratio={ratio:.3f}'
        diag['thread_ids'] = [best['thread_id']]
        return diag

    second = agg[1]
    gap_dist = second['min_distance'] - best['min_distance']
    second_ratio = second['min_distance'] / p75 if p75 > 0 else 0.0
    diag['gap'] = round(gap_dist, 4)
    diag['second_ratio'] = round(second_ratio, 4)

    if ratio > RATIO_BROAD_C:
        diag['decision'] = 'BROAD'
        diag['reason'] = f'ratio {ratio:.3f} > {RATIO_BROAD_C} (moderate relevance)'
        diag['thread_ids'] = [t['thread_id'] for t in agg[:MAX_BROAD_C]]
    elif (second_ratio - ratio) < 0.15:
        diag['decision'] = 'BROAD'
        diag['reason'] = f'second_ratio - ratio = {second_ratio - ratio:.3f} < 0.15 (close threads)'
        diag['thread_ids'] = [t['thread_id'] for t in agg[:MAX_BROAD_C]]
    else:
        diag['decision'] = 'NARROW'
        diag['reason'] = f'ratio={ratio:.3f}, gap clear'
        diag['thread_ids'] = [best['thread_id']]

    return diag


TEST_QUERIES = [
    ("Login configuration issue",      "NARROW",  1773000000.0,  "slightly related → Thread 1 (OAuth login)"),
    ("OAuth redirect problem",         "NARROW",  1773000000.0,  "slightly related → Thread 1 (OAuth redirect URI)"),
    ("Authentication failure",         "NARROW",  1773000000.0,  "slightly related → Thread 1 (401 Unauthorized)"),
    ("API improvements",               "BROAD",   None,          "weak/vague — multiple API threads match"),
    ("System upgrades",                "BROAD",   None,          "weak/vague — deployment + release threads"),
    ("New features discussion",        "REJECT",  None,          "too vague — no thread about new features"),
    ("API issue after deployment",     "BROAD",   None,          "mixed — Thread 2 + Thread 4"),
    ("Bugs and performance problems",  "BROAD",   None,          "mixed — Thread 6 + Thread 5"),
    ("Release performance regression", "NARROW",  1773400000.0,  "mixed but specific — Thread 5"),
]


FORMULAS = {
    'A (Score+Gap+Signal)': formula_a,
    'B (Z-Score)':          formula_b,
    'C (Pct-Ratio)':        formula_c,
}

COL_W = 30


def _tid_label(tid):
    return '-' if tid is None else str(int(tid))


def run_benchmark():
    print("\n" + "=" * 110)
    print("  RANKING FORMULA BENCHMARK")
    print("=" * 110)

    results = []

    for query, expected, expected_tid, desc in TEST_QUERIES:
        intent = analyze_query_intent(query)
        candidates = retrieve_candidates(query, intent, with_filter=False)

        if not candidates:
            print(f"\n⚠  Query '{query}' — 0 candidates, skipping.")
            results.append({
                'query': query, 'expected': expected, 'desc': desc,
                'A': 'NO_CAND', 'B': 'NO_CAND', 'C': 'NO_CAND',
            })
            continue

        diag_a = formula_a(candidates)
        diag_b = formula_b(candidates)
        diag_c = formula_c(candidates)

        print(f"\n{'─' * 110}")
        print(f"  Query: \"{query}\"")
        print(f"  Expected: {expected}  ({desc})")
        print(f"{'─' * 110}")

        header = f"  {'Metric':<22} {'A (Score+Gap+Signal)':>{COL_W}} {'B (Z-Score)':>{COL_W}} {'C (Pct-Ratio)':>{COL_W}}"
        print(header)
        print(f"  {'─'*22} {'─'*COL_W} {'─'*COL_W} {'─'*COL_W}")

        for m in ['best_distance', 'mean_distance', 'std_distance', 'z_score', 'gap', 'signal']:
            va, vb, vc = diag_a.get(m, '-'), diag_b.get(m, '-'), diag_c.get(m, '-')
            print(f"  {m:<22} {str(va):>{COL_W}} {str(vb):>{COL_W}} {str(vc):>{COL_W}}")

        extras_a = f"rel_gap={diag_a.get('rel_gap', '-')}, abs_r={diag_a.get('abs_ratio', '-')}"
        extras_b = f"gap_z={diag_b.get('gap_z', '-')}, abs_r={diag_b.get('abs_ratio', '-')}"
        extras_c = f"ratio={diag_c.get('ratio', '-')}, p75={diag_c.get('p75', '-')}"
        print(f"  {'formula-specific':<22} {extras_a:>{COL_W}} {extras_b:>{COL_W}} {extras_c:>{COL_W}}")

        ra, rb, rc = diag_a.get('reason', '-'), diag_b.get('reason', '-'), diag_c.get('reason', '-')
        print(f"  {'reason':<22} {ra:>{COL_W}} {rb:>{COL_W}} {rc:>{COL_W}}")

        def _mark(dec, exp):
            return f"✅ {dec}" if dec == exp else f"❌ {dec}"

        print(f"  {'DECISION':<22} {_mark(diag_a['decision'], expected):>{COL_W}} {_mark(diag_b['decision'], expected):>{COL_W}} {_mark(diag_c['decision'], expected):>{COL_W}}")

        tids_a = ', '.join(_tid_label(t) for t in diag_a.get('thread_ids', []))
        tids_b = ', '.join(_tid_label(t) for t in diag_b.get('thread_ids', []))
        tids_c = ', '.join(_tid_label(t) for t in diag_c.get('thread_ids', []))
        print(f"  {'thread_ids':<22} {tids_a:>{COL_W}} {tids_b:>{COL_W}} {tids_c:>{COL_W}}")

        results.append({
            'query': query, 'expected': expected, 'desc': desc,
            'A': diag_a['decision'], 'B': diag_b['decision'], 'C': diag_c['decision'],
            'A_tid': diag_a.get('thread_ids', []),
            'B_tid': diag_b.get('thread_ids', []),
            'C_tid': diag_c.get('thread_ids', []),
            'expected_tid': expected_tid,
        })

    # Summary table
    print("\n\n" + "=" * 110)
    print("  SUMMARY TABLE")
    print("=" * 110)

    qw, fw = 36, 20
    print(f"  {'Query':<{qw}} {'Expected':<10} {'Formula A':<{fw}} {'Formula B':<{fw}} {'Formula C':<{fw}}")
    print(f"  {'─'*qw} {'─'*10} {'─'*fw} {'─'*fw} {'─'*fw}")

    for r in results:
        def _cell(dec, exp, tids, exp_tid):
            sym = '✅' if dec == exp else '❌'
            tid_ok = ''
            if exp_tid and tids:
                tid_ok = ' 🎯' if exp_tid in tids else ' ⚠️'
            return f"{sym} {dec}{tid_ok}"

        ca = _cell(r['A'], r['expected'], r.get('A_tid', []), r.get('expected_tid'))
        cb = _cell(r['B'], r['expected'], r.get('B_tid', []), r.get('expected_tid'))
        cc = _cell(r['C'], r['expected'], r.get('C_tid', []), r.get('expected_tid'))
        print(f"  {r['query'][:qw-2]:<{qw}} {r['expected']:<10} {ca:<{fw}} {cb:<{fw}} {cc:<{fw}}")

    # Scoring
    print("\n\n" + "=" * 110)
    print("  SCORING")
    print("=" * 110)

    for label in ['A', 'B', 'C']:
        total = len(results)
        correct = sum(1 for r in results if r[label] == r['expected'])

        narrow_qs  = [r for r in results if r['expected'] == 'NARROW']
        reject_qs  = [r for r in results if r['expected'] == 'REJECT']
        broad_qs   = [r for r in results if r['expected'] == 'BROAD']

        recall    = sum(1 for r in narrow_qs if r[label] == 'NARROW') / len(narrow_qs) if narrow_qs else 0
        precision = sum(1 for r in reject_qs if r[label] == 'REJECT') / len(reject_qs) if reject_qs else 0
        ambiguity = sum(1 for r in broad_qs  if r[label] == 'BROAD')  / len(broad_qs)  if broad_qs  else 0

        narrow_correct = [r for r in narrow_qs if r[label] == 'NARROW']
        right_thread = sum(
            1 for r in narrow_correct
            if r.get('expected_tid') and r.get(f'{label}_tid') and r['expected_tid'] in r[f'{label}_tid']
        )
        thread_acc = right_thread / len(narrow_correct) if narrow_correct else 0

        print(f"\n  Formula {label}:")
        print(f"    Overall accuracy : {correct}/{total} ({correct/total*100:.0f}%)")
        print(f"    Recall  (strong) : {recall*100:.0f}%  ({sum(1 for r in narrow_qs if r[label]=='NARROW')}/{len(narrow_qs)})")
        print(f"    Precision(reject): {precision*100:.0f}%  ({sum(1 for r in reject_qs if r[label]=='REJECT')}/{len(reject_qs)})")
        print(f"    Ambiguity(broad) : {ambiguity*100:.0f}%  ({sum(1 for r in broad_qs if r[label]=='BROAD')}/{len(broad_qs)})")
        print(f"    Thread accuracy  : {thread_acc*100:.0f}%  ({right_thread}/{len(narrow_correct)})")

    print("\n" + "=" * 110)
    print("  ✅ = correct   ❌ = wrong   🎯 = correct thread   ⚠️ = wrong thread")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        print("Re-seeding database...")
        from seed_db import seed
        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])
            print(f"  Cleared {len(existing['ids'])} existing documents.")
        seed()
        print("  Seeding complete.\n")

    run_benchmark()
