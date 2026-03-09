import sys, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from memory.storage import collection
from memory.retrieval import retrieve_candidates
from memory.intent import analyze_query_intent

ALPHA = 0.25
ENTROPY_TEMP = 0.1
MIN_THREAD_SIZE = 2

# ── Z-score thresholds (in std-dev units from population mean) ──
Z_REL_GAP_HIGH   =  0.50   # NARROW: z(rel_gap) above this
Z_ENTROPY_LOW    = -0.50   # NARROW: z(entropy) below this
Z_REL_GAP_AMB_LO = -0.80   # AMBIGUOUS: z(rel_gap) above this
Z_REL_GAP_AMB_HI =  1.00   # AMBIGUOUS: z(rel_gap) below this
Z_ENTROPY_AMB_LO = -1.00   # AMBIGUOUS: z(entropy) above this
Z_ENTROPY_AMB_HI =  0.50   # AMBIGUOUS: z(entropy) below this
Z_SIGNAL_BROAD   =  1.50   # BROAD: z(signal_norm) above this


def _group_threads(candidates):
    threads = {}
    for c in candidates:
        tid = c['metadata']['thread_id']
        threads.setdefault(tid, {'candidates': [], 'distances': []})
        threads[tid]['candidates'].append(c)
        threads[tid]['distances'].append(c['distance'])

    aggregates = []
    for tid, td in threads.items():
        avg_d = float(np.mean(td['distances']))
        aggregates.append({
            'thread_id': tid,
            'avg_distance': avg_d,
            'min_distance': float(np.min(td['distances'])),
            'message_count': len(td['candidates']),
            'thread_score': avg_d - math.log(len(td['candidates']) + 1) * ALPHA,
        })
    return aggregates


def _softmax_entropy(values, temp):
    v = np.array(values)
    neg_over_T = -v / temp
    neg_over_T -= neg_over_T.max()
    exp_s = np.exp(neg_over_T)
    probs = exp_s / exp_s.sum()
    ent = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_ent = float(np.log2(len(v))) if len(v) > 1 else 1.0
    return round(ent / max_ent, 4)


def compute_metrics(candidates):
    agg = _group_threads(candidates)

    thread_dists = np.array([t['min_distance'] for t in agg])
    mean_d = float(np.mean(thread_dists))
    std_d  = float(np.std(thread_dists))
    var_d  = float(np.var(thread_dists))
    p75    = float(np.percentile(thread_dists, 75))

    agg_by_score = sorted(agg, key=lambda x: x['thread_score'])
    multi = [t for t in agg_by_score if t['message_count'] >= MIN_THREAD_SIZE]
    if multi:
        agg_by_score = multi

    best = agg_by_score[0]

    signal   = mean_d - best['min_distance']
    abs_ratio = best['min_distance'] / mean_d if mean_d > 0 else 1.0
    z_score  = signal / std_d if std_d > 0 else 0.0
    ratio_p75 = best['min_distance'] / p75 if p75 > 0 else 0.0

    m = {
        'best_thread':   int(best['thread_id']),
        'best_msgs':     best['message_count'],
        'best_distance': round(best['min_distance'], 4),
        'best_avg_dist': round(best['avg_distance'], 4),
        'best_score':    round(best['thread_score'], 4),
        'mean_distance': round(mean_d, 4),
        'std_distance':  round(std_d, 4),
        'variance':      round(var_d, 4),
        'p75':           round(p75, 4),
        'signal':        round(signal, 4),
        'abs_ratio':     round(abs_ratio, 4),
        'z_score':       round(z_score, 4),
        'signal_norm':   round(z_score, 4),
        'ratio_p75':     round(ratio_p75, 4),
        'n_threads':     len(agg),
    }

    if len(agg_by_score) >= 2:
        second = agg_by_score[1]
        gap_score = second['thread_score'] - best['thread_score']
        spread    = agg_by_score[-1]['thread_score'] - best['thread_score']
        gap_dist  = second['min_distance'] - best['min_distance']
        gap_z     = gap_dist / std_d if std_d > 0 else 0.0
        rel_gap   = gap_score / spread if spread > 0 else 1.0
        second_ratio = second['min_distance'] / p75 if p75 > 0 else 0.0

        m['2nd_thread']    = int(second['thread_id'])
        m['gap_score']     = round(gap_score, 4)
        m['gap_dist']      = round(gap_dist, 4)
        m['gap_z']         = round(gap_z, 4)
        m['rel_gap']       = round(rel_gap, 4)
        m['spread']        = round(spread, 4)
        m['2nd_ratio_p75'] = round(second_ratio, 4)
    else:
        m['2nd_thread'] = '-'
        m['gap_score'] = m['gap_dist'] = m['gap_z'] = m['rel_gap'] = m['spread'] = '-'
        m['2nd_ratio_p75'] = '-'

    thread_scores = [t['thread_score'] for t in agg_by_score]
    min_dists     = [t['min_distance'] for t in agg_by_score]
    avg_dists     = [t['avg_distance'] for t in agg_by_score]

    m['ent_score_T0.1']   = _softmax_entropy(thread_scores, 0.1)
    m['ent_score_T0.05']  = _softmax_entropy(thread_scores, 0.05)
    m['ent_score_T0.2']   = _softmax_entropy(thread_scores, 0.2)
    m['ent_mindist_T0.1'] = _softmax_entropy(min_dists, 0.1)
    m['ent_avgdist_T0.1'] = _softmax_entropy(avg_dists, 0.1)

    raw = np.array(min_dists)
    d_range = raw.max() - raw.min()
    normed = ((raw - raw.min()) / d_range).tolist() if d_range > 0 else [0.0] * len(raw)
    m['ent_mindist_norm_T0.1'] = _softmax_entropy(normed, 0.1)

    m['top3_threads'] = [int(t['thread_id']) for t in agg_by_score[:3]]

    top_k = len(candidates)
    thread_ids_in_topk = [c['metadata']['thread_id'] for c in candidates]
    unique_threads = len(set(thread_ids_in_topk))
    thread_concentration = 1.0 - (unique_threads / top_k) if top_k > 0 else 0.0

    m['unique_threads'] = unique_threads
    m['thread_concentration'] = round(thread_concentration, 4)

    return m


def compute_population_stats(all_metrics):
    rel_gaps, entropies, signal_norms = [], [], []
    for m in all_metrics:
        rg = m.get('rel_gap', None)
        if isinstance(rg, (int, float)):
            rel_gaps.append(rg)
        entropies.append(m['ent_score_T0.1'])
        sig = m['signal'] / m['std_distance'] if m['std_distance'] > 0 else 0.0
        signal_norms.append(sig)

    return {
        'rel_gap_mean':     float(np.mean(rel_gaps)) if rel_gaps else 0.0,
        'rel_gap_std':      float(np.std(rel_gaps))  if rel_gaps else 1.0,
        'entropy_mean':     float(np.mean(entropies)),
        'entropy_std':      float(np.std(entropies)) if len(entropies) > 1 else 1.0,
        'signal_norm_mean': float(np.mean(signal_norms)),
        'signal_norm_std':  float(np.std(signal_norms)) if len(signal_norms) > 1 else 1.0,
    }


def decide(m, pop_stats):
    """4-class decision with z-normalised thresholds: NARROW -> AMBIGUOUS -> BROAD -> REJECT."""
    signal_norm = m['signal'] / m['std_distance'] if m['std_distance'] > 0 else 0.0
    entropy = m['ent_score_T0.1']
    rg = m.get('rel_gap', 1.0)
    if not isinstance(rg, (int, float)):
        rg = 1.0

    # z-score normalisation using population stats
    ps = pop_stats
    z_rg  = (rg - ps['rel_gap_mean']) / ps['rel_gap_std'] if ps['rel_gap_std'] > 0 else 0.0
    z_ent = (entropy - ps['entropy_mean']) / ps['entropy_std'] if ps['entropy_std'] > 0 else 0.0
    z_sig = (signal_norm - ps['signal_norm_mean']) / ps['signal_norm_std'] if ps['signal_norm_std'] > 0 else 0.0

    # 1. rel_gap high AND entropy low -> NARROW
    if z_rg > Z_REL_GAP_HIGH and z_ent < Z_ENTROPY_LOW:
        return 'NARROW'

    # 2. rel_gap medium AND entropy medium -> AMBIGUOUS
    if Z_REL_GAP_AMB_LO < z_rg < Z_REL_GAP_AMB_HI and Z_ENTROPY_AMB_LO < z_ent < Z_ENTROPY_AMB_HI:
        return 'AMBIGUOUS'

    # 3. signal_norm high -> BROAD (relevant but spread across threads)
    if z_sig >= Z_SIGNAL_BROAD:
        return 'BROAD'

    # 4. fallthrough -> REJECT
    return 'REJECT'

FEATURE_KEYS = [
    'rel_gap',
    'entropy',
    'signal_norm',
    'abs_ratio',
    'thread_concentration',
]

def extract_features(m):
    """Pull numeric feature vector from a metrics dict."""
    sn = m['signal'] / m['std_distance'] if m['std_distance'] > 0 else 0.0
    rg = m.get('rel_gap', 1.0)
    if not isinstance(rg, (int, float)):
        rg = 1.0
    return [
        rg,                          # rel_gap
        m['ent_score_T0.1'],         # entropy
        sn,                          # signal_norm
        m['abs_ratio'],              # abs_ratio
        m['thread_concentration'],   # thread_concentration = 1 - (unique_threads / k)
    ]


def run_logreg_loo(all_results):
    """Leave-one-out logistic regression evaluation. Returns list of predictions."""
    X = np.array([extract_features(m) for m in all_results])
    labels = [m['expected'] for m in all_results]
    le = LabelEncoder()
    y = le.fit_transform(labels)

    loo = LeaveOneOut()
    predictions = [''] * len(all_results)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)
        predictions[test_idx[0]] = le.inverse_transform(pred)[0]

    return predictions


TEST_QUERIES = [
    # ── Thread 1 — OAuth token refresh failing ────────────────────────────
    ("OAuth token refresh failing",                   "NARROW", "exact → T1"),
    ("401 on token refresh in staging",               "NARROW", "specific detail → T1"),
    ("client secret stale in staging env",            "NARROW", "specific → T1"),
    ("refresh grant rejected by OAuth server",        "NARROW", "rephrased → T1"),
    ("oauuth token failing staging",                  "NARROW", "typo → T1"),
    ("staging auth broken after secret rotation",     "NARROW", "indirect → T1"),

    # ── Thread 2 — Google login redirect loop ─────────────────────────────
    ("Google login redirect loop",                    "NARROW", "exact → T2"),
    ("GOOGLE_CALLBACK_URL pointing to localhost",     "NARROW", "specific detail → T2"),
    ("infinite redirect on Google sign-in",          "NARROW", "rephrased → T2"),
    ("callback URL misconfigured in prod",            "NARROW", "indirect → T2"),
    ("google login smoke test failing",               "NARROW", "specific → T2"),

    # ── Thread 3 — Slow queries on orders table ───────────────────────────
    ("slow queries on orders table",                  "NARROW", "exact → T3"),
    ("orders endpoint taking 4 to 6 seconds",         "NARROW", "specific → T3"),
    ("missing index on created_at column",            "NARROW", "specific → T3"),
    ("CREATE INDEX CONCURRENTLY orders table",        "NARROW", "exact detail → T3"),
    ("P99 latency dropped after adding index",        "NARROW", "outcome → T3"),
    ("full table scan on 8 million rows",             "NARROW", "specific → T3"),

    # ── Thread 4 — DB migration rollback after failed deploy ──────────────
    ("database migration rollback in prod",           "NARROW", "exact → T4"),
    ("flask db downgrade after failed migration",     "NARROW", "specific → T4"),
    ("FK constraint failed on add_user_preferences",  "NARROW", "exact detail → T4"),
    ("prod inconsistent state after migration",       "NARROW", "rephrased → T4"),
    ("migration caused 4 minute downtime",            "NARROW", "specific outcome → T4"),

    # ── Thread 5 — GitHub Actions build timeout ───────────────────────────
    ("GitHub Actions build timing out",               "NARROW", "exact → T5"),
    ("test suite jumped from 4 to 22 minutes",        "NARROW", "specific → T5"),
    ("pytest parallel runners fix CI",                "NARROW", "specific → T5"),
    ("split tests across matrix partitions",          "NARROW", "specific → T5"),
    ("integration tests slowing down pipeline",       "NARROW", "rephrased → T5"),

    # ── Thread 6 — Docker image size ballooning ───────────────────────────
    ("Docker image grew from 280MB to 1GB",           "NARROW", "exact → T6"),
    ("node_modules copied into Docker final stage",   "NARROW", "specific → T6"),
    ("multi-stage build to reduce image size",        "NARROW", "specific → T6"),
    ("docekr image too big after charting library",   "NARROW", "typo → T6"),
    ("frontend image bloated in CI",                  "NARROW", "rephrased → T6"),

    # ── Thread 7 — React hydration mismatch errors ────────────────────────
    ("React hydration mismatch in production",        "NARROW", "exact → T7"),
    ("SSR hydration error on product detail page",    "NARROW", "specific → T7"),
    ("Intl.NumberFormat different server vs client",  "NARROW", "specific → T7"),
    ("currency formatting causing hydration bug",     "NARROW", "rephrased → T7"),

    # ── Thread 8 — Memory leak in background worker ───────────────────────
    ("worker pod OOMKilled every 6 hours",            "NARROW", "exact → T8"),
    ("memory leak in background worker",              "NARROW", "exact → T8"),
    ("memray profiling found closure holding DataFrame", "NARROW", "specific → T8"),
    ("retry queue caching full response payloads",    "NARROW", "specific → T8"),
    ("RSS memory growing linearly in worker",         "NARROW", "specific → T8"),

    # ── Thread 9 — Dependency vulnerability in lodash ────────────────────
    ("lodash CVE prototype pollution",                "NARROW", "exact → T9"),
    ("Dependabot flagged lodash vulnerability",       "NARROW", "exact → T9"),
    ("upgrading lodash to 4.17.21",                  "NARROW", "specific → T9"),
    ("npm audit fix lodash packages",                 "NARROW", "specific → T9"),
    ("lodash-es to replace vulnerable lodash",        "NARROW", "specific → T9"),

    # ── Thread 10 — Exposed API keys in git history ───────────────────────
    ("AWS API key committed to git repo",             "NARROW", "exact → T10"),
    ("SES sender key exposed in commit a3f99bx",      "NARROW", "exact detail → T10"),
    ("git filter-repo to purge secrets from history", "NARROW", "specific → T10"),
    ("rotating compromised AWS credentials",          "NARROW", "rephrased → T10"),
    ("detect-secrets pre-commit hook added",          "NARROW", "specific outcome → T10"),

    # ── Thread 11 — Redis cache eviction causing slowdowns ────────────────
    ("Redis cache hit rate dropped from 94 to 31",    "NARROW", "exact → T11"),
    ("allkeys-lru eviction too aggressive",           "NARROW", "specific → T11"),
    ("Redis memory at 99 percent used",               "NARROW", "specific → T11"),
    ("gzip compress large API responses in Redis",    "NARROW", "specific → T11"),
    ("cahce eviction slowing down API",               "NARROW", "typo → T11"),

    # ── Thread 12 — Kubernetes pod crash loop ────────────────────────────
    ("api-gateway pod in CrashLoopBackOff",           "NARROW", "exact → T12"),
    ("Kubernetes pod OOMKilled in prod",              "NARROW", "exact → T12"),
    ("increase memory limit from 256Mi to 512Mi",     "NARROW", "specific → T12"),
    ("pod memory limit too low after feature",        "NARROW", "rephrased → T12"),

    # ── Thread 13 — Sprint planning for Q2 ───────────────────────────────
    ("Q2 sprint planning estimates in Jira",          "NARROW", "exact → T13"),
    ("submit story point estimates by Friday",        "NARROW", "specific → T13"),
    ("sprint scope locked for Q2",                    "NARROW", "specific outcome → T13"),

    # ── Thread 14 — On-call handoff issues ───────────────────────────────
    ("on-call handoff notes were incomplete",         "NARROW", "exact → T14"),
    ("no runbook for S3 timeout incident",            "NARROW", "specific → T14"),
    ("document search index rebuild procedure",       "NARROW", "specific → T14"),
    ("runbook required before closing incident",      "NARROW", "specific outcome → T14"),

    # ── Thread 15 — Push notifications not delivered on iOS ──────────────
    ("iOS push notifications failing for 20 percent of users", "NARROW", "exact → T15"),
    ("APNs certificate expiring in 2 days",           "NARROW", "specific → T15"),
    ("push delivery rate back to 98 after cert renewal", "NARROW", "outcome → T15"),
    ("ios push not being deliivered",                 "NARROW", "typo → T15"),

    # ═══════════════════════════════════════════════════════════
    #  AMBIGUOUS — plausibly 2 threads
    # ═══════════════════════════════════════════════════════════
    ("login not working in production",               "AMBIGUOUS", "T1 + T2 both auth"),
    ("authentication broken",                         "AMBIGUOUS", "T1 + T2 both auth"),
    ("OAuth or login issue",                          "AMBIGUOUS", "T1 + T2"),
    ("database performance problem",                  "AMBIGUOUS", "T3 slow queries + T11 Redis"),
    ("database issue after deploy",                   "AMBIGUOUS", "T3 + T4 both DB"),
    ("security vulnerability found in dependency",    "AMBIGUOUS", "T9 lodash + T10 key exposure"),
    ("secret or credential issue",                    "AMBIGUOUS", "T1 stale secret + T10 exposed key"),
    ("API response latency issue",                    "AMBIGUOUS", "T3 orders slow + T11 cache miss"),
    ("CI build or Docker issue",                      "AMBIGUOUS", "T5 GH Actions + T6 Docker"),
    ("memory or OOM problem",                         "AMBIGUOUS", "T8 worker + T12 k8s pod"),
    ("pod or worker crashing",                        "AMBIGUOUS", "T8 + T12 both OOM"),
    ("credentials rotation needed",                   "AMBIGUOUS", "T1 client_secret + T10 AWS key"),

    # ═══════════════════════════════════════════════════════════
    #  BROAD — spans many threads
    # ═══════════════════════════════════════════════════════════
    ("what security issues did we have",              "BROAD", "T9 + T10 + T1"),
    ("recent production incidents",                   "BROAD", "T4 + T8 + T11 + T12"),
    ("any database problems lately",                  "BROAD", "T3 + T4 + T11"),
    ("performance issues in the system",              "BROAD", "T3 + T8 + T11 + T12"),
    ("deployment or build problems",                  "BROAD", "T4 + T5 + T6"),
    ("infrastructure issues",                         "BROAD", "T8 + T11 + T12"),
    ("what engineering work happened this week",      "BROAD", "all threads"),
    ("any incidents or outages",                      "BROAD", "T4 + T8 + T11 + T12"),
    ("team updates and process changes",              "BROAD", "T13 + T14"),
    ("auth or security problems",                     "BROAD", "T1 + T2 + T9 + T10"),

    # ═══════════════════════════════════════════════════════════
    #  REJECT — no thread is a good match
    # ═══════════════════════════════════════════════════════════
    ("quarterly revenue forecasting",                 "REJECT", "no finance threads"),
    ("machine learning model training",               "REJECT", "no ML threads"),
    ("Figma design system components",                "REJECT", "no design threads"),
    ("hiring plan for Q3",                            "REJECT", "no hiring threads"),
    ("Stripe payment gateway integration",            "REJECT", "no payments threads"),
    ("GraphQL subscription setup",                    "REJECT", "no GraphQL threads"),
    ("Terraform provider configuration",              "REJECT", "no Terraform threads"),
    ("Kafka consumer group lag",                      "REJECT", "no Kafka threads"),
    ("DNS TTL configuration",                         "REJECT", "no DNS threads"),
    ("video streaming latency",                       "REJECT", "no video threads"),
]


COL_W = 14


def run_benchmark():
    print("\n" + "=" * 100)
    print("  BENCHMARK — rule-based vs logistic regression (LOO)")
    print("=" * 100)

    all_results = []

    # ─── Pass 1: compute metrics for every query ───
    for query, expected, desc in TEST_QUERIES:
        intent = analyze_query_intent(query)
        candidates = retrieve_candidates(query, intent, with_filter=False)

        if not candidates:
            print(f"\n-  Query '{query}' -- 0 candidates, skipping.")
            continue

        m = compute_metrics(candidates)
        m['query'] = query
        m['expected'] = expected
        all_results.append(m)

    # ─── Pass 2: z-normalise and classify ───
    pop_stats = compute_population_stats(all_results)
    print(f"\n  Population stats (for z-score normalisation):")
    for k, v in pop_stats.items():
        print(f"    {k:<20} = {v:.4f}")

    # ── Write parameters.json for ranking.py ──
    import json as _json
    params = {
        "Z_REL_GAP_HIGH":    Z_REL_GAP_HIGH,
        "Z_ENTROPY_LOW":     Z_ENTROPY_LOW,
        "Z_REL_GAP_AMB_LO":  Z_REL_GAP_AMB_LO,
        "Z_REL_GAP_AMB_HI":  Z_REL_GAP_AMB_HI,
        "Z_ENTROPY_AMB_LO":  Z_ENTROPY_AMB_LO,
        "Z_ENTROPY_AMB_HI":  Z_ENTROPY_AMB_HI,
        "Z_SIGNAL_BROAD":    Z_SIGNAL_BROAD,
        **pop_stats,
    }
    with open("parameters.json", "w") as f:
        _json.dump(params, f, indent=2)
    print("  Saved parameters.json")

    for m in all_results:
        m['predicted'] = decide(m, pop_stats)
        m['correct'] = m['predicted'] == m['expected']

    # ─── Run logistic regression LOO ───
    print("\n  Training logistic regression (leave-one-out)...")
    lr_preds = run_logreg_loo(all_results)
    for i, m in enumerate(all_results):
        m['lr_predicted'] = lr_preds[i]
        m['lr_correct'] = lr_preds[i] == m['expected']

    # ─── Summary table ───
    print()
    cols = ['expected', 'predicted', 'lr_pred', 'abs_ratio', 'rel_gap', 'signal_norm', 'entropy']

    header = f"  {'Query':<32}"
    for c in cols:
        header += f" {c:>{COL_W}}"
    header += f" {'rule':>5} {'lr':>5}"
    print(header)
    print(f"  {'─'*32} " + " ".join('─' * COL_W for _ in cols) + " " + '─' * 5 + " " + '─' * 5)

    correct_rules = sum(1 for m in all_results if m['correct'])
    correct_lr = sum(1 for m in all_results if m['lr_correct'])
    total = len(all_results)

    for m in all_results:
        row = f"  {m['query'][:30]:<32}"
        vals = {
            'expected': m['expected'],
            'predicted': m['predicted'],
            'lr_pred': m['lr_predicted'],
            'abs_ratio': m['abs_ratio'],
            'rel_gap': m.get('rel_gap', '-'),
            'signal_norm': m['signal_norm'],
            'entropy': m['ent_score_T0.1'],
        }
        for c in cols:
            row += f" {str(vals[c]):>{COL_W}}"
        row += f" {'  ✓' if m['correct'] else '  ✗':>5}"
        row += f" {'  ✓' if m['lr_correct'] else '  ✗':>5}"
        print(row)

    # ─── Per-class stats ───
    print()
    print(f"  {'CLASS':<10} {'RULES':>12} {'LOGREG':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12}")
    for label in ['NARROW', 'AMBIGUOUS', 'BROAD', 'REJECT']:
        expected_set = [m for m in all_results if m['expected'] == label]
        if not expected_set:
            continue
        r_hits = sum(1 for m in expected_set if m['correct'])
        l_hits = sum(1 for m in expected_set if m['lr_correct'])
        n = len(expected_set)
        print(f"  {label:<10} {r_hits:>5}/{n:<5} ({r_hits/n*100:>3.0f}%) {l_hits:>5}/{n:<5} ({l_hits/n*100:>3.0f}%)")

    print(f"\n  OVERALL ACCURACY:")
    print(f"    Rules:    {correct_rules}/{total} = {correct_rules/total*100:.0f}%")
    print(f"    LogReg:   {correct_lr}/{total} = {correct_lr/total*100:.0f}%")

    # ─── Confusion matrices ───
    labels = ['NARROW', 'AMBIGUOUS', 'BROAD', 'REJECT']
    for name, pred_key in [("RULES", "predicted"), ("LOGREG (LOO)", "lr_predicted")]:
        print(f"\n  {'CONFUSION MATRIX — ' + name:^50}")
        print(f"  {'(rows = expected, cols = predicted)':^50}")
        header = f"  {'':>10}"
        for pl in labels:
            header += f" {pl:>8}"
        print(header)
        for el in labels:
            row = f"  {el:>10}"
            for pl in labels:
                count = sum(1 for m in all_results if m['expected'] == el and m[pred_key] == pl)
                row += f" {count:>8}"
            print(row)

    # ─── LogReg feature importance ───
    print(f"\n  LOGREG FEATURE IMPORTANCE (full-data fit):")
    X_all = np.array([extract_features(m) for m in all_results])
    y_all = np.array([m['expected'] for m in all_results])
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    clf.fit(X_scaled, y_enc)
    class_names = le.classes_
    print(f"  {'Feature':<22}", end="")
    for cn in class_names:
        print(f" {cn:>10}", end="")
    print()
    for i, feat in enumerate(FEATURE_KEYS):
        print(f"  {feat:<22}", end="")
        for j in range(len(class_names)):
            print(f" {clf.coef_[j][i]:>10.3f}", end="")
        print()

    print("\n" + "=" * 100 + "\n")


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
