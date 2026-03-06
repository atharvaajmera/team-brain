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
    # Thread 1 — Google OAuth Bug (7)
    ("OAuth redirect problem",                "NARROW",  "exact → T1"),
    ("Authentication failure",                "NARROW",  "synonym → T1 (401)"),
    ("Login configuration issue",             "NARROW",  "indirect → T1"),
    ("login isnt working after oauth",        "NARROW",  "casual → T1"),
    ("authentcation fail",                    "NARROW",  "typo → T1"),
    ("401 unauthorized error on login",       "NARROW",  "specific detail → T1"),
    ("Google login HTTPS redirect fix",       "NARROW",  "exact detail → T1"),

    # Thread 2 — General API Issues (1)
    ("API endpoints timing out",              "NARROW",  "exact → T2"),

    # Thread 3 — API Documentation (2)
    ("API documentation for new endpoints",   "NARROW",  "exact → T3"),
    ("reviewing the new endpoint docs",       "NARROW",  "rephrased → T3"),

    # Thread 4 — Deployment v3.0 (2)
    ("version 3.0 production deploy",         "NARROW",  "exact → T4"),
    ("QA status after v3 release",            "NARROW",  "specific → T4"),

    # Thread 5 — Dashboard Performance (3)
    ("Dashboard loading slow",                "NARROW",  "exact → T5"),
    ("why dashboard slow today",              "NARROW",  "casual → T5"),
    ("dashbord slow",                         "NARROW",  "typo → T5"),

    # Thread 6 — Multi-Bug Release (1)
    ("file upload bug in latest release",     "NARROW",  "specific → T6"),

    # Thread 7 — CI/CD Pipeline (3)
    ("CI pipeline broken",                    "NARROW",  "exact → T7"),
    ("ESLint config migration",               "NARROW",  "specific detail → T7"),
    ("linter update broke the build",         "NARROW",  "rephrased → T7"),

    # Thread 8 — Pagination (3)
    ("Cursor pagination implementation",      "NARROW",  "exact → T8"),
    ("cursr pagination",                      "NARROW",  "typo → T8"),
    ("cursor-based pagination for large datasets", "NARROW", "expanded → T8"),

    # Thread 9 — Caching / Redis (3)
    ("Redis caching setup",                   "NARROW",  "exact → T9"),
    ("redis",                                 "NARROW",  "keyword → T9"),
    ("caching API responses to reduce DB queries", "NARROW", "rephrased → T9"),

    # Thread 10 — Offsite (1)
    ("team offsite plans next month",         "NARROW",  "exact → T10"),

    # Thread 11 — DB Migration (2)
    ("Database schema migration",             "NARROW",  "exact → T11"),
    ("ALTER TABLE backup for user profiles",  "NARROW",  "specific → T11"),

    # Thread 12 — Security (2)
    ("XSS vulnerability in search",           "NARROW",  "exact → T12"),
    ("sanitize inputs to prevent XSS",        "NARROW",  "rephrased → T12"),

    # ═══════════════════════════════════════════════════════════
    #  AMBIGUOUS (30) — 2-3 threads plausibly relevant
    # ═══════════════════════════════════════════════════════════

    # Moved from NARROW — borderline / cross-thread queries
    ("database connection pool problem",      "AMBIGUOUS", "T2 specific but DB → T5/T11"),
    ("intermittent timeouts on API endpoints","AMBIGUOUS", "T2 but phrasing broader"),
    ("monitoring logs after deployment",      "AMBIGUOUS", "T4 deploy + monitoring overlap"),
    ("missing index on users table",          "AMBIGUOUS", "T5 perf but users table → T11"),
    ("query performance on users table",      "AMBIGUOUS", "T5 but users table → T11"),
    ("duplicate search results after release","AMBIGUOUS", "T6 bug + release → T4"),
    ("multiple bugs introduced in release",   "AMBIGUOUS", "T6 bugs + release → T4"),
    ("pagination",                            "AMBIGUOUS", "keyword T8 but mentioned in T2"),
    ("pre-commit hook for code quality",      "AMBIGUOUS", "T7 CI but code quality → T3"),
    ("redis caching not working?",            "AMBIGUOUS", "T9 cache + issue framing → T2"),

    # Moved from BROAD — partial thread focus (2-3 threads)
    ("deployment",                            "AMBIGUOUS", "mainly T4 but also T6"),
    ("API bug after deployment",              "AMBIGUOUS", "T2+T4 focused"),
    ("recent bug fixes",                      "AMBIGUOUS", "T1+T6 focused"),
    ("documentation and API updates",         "AMBIGUOUS", "T2+T3 focused"),
    ("production monitoring and error fixes", "AMBIGUOUS", "T2+T4+T5 focused"),
    ("API improvements",                      "AMBIGUOUS", "T2+T3 moderate focus"),
    ("Bugs and performance problems",         "AMBIGUOUS", "T5+T6 moderate focus"),
    ("Infrastructure and DevOps work",        "AMBIGUOUS", "T4+T7 moderate focus"),
    ("pipeline and deploy updates",           "AMBIGUOUS", "T4+T7 moderate focus"),
    ("error fixing and patches",              "AMBIGUOUS", "T1+T6+T12 partial focus"),

    # New — designed 2-3 thread ambiguity
    ("API issues and documentation",          "AMBIGUOUS", "T2+T3 overlap"),
    ("OAuth and security concerns",           "AMBIGUOUS", "T1+T12 overlap"),
    ("deployment and release issues",         "AMBIGUOUS", "T4+T6 overlap"),
    ("database table changes",               "AMBIGUOUS", "T5+T11 overlap"),
    ("build and CI pipeline issues",          "AMBIGUOUS", "T7+T4 overlap"),
    ("API timeout and caching",               "AMBIGUOUS", "T2+T9 overlap"),
    ("search bugs and vulnerabilities",       "AMBIGUOUS", "T6+T12 overlap"),
    ("user data migration issues",            "AMBIGUOUS", "T11+T5 overlap"),
    ("release stability concerns",            "AMBIGUOUS", "T4+T6 overlap"),
    ("pagination and API performance",        "AMBIGUOUS", "T8+T2 overlap"),

    # ═══════════════════════════════════════════════════════════
    #  BROAD (30) — query spans many threads
    # ═══════════════════════════════════════════════════════════

    ("security issue",                        "BROAD",   "vague → T12 + others"),
    ("System upgrades",                       "BROAD",   "vague → T4+T6"),
    ("API issue after deployment",            "BROAD",   "mixed → T2+T4+T6"),
    ("Production issues this week",           "BROAD",   "vague → T2+T4+T6"),
    ("login",                                 "BROAD",   "keyword → T1+others"),
    ("deploy broke api again",               "BROAD",   "casual → T4+T2"),
    ("database work this sprint",             "BROAD",   "vague → T2+T5+T11"),
    ("what went wrong in production",         "BROAD",   "vague → T2+T4+T6"),
    ("backend infrastructure changes",        "BROAD",   "mixed → T4+T7+T9+T11"),
    ("security and stability fixes",          "BROAD",   "mixed → T1+T6+T12"),
    ("release status update",                 "BROAD",   "mixed → T4+T6"),
    ("what is everyone working on",           "BROAD",   "vague → all threads"),
    ("database performance and migration",    "BROAD",   "mixed → T5+T11"),
    ("code quality and tooling",              "BROAD",   "mixed → T3+T7"),
    ("recent engineering decisions",          "BROAD",   "vague → multiple"),
    ("API problems this quarter",             "BROAD",   "vague → T2+T3+T6"),
    ("configuration changes recently",        "BROAD",   "mixed → T1+T7"),
    ("bug",                                   "BROAD",   "keyword → T1+T6+T12"),
    ("fixes",                                 "BROAD",   "keyword → multiple"),
    ("API and database issues",               "BROAD",   "mixed → T2+T5+T11"),
    ("testing and QA work",                   "BROAD",   "mixed → T4+T7"),
    ("performance improvements this sprint",  "BROAD",   "mixed → T5+T9"),
    ("endpoint changes and updates",          "BROAD",   "mixed → T2+T3+T8"),
    ("weekly engineering summary",            "BROAD",   "vague → all threads"),
    ("server and database problems",          "BROAD",   "mixed → T2+T5+T11"),
    ("API response time improvements",        "BROAD",   "mixed → T2+T5+T9"),
    ("schema and table modifications",        "BROAD",   "mixed → T5+T11"),
    ("what needs attention this week",        "BROAD",   "vague → multiple"),
    ("development workflow changes",          "BROAD",   "mixed → T4+T7"),
    ("recurring technical issues",            "BROAD",   "vague → T2+T5+T6"),

    # ═══════════════════════════════════════════════════════════
    #  REJECT (30) — no thread is a good match
    # ═══════════════════════════════════════════════════════════

    ("New features discussion",               "REJECT",  "no features thread"),
    ("Mobile app design",                     "REJECT",  "no mobile threads"),
    ("Machine learning model training",       "REJECT",  "no ML threads"),
    ("Kubernetes pod autoscaling policy",     "REJECT",  "no k8s threads"),
    ("React component rendering issues",      "REJECT",  "no React threads"),
    ("hiring senior backend engineers",       "REJECT",  "no hiring threads"),
    ("customer satisfaction survey results",  "REJECT",  "no customer threads"),
    ("marketing campaign performance",        "REJECT",  "no marketing threads"),
    ("quarterly revenue forecast",            "REJECT",  "no finance threads"),
    ("iOS app crash on startup",              "REJECT",  "no iOS threads"),
    ("Android notification delivery failure", "REJECT",  "no Android threads"),
    ("data warehouse ETL pipeline",           "REJECT",  "no ETL threads"),
    ("blockchain consensus mechanism",        "REJECT",  "no blockchain threads"),
    ("Figma design system components",        "REJECT",  "no design threads"),
    ("AWS S3 bucket permissions",             "REJECT",  "no AWS threads"),
    ("Docker image size optimization",        "REJECT",  "no Docker threads"),
    ("GraphQL schema design patterns",        "REJECT",  "no GraphQL threads"),
    ("Stripe payment integration",            "REJECT",  "no payment threads"),
    ("email template redesign",               "REJECT",  "no email threads"),
    ("video transcoding latency",             "REJECT",  "no video threads"),
    ("product roadmap next quarter",          "REJECT",  "no roadmap threads"),
    ("SSL certificate renewal",               "REJECT",  "no SSL threads"),
    ("VPN tunnel setup guide",                "REJECT",  "no VPN threads"),
    ("Kafka consumer rebalancing issues",     "REJECT",  "no Kafka threads"),
    ("Terraform state file management",       "REJECT",  "no Terraform threads"),
    ("Python virtual environment setup",      "REJECT",  "no Python env threads"),
    ("DNS propagation delay",                 "REJECT",  "no DNS threads"),
    ("JIRA workflow customization",           "REJECT",  "no JIRA threads"),
    ("load balancer sticky sessions",         "REJECT",  "no LB threads"),
    ("Webpack bundle size optimization",      "REJECT",  "no Webpack threads"),
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
