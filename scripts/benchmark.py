import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
try:
    from memory.decision_rules import decide_label
except ImportError:
    def decide_label(signal_norm, abs_ratio, rel_gap, entropy, coherence, pop_stats, thresholds):
        if signal_norm > thresholds['Z_SIGNAL_BROAD']: return "BROAD"
        if entropy < thresholds['Z_ENTROPY_LOW'] and rel_gap > thresholds['Z_REL_GAP_HIGH']: return "NARROW"
        if (thresholds['Z_ENTROPY_AMB_LO'] <= entropy <= thresholds['Z_ENTROPY_AMB_HI'] and
            thresholds['Z_REL_GAP_AMB_LO'] <= rel_gap <= thresholds['Z_REL_GAP_AMB_HI']): return "AMBIGUOUS"
        return "REJECT"
from memory.ranking import compute_semantic_coherence
from memory.storage import collection
from memory.retrieval import execute_semantic_search
try:
    from memory.intent import analyze_query_intent
except ImportError:
    analyze_query_intent = None
from memory.shared import softmax_entropy, group_threads, ALPHA, MIN_THREAD_SIZE

# â”€â”€ Z-score thresholds (in std-dev units from population mean) â”€â”€
Z_REL_GAP_HIGH   =  0.50   # NARROW: z(rel_gap) above this
Z_ENTROPY_LOW    = -0.50   # NARROW: z(entropy) below this
Z_REL_GAP_AMB_LO = -0.80   # AMBIGUOUS: z(rel_gap) above this
Z_REL_GAP_AMB_HI =  1.00   # AMBIGUOUS: z(rel_gap) below this
Z_ENTROPY_AMB_LO = -1.00   # AMBIGUOUS: z(entropy) above this
Z_ENTROPY_AMB_HI =  0.50   # AMBIGUOUS: z(entropy) below this
Z_SIGNAL_BROAD   =  1.50   # BROAD: z(signal_norm) above this

DEFAULT_QUERIES_FILE = str(REPO_ROOT / "config" / "benchmark_queries.json")
DEFAULT_PARAMS_FILE = str(REPO_ROOT / "config" / "parameters.json")




def compute_metrics(candidates):
    agg = group_threads(candidates)

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
    coherence = compute_semantic_coherence(
        [t['best_candidate'].get('embedding') for t in agg_by_score],
        top_k=5,
    )

    m['ent_score_T0.1']   = round(softmax_entropy(thread_scores, 0.1), 4)
    m['ent_score_T0.05']  = round(softmax_entropy(thread_scores, 0.05), 4)
    m['ent_score_T0.2']   = round(softmax_entropy(thread_scores, 0.2), 4)
    m['ent_mindist_T0.1'] = round(softmax_entropy(min_dists, 0.1), 4)
    m['ent_avgdist_T0.1'] = round(softmax_entropy(avg_dists, 0.1), 4)

    raw = np.array(min_dists)
    d_range = raw.max() - raw.min()
    normed = ((raw - raw.min()) / d_range).tolist() if d_range > 0 else [0.0] * len(raw)
    m['ent_mindist_norm_T0.1'] = round(softmax_entropy(normed, 0.1), 4)
    m['semantic_coherence_top5'] = round(coherence, 4)

    m['top3_threads'] = [int(t['thread_id']) for t in agg_by_score[:3]]

    return m


def compute_population_stats(all_metrics):
    rel_gaps, entropies, signal_norms, coherences = [], [], [], []
    for m in all_metrics:
        rg = m.get('rel_gap', None)
        if isinstance(rg, (int, float)):
            rel_gaps.append(rg)
        entropies.append(m['ent_score_T0.1'])
        sig = m['signal'] / m['std_distance'] if m['std_distance'] > 0 else 0.0
        signal_norms.append(sig)
        coherences.append(m.get('semantic_coherence_top5', 1.0))

    return {
        'rel_gap_mean':     float(np.mean(rel_gaps)) if rel_gaps else 0.0,
        'rel_gap_std':      float(np.std(rel_gaps))  if rel_gaps else 1.0,
        'entropy_mean':     float(np.mean(entropies)),
        'entropy_std':      float(np.std(entropies)) if len(entropies) > 1 else 1.0,
        'signal_norm_mean': float(np.mean(signal_norms)),
        'signal_norm_std':  float(np.std(signal_norms)) if len(signal_norms) > 1 else 1.0,
        'coherence_mean':   float(np.mean(coherences)),
        'coherence_std':    float(np.std(coherences)) if len(coherences) > 1 else 1.0,
    }

def _safe_percentiles(values):
    if not values:
        return {
            'p10': 0.0,
            'p25': 0.0,
            'p50': 0.0,
            'p75': 0.0,
            'p90': 0.0,
        }
    arr = np.array(values)
    return {
        'p10': float(np.percentile(arr, 10)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
    }


def _extract_query_text(item):
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, (list, tuple)) and item:
        return str(item[0]).strip()

    if isinstance(item, dict):
        for key in ('query', 'text', 'q'):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
def _load_queries_from_file(path):
    if not os.path.exists(path):
        return []

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    queries = []
    for item in data:
        q = _extract_query_text(item)
        if q:
            queries.append(q)
    return queries

def calibrate_parameters(queries, with_filter=False):
    all_metrics = []
    total = len(queries)

    for i, query in enumerate(queries, start=1):
        candidates = execute_semantic_search(query, filters=None, limit=40)
        if not candidates:
            print(f"[{i}/{total}] skipped (no candidates): {query}")
            continue

        m = compute_metrics(candidates)
        all_metrics.append(m)
        print(f"[{i}/{total}] ok: {query}")

    if not all_metrics:
        raise RuntimeError('Calibration failed: no queries produced retrieval metrics.')

    pop_stats = compute_population_stats(all_metrics)

    rel_gaps = [m.get('rel_gap') for m in all_metrics if isinstance(m.get('rel_gap'), (int, float))]
    entropies = [m['ent_score_T0.1'] for m in all_metrics]
    signal_norms = [m['signal'] / m['std_distance'] if m['std_distance'] > 0 else 0.0 for m in all_metrics]
    coherences = [m.get('semantic_coherence_top5', 1.0) for m in all_metrics]

    params = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'calibration_queries_total': total,
        'calibration_queries_used': len(all_metrics),

        # Decision thresholds (kept explicit in file for runtime transparency)
        'Z_REL_GAP_HIGH': Z_REL_GAP_HIGH,
        'Z_ENTROPY_LOW': Z_ENTROPY_LOW,
        'Z_REL_GAP_AMB_LO': Z_REL_GAP_AMB_LO,
        'Z_REL_GAP_AMB_HI': Z_REL_GAP_AMB_HI,
        'Z_ENTROPY_AMB_LO': Z_ENTROPY_AMB_LO,
        'Z_ENTROPY_AMB_HI': Z_ENTROPY_AMB_HI,
        'Z_SIGNAL_BROAD': Z_SIGNAL_BROAD,

        # Population means/std for z-normalization
        **pop_stats,

        # Percentiles for diagnostics/tuning
        'rel_gap_percentiles': _safe_percentiles(rel_gaps),
        'entropy_percentiles': _safe_percentiles(entropies),
        'signal_norm_percentiles': _safe_percentiles(signal_norms),
        'coherence_percentiles': _safe_percentiles(coherences),
    }

    return params


def write_parameters(params, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
        f.write('\n')


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Offline benchmark calibration: query -> retrieve -> metrics -> parameters.json'
    )
    parser.add_argument('--queries-file', default=DEFAULT_QUERIES_FILE, help='JSON file containing query list')
    parser.add_argument('--query', action='append', default=[], help='Inline query (repeatable)')
    parser.add_argument('--output', default=DEFAULT_PARAMS_FILE, help='Output JSON parameter file')
    parser.add_argument('--with-filter', action='store_true', help='Use temporal filter during retrieval')
    return parser


def decide(m, pop_stats):
    """4-class decision with z-normalised thresholds: NARROW -> AMBIGUOUS -> BROAD -> REJECT."""
    signal_norm = m['signal'] / m['std_distance'] if m['std_distance'] > 0 else 0.0
    entropy = m['ent_score_T0.1']
    rg = m.get('rel_gap', 1.0)
    if not isinstance(rg, (int, float)):
        rg = 1.0

    return decide_label(
        signal_norm=signal_norm,
        abs_ratio=m['abs_ratio'],
        rel_gap=rg,
        entropy=entropy,
        coherence=m.get('semantic_coherence_top5', 1.0),
        pop_stats=pop_stats,
        thresholds={
            'Z_REL_GAP_HIGH': Z_REL_GAP_HIGH,
            'Z_ENTROPY_LOW': Z_ENTROPY_LOW,
            'Z_REL_GAP_AMB_LO': Z_REL_GAP_AMB_LO,
            'Z_REL_GAP_AMB_HI': Z_REL_GAP_AMB_HI,
            'Z_ENTROPY_AMB_LO': Z_ENTROPY_AMB_LO,
            'Z_ENTROPY_AMB_HI': Z_ENTROPY_AMB_HI,
            'Z_SIGNAL_BROAD': Z_SIGNAL_BROAD,
        },
    )

FEATURE_KEYS = [
    'rel_gap',
    'entropy',
    'signal_norm',
    'abs_ratio',
    'semantic_coherence_top5',
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
        m.get('semantic_coherence_top5', 1.0),  # low coherence -> scattered neighborhood
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


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    file_queries = _load_queries_from_file(args.queries_file)
    cli_queries = [q.strip() for q in args.query if q and q.strip()]
    queries = file_queries + cli_queries

    if not queries:
        print(
            "No queries provided. Add queries in benchmark_queries.json "
            "or pass --query multiple times."
        )
        sys.exit(1)

    params = calibrate_parameters(queries, with_filter=args.with_filter)
    write_parameters(params, args.output)

    print(f"\nWrote calibration parameters to {args.output}")
    print(
        "Used "
        f"{params['calibration_queries_used']}/{params['calibration_queries_total']} "
        "queries for population stats."
    )
