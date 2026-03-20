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

# â”€â”€ Z-score thresholds (in std-dev units from population mean) â”€â”€
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


if __name__ == "__main__":
    # TODO: auto-generate queries from ChromaDB using IR techniques,
    # run retrieval + classification, compute pop_stats, write parameters.json
    pass
