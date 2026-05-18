from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from memory.decision_rules import decide_label

def mmr_sort(query_embedding, candidate_embeddings, top_k=5, lambda_param=0.5):
    if len(candidate_embeddings) == 0:
        return []

    selected_indices = []
    candidate_indices = list(range(len(candidate_embeddings)))

    query_vec = np.array(query_embedding).reshape(1, -1)
   
    cand_vecs = np.array(candidate_embeddings)

    while len(selected_indices) < top_k and candidate_indices:
        best_score = -np.inf
        best_idx = -1

        for idx in candidate_indices:
            relevance = cosine_similarity(
                cand_vecs[idx].reshape(1, -1), 
                query_vec
            )[0][0]
            
            if not selected_indices:
                max_sim_to_selected = 0
            else:
                selected_vecs = cand_vecs[selected_indices]
                similarities = cosine_similarity(
                    cand_vecs[idx].reshape(1, -1),
                    selected_vecs
                )
                max_sim_to_selected = np.max(similarities)

            score = (lambda_param * relevance) - ((1 - lambda_param) * max_sim_to_selected)

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return selected_indices

import json as _json
import os as _os

ALPHA = 0.25
ENTROPY_TEMP = 0.1
MAX_BROAD_THREADS = 3
MAX_AMBIGUOUS_THREADS = 2
MIN_THREAD_SIZE = 2

_PARAMS_PATH = str(_os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "config", "parameters.json"))

# Fallback defaults — used when parameters.json doesn't exist yet.
_DEFAULTS = {
    "Z_REL_GAP_HIGH":    0.50,
    "Z_ENTROPY_LOW":    -0.50,
    "Z_REL_GAP_AMB_LO": -0.80,
    "Z_REL_GAP_AMB_HI":  1.00,
    "Z_ENTROPY_AMB_LO": -1.00,
    "Z_ENTROPY_AMB_HI":  0.50,
    "Z_SIGNAL_BROAD":    1.50,
    "rel_gap_mean":      0.50,
    "rel_gap_std":       0.30,
    "entropy_mean":      0.50,
    "entropy_std":       0.20,
    "signal_norm_mean":  1.50,
    "signal_norm_std":   1.00,
}

def _load_params():
    if _os.path.exists(_PARAMS_PATH):
        try:
            with open(_PARAMS_PATH) as f:
                return {**_DEFAULTS, **_json.load(f)}
        except Exception as e:
            print(f"[ranking] Warning: could not load {_PARAMS_PATH}: {e}")
    return dict(_DEFAULTS)

_PARAMS = _load_params()

Z_REL_GAP_HIGH   = _PARAMS["Z_REL_GAP_HIGH"]
Z_ENTROPY_LOW    = _PARAMS["Z_ENTROPY_LOW"]
Z_REL_GAP_AMB_LO = _PARAMS["Z_REL_GAP_AMB_LO"]
Z_REL_GAP_AMB_HI = _PARAMS["Z_REL_GAP_AMB_HI"]
Z_ENTROPY_AMB_LO = _PARAMS["Z_ENTROPY_AMB_LO"]
Z_ENTROPY_AMB_HI = _PARAMS["Z_ENTROPY_AMB_HI"]
Z_SIGNAL_BROAD   = _PARAMS["Z_SIGNAL_BROAD"]

_POP_STATS = {
    "rel_gap_mean":     _PARAMS["rel_gap_mean"],
    "rel_gap_std":      _PARAMS["rel_gap_std"],
    "entropy_mean":     _PARAMS["entropy_mean"],
    "entropy_std":      _PARAMS["entropy_std"],
    "signal_norm_mean": _PARAMS["signal_norm_mean"],
    "signal_norm_std":  _PARAMS["signal_norm_std"],
}


def _softmax_entropy(values, temp):
    """Compute normalized softmax entropy over values."""
    import math as _math
    v = np.array(values)
    neg_over_T = -v / temp
    neg_over_T -= neg_over_T.max()
    exp_s = np.exp(neg_over_T)
    probs = exp_s / exp_s.sum()
    ent = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_ent = float(_math.log2(len(v))) if len(v) > 1 else 1.0
    return ent / max_ent

def select_anchor(candidates, mode):
    if not candidates:
        return None

    # --- Group candidates by thread ---
    threads = {}
    for candidate in candidates:
        thread_id = candidate['metadata']['thread_id']
        if thread_id not in threads:
            threads[thread_id] = {
                'candidates': [],
                'distances': []
            }
        threads[thread_id]['candidates'].append(candidate)
        threads[thread_id]['distances'].append(candidate['distance'])
    
    # --- Compute thread scores ---
    thread_aggregates = []
    for thread_id, thread_data in threads.items():
        avg_distance = np.mean(thread_data['distances'])
        min_distance = np.min(thread_data['distances'])
        message_count = len(thread_data['candidates'])
        thread_score = avg_distance - np.log(message_count + 1) * ALPHA
        thread_aggregates.append({
            'thread_id': thread_id,
            'avg_distance': avg_distance,
            'thread_score': thread_score,
            'min_distance': min_distance,
            'message_count': message_count,
            'best_candidate': min(thread_data['candidates'], key=lambda x: x['distance'])
        })
    
    sorted_threads = sorted(thread_aggregates, key=lambda x: x['thread_score'])

    # --- Filter single-message threads in NORMAL mode ---
    if mode == "NORMAL":
        multi_msg = [t for t in sorted_threads if t['message_count'] >= MIN_THREAD_SIZE]
        if multi_msg:
            sorted_threads = multi_msg

    best = sorted_threads[0]

    # --- Compute decision metrics ---
    all_distances = [c['distance'] for c in candidates]
    mean_distance = np.mean(all_distances)
    std_distance = float(np.std(all_distances))
    signal = mean_distance - best['min_distance']
    signal_norm = signal / std_distance if std_distance > 0 else 0.0
    abs_ratio = best['min_distance'] / mean_distance if mean_distance > 0 else 1.0

    # --- Compute entropy over thread scores ---
    thread_scores = [t['thread_score'] for t in sorted_threads]
    entropy = _softmax_entropy(thread_scores, ENTROPY_TEMP)

    if len(sorted_threads) >= 2:
        second = sorted_threads[1]
        gap_score = second['thread_score'] - best['thread_score']
        spread = sorted_threads[-1]['thread_score'] - best['thread_score']
        rel_gap = gap_score / spread if spread > 0 else 1.0
    else:
        rel_gap = 1.0  # only one thread → treat as narrow

    # --- Stats dict to pass through ---
    stats = {
        'signal_norm': round(signal_norm, 4),
        'abs_ratio': round(abs_ratio, 4),
        'rel_gap': round(rel_gap, 4),
        'entropy': round(entropy, 4),
        'n_threads': len(sorted_threads),
        'best_score': round(float(best['thread_score']), 4),
        'best_msgs': best['message_count'],
    }

    # --- Decision rule: 4-class with z-score normalised thresholds ---
    def _decide():
        label = decide_label(
            signal_norm=signal_norm,
            abs_ratio=abs_ratio,
            rel_gap=rel_gap,
            entropy=entropy,
            pop_stats=_POP_STATS,
            thresholds={
                "Z_REL_GAP_HIGH": Z_REL_GAP_HIGH,
                "Z_ENTROPY_LOW": Z_ENTROPY_LOW,
                "Z_REL_GAP_AMB_LO": Z_REL_GAP_AMB_LO,
                "Z_REL_GAP_AMB_HI": Z_REL_GAP_AMB_HI,
                "Z_ENTROPY_AMB_LO": Z_ENTROPY_AMB_LO,
                "Z_ENTROPY_AMB_HI": Z_ENTROPY_AMB_HI,
                "Z_SIGNAL_BROAD": Z_SIGNAL_BROAD,
            },
        )

        if label == "NARROW":
            return {
                'type': 'narrow',
                'threads': [best['best_candidate']],
                'thread_ids': [best['thread_id']],
                'stats': stats,
            }

        if label == "AMBIGUOUS":
            top_threads = sorted_threads[:MAX_AMBIGUOUS_THREADS]
            return {
                'type': 'ambiguous',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads],
                'stats': stats,
            }

        if label == "BROAD":
            top_threads = sorted_threads[:MAX_BROAD_THREADS]
            return {
                'type': 'broad',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads],
                'stats': stats,
            }

        return None

    return _decide()
