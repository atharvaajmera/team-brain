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


def _normalize_relevance_scores(values):
    if not values:
        return []

    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        return [1.0]

    min_v = float(arr.min())
    max_v = float(arr.max())
    spread = max_v - min_v
    if spread <= 0:
        return [1.0] * len(arr)

    # Lower thread scores are better, so invert into a 0..1 relevance score.
    normalized = 1.0 - ((arr - min_v) / spread)
    return normalized.tolist()


def diversify_threads(thread_aggregates, top_k, lambda_param=0.55, candidate_pool=None):
    if not thread_aggregates or top_k <= 0:
        return []

    if len(thread_aggregates) <= top_k:
        return list(thread_aggregates)

    pool_size = candidate_pool or max(top_k * 2, top_k)
    pool = list(thread_aggregates[:pool_size])
    embeddings = []

    for thread in pool:
        embedding = thread.get('best_candidate', {}).get('embedding')
        if embedding is None:
            return pool[:top_k]
        embeddings.append(embedding)

    selected_indices = []
    remaining_indices = list(range(len(pool)))
    relevance_scores = _normalize_relevance_scores(
        [thread['thread_score'] for thread in pool]
    )
    cand_vecs = np.array(embeddings)

    while remaining_indices and len(selected_indices) < top_k:
        best_idx = remaining_indices[0]
        best_score = -np.inf

        for idx in remaining_indices:
            relevance = relevance_scores[idx]
            if not selected_indices:
                novelty_penalty = 0.0
            else:
                similarities = cosine_similarity(
                    cand_vecs[idx].reshape(1, -1),
                    cand_vecs[selected_indices],
                )[0]
                novelty_penalty = float(np.max(similarities))

            score = (lambda_param * relevance) - ((1 - lambda_param) * novelty_penalty)
            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [pool[idx] for idx in selected_indices]


def compute_semantic_coherence(embeddings, top_k=5):
    if not embeddings:
        return 1.0

    limited = [embedding for embedding in embeddings[:top_k] if embedding is not None]
    if len(limited) <= 1:
        return 1.0

    vectors = np.array(limited)
    sims = cosine_similarity(vectors)
    upper_indices = np.triu_indices(len(limited), k=1)
    pairwise = sims[upper_indices]
    if pairwise.size == 0:
        return 1.0

    return float(np.mean(pairwise))

import json as _json
import os as _os

ALPHA = 0.25
ENTROPY_TEMP = 0.1
MAX_BROAD_THREADS = 3
MAX_AMBIGUOUS_THREADS = 2
MAX_REJECT_THREADS = 2
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
    "rel_gap_mean":      0.50,
    "rel_gap_std":       0.30,
    "entropy_mean":      0.50,
    "entropy_std":       0.20,
    "coherence_mean":    0.50,
    "coherence_std":     0.20,
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

_POP_STATS = {
    "rel_gap_mean":     _PARAMS["rel_gap_mean"],
    "rel_gap_std":      _PARAMS["rel_gap_std"],
    "entropy_mean":     _PARAMS["entropy_mean"],
    "entropy_std":      _PARAMS["entropy_std"],
    "coherence_mean":   _PARAMS["coherence_mean"],
    "coherence_std":    _PARAMS["coherence_std"],
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
    abs_ratio = best['min_distance'] / mean_distance if mean_distance > 0 else 1.0
    query_debug = best['best_candidate'].get('query_debug', {})
    domain_confidence = query_debug.get('domain_confidence')
    support_ratio = query_debug.get('support_ratio')

    # --- Compute entropy over thread scores ---
    thread_scores = [t['thread_score'] for t in sorted_threads]
    entropy = _softmax_entropy(thread_scores, ENTROPY_TEMP)
    coherence = compute_semantic_coherence(
        [t['best_candidate'].get('embedding') for t in sorted_threads],
        top_k=5,
    )

    if len(sorted_threads) >= 2:
        second = sorted_threads[1]
        gap_score = second['thread_score'] - best['thread_score']
        spread = sorted_threads[-1]['thread_score'] - best['thread_score']
        rel_gap = gap_score / spread if spread > 0 else 1.0
    else:
        rel_gap = 1.0  # only one thread → treat as narrow

    # --- Stats dict to pass through ---
    stats = {
        'abs_ratio': round(abs_ratio, 4),
        'rel_gap': round(rel_gap, 4),
        'entropy': round(entropy, 4),
        'coherence': round(coherence, 4),
        'n_threads': len(sorted_threads),
        'best_score': round(float(best['thread_score']), 4),
        'best_msgs': best['message_count'],
        'domain_confidence': round(float(domain_confidence), 4) if isinstance(domain_confidence, (int, float)) else None,
        'support_ratio': round(float(support_ratio), 4) if isinstance(support_ratio, (int, float)) else None,
    }

    # --- Decision rule: 4-class with z-score normalised thresholds ---
    def _decide():
        label = decide_label(
            signal_norm=0.0,
            abs_ratio=abs_ratio,
            rel_gap=rel_gap,
            entropy=entropy,
            coherence=coherence,
            pop_stats=_POP_STATS,
            thresholds={
                "Z_REL_GAP_HIGH": Z_REL_GAP_HIGH,
                "Z_ENTROPY_LOW": Z_ENTROPY_LOW,
                "Z_REL_GAP_AMB_LO": Z_REL_GAP_AMB_LO,
                "Z_REL_GAP_AMB_HI": Z_REL_GAP_AMB_HI,
                "Z_ENTROPY_AMB_LO": Z_ENTROPY_AMB_LO,
                "Z_ENTROPY_AMB_HI": Z_ENTROPY_AMB_HI,
            },
            domain_confidence=domain_confidence,
            support_ratio=support_ratio,
        )

        if label == "NARROW":
            return {
                'type': 'narrow',
                'threads': [best['best_candidate']],
                'thread_ids': [best['thread_id']],
                'stats': stats,
            }

        if label == "AMBIGUOUS":
            top_threads = diversify_threads(
                sorted_threads,
                top_k=MAX_AMBIGUOUS_THREADS,
                lambda_param=0.60,
                candidate_pool=max(6, MAX_AMBIGUOUS_THREADS * 3),
            )
            return {
                'type': 'ambiguous',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads],
                'stats': stats,
            }

        if label == "BROAD":
            top_threads = diversify_threads(
                sorted_threads,
                top_k=MAX_BROAD_THREADS,
                lambda_param=0.45,
                candidate_pool=max(8, MAX_BROAD_THREADS * 3),
            )
            return {
                'type': 'broad',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads],
                'stats': stats,
            }

        top_threads = diversify_threads(
            sorted_threads,
            top_k=MAX_REJECT_THREADS,
            lambda_param=0.35,
            candidate_pool=max(6, MAX_REJECT_THREADS * 3),
        )
        return {
            'type': 'reject',
            'threads': [t['best_candidate'] for t in top_threads],
            'thread_ids': [t['thread_id'] for t in top_threads],
            'stats': stats,
        }

    return _decide()
