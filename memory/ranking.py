from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

ALPHA = 0.25
# ── 4-class thresholds ──
REL_GAP_HIGH = 0.30
ENTROPY_LOW = 0.50
ENTROPY_MED_HI = 0.62
SIGNAL_LOW_THRESH = 2.2
ENTROPY_TEMP = 0.1
MAX_BROAD_THREADS = 3
MAX_AMBIGUOUS_THREADS = 2
MIN_THREAD_SIZE = 2


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
        print(f"[DEBUG] select_anchor FAILED: No candidates provided | mode={mode}")
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
            print(f"[DEBUG] Filtered to {len(sorted_threads)} multi-message threads")

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

    print(f"[DEBUG] select_anchor: best={best['thread_score']:.4f} (msgs={best['message_count']}), "
          f"signal_norm={signal_norm:.4f}, abs_ratio={abs_ratio:.4f}, "
          f"rel_gap={rel_gap:.4f}, entropy={entropy:.4f} | mode={mode}")

    # --- Decision rule: 4-class (NARROW → AMBIGUOUS → BROAD → REJECT) ---
    def _decide():
        # 1. rel_gap high AND entropy low → NARROW
        if rel_gap > REL_GAP_HIGH and entropy < ENTROPY_LOW:
            print(f"[DEBUG] select_anchor NARROW: rel_gap {rel_gap:.4f} > {REL_GAP_HIGH} "
                  f"AND entropy {entropy:.4f} < {ENTROPY_LOW}")
            return {
                'type': 'narrow',
                'threads': [best['best_candidate']],
                'thread_ids': [best['thread_id']]
            }

        # 2. rel_gap medium AND entropy medium → AMBIGUOUS
        if 0.05 < rel_gap < 0.50 and 0.20 < entropy < ENTROPY_MED_HI:
            top_threads = sorted_threads[:MAX_AMBIGUOUS_THREADS]
            print(f"[DEBUG] select_anchor AMBIGUOUS: rel_gap {rel_gap:.4f}, "
                  f"entropy {entropy:.4f} → returning {len(top_threads)} threads")
            return {
                'type': 'ambiguous',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads]
            }

        # 3. signal medium → BROAD
        if signal_norm >= SIGNAL_LOW_THRESH:
            top_threads = sorted_threads[:MAX_BROAD_THREADS]
            print(f"[DEBUG] select_anchor BROAD: signal_norm {signal_norm:.4f} ≥ {SIGNAL_LOW_THRESH} "
                  f"→ returning {len(top_threads)} threads")
            return {
                'type': 'broad',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads]
            }

        # 4. signal low → REJECT
        print(f"[DEBUG] select_anchor REJECT: signal_norm {signal_norm:.4f} < {SIGNAL_LOW_THRESH}")
        return None

    return _decide()