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

ALPHA = 0.25              # Weight for message count bonus
RELATIVE_GAP_THRESHOLD = 0.1  # Below this → flat distribution → broad query
MIN_SIGNAL_STRENGTH = 0.2 # Minimum (mean_dist - best_dist) to consider a real match
MAX_BROAD_THREADS = 3     # Max threads returned for broad queries
MIN_THREAD_SIZE = 2       # Minimum messages for NORMAL mode selection

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

    # --- Signal strength: does the best thread stand out from the noise? ---
    all_distances = [c['distance'] for c in candidates]
    mean_distance = np.mean(all_distances)
    signal_strength = mean_distance - best['min_distance']

    if len(sorted_threads) < 2:
        print(f"[DEBUG] select_anchor: Only 1 thread, score={best['thread_score']:.4f}, "
              f"signal_strength={signal_strength:.4f} (mean={mean_distance:.4f}, best_dist={best['min_distance']:.4f})")
        if mode == "NORMAL" and signal_strength < MIN_SIGNAL_STRENGTH:
            print(f"[DEBUG] select_anchor FAILED: signal_strength {signal_strength:.4f} < {MIN_SIGNAL_STRENGTH}, query is noise")
            return None
        return {
            'type': 'narrow',
            'threads': [best['best_candidate']],
            'thread_ids': [best['thread_id']]
        }

    # --- Relative gap analysis ---
    second = sorted_threads[1]
    gap = second['thread_score'] - best['thread_score']
    spread = sorted_threads[-1]['thread_score'] - best['thread_score']
    relative_gap = gap / spread if spread > 0 else 1.0

    print(f"[DEBUG] select_anchor: best={best['thread_score']:.4f} (msgs={best['message_count']}), "
          f"second={second['thread_score']:.4f}, gap={gap:.4f}, spread={spread:.4f}, "
          f"relative_gap={relative_gap:.4f}, signal_strength={signal_strength:.4f} | mode={mode}")

    if mode == "FALLBACK":
        if relative_gap < RELATIVE_GAP_THRESHOLD:
            top_threads = sorted_threads[:MAX_BROAD_THREADS]
            print(f"[DEBUG] select_anchor BROAD (FALLBACK): returning {len(top_threads)} threads")
            return {
                'type': 'broad',
                'threads': [t['best_candidate'] for t in top_threads],
                'thread_ids': [t['thread_id'] for t in top_threads]
            }
        print(f"[DEBUG] select_anchor SUCCESS (FALLBACK): returning best thread")
        return {
            'type': 'narrow',
            'threads': [best['best_candidate']],
            'thread_ids': [best['thread_id']]
        }

    # --- NORMAL mode decision ---
    if relative_gap < RELATIVE_GAP_THRESHOLD:
        # Flat distribution → broad query → return top N threads
        top_threads = sorted_threads[:MAX_BROAD_THREADS]
        print(f"[DEBUG] select_anchor BROAD: relative_gap {relative_gap:.4f} < {RELATIVE_GAP_THRESHOLD}, "
              f"returning {len(top_threads)} threads")
        return {
            'type': 'broad',
            'threads': [t['best_candidate'] for t in top_threads],
            'thread_ids': [t['thread_id'] for t in top_threads]
        }
    elif signal_strength < MIN_SIGNAL_STRENGTH:
        # Best thread doesn't stand out from the noise
        print(f"[DEBUG] select_anchor FAILED: signal_strength {signal_strength:.4f} < {MIN_SIGNAL_STRENGTH}, query is noise")
        return None
    else:
        # Clear winner → narrow query
        print(f"[DEBUG] select_anchor SUCCESS (NARROW): score={best['thread_score']:.4f}, relative_gap={relative_gap:.4f}")
        return {
            'type': 'narrow',
            'threads': [best['best_candidate']],
            'thread_ids': [best['thread_id']]
        }