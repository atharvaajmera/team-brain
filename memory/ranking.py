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

ALPHA = 0.05  # Weight for message count bonus: log(count+1) * alpha

def select_anchor(candidates, mode,):
    if not candidates:
        print(f"[DEBUG] select_anchor FAILED: No candidates provided | mode={mode}")
        return None

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
    
    thread_aggregates = []
    for thread_id, thread_data in threads.items():
        avg_distance = np.mean(thread_data['distances'])
        min_distance = np.min(thread_data['distances'])
        message_count = len(thread_data['candidates'])
        # Penalize single-message threads: higher count → bigger bonus (lower score)
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
    best_thread = sorted_threads[0]
    
    confidence_gap = None
    if len(sorted_threads) > 1:
        second_best_thread = sorted_threads[1]
        confidence_gap = second_best_thread['thread_score'] - best_thread['thread_score']
        print(f"[DEBUG] select_anchor: Best thread score={best_thread['thread_score']:.4f} "
              f"(avg_dist={best_thread['avg_distance']:.4f}, msgs={best_thread['message_count']}), "
              f"Second-best score={second_best_thread['thread_score']:.4f}, "
              f"Confidence gap={confidence_gap:.4f}, "
              f"min_distance={best_thread['min_distance']:.4f} | mode={mode}")
    else:
        print(f"[DEBUG] select_anchor: Best thread score={best_thread['thread_score']:.4f} "
              f"(avg_dist={best_thread['avg_distance']:.4f}, msgs={best_thread['message_count']}), "
              f"min_distance={best_thread['min_distance']:.4f} (only thread) | mode={mode}")

    if mode == "NORMAL":
        # Dynamic threshold: 80th percentile of all candidate distances
        all_distances = [c['distance'] for c in candidates]
        threshold = np.percentile(all_distances, 80)
        
        if best_thread['thread_score'] >= threshold:
            print(f"[DEBUG] select_anchor FAILED: Best thread score {best_thread['thread_score']:.4f} >= p80 threshold {threshold:.4f}")
            return None
        
        # Check confidence gap - if too small relative to score spread, results may be ambiguous
        if confidence_gap is not None:
            all_scores = [t['thread_score'] for t in sorted_threads]
            score_spread = np.percentile(all_scores, 80) - np.percentile(all_scores, 20)
            min_gap = score_spread * 0.1  # gap must be at least 10% of the score spread
            if confidence_gap < min_gap:
                print(f"[DEBUG] select_anchor FAILED: Confidence gap {confidence_gap:.4f} < min_gap {min_gap:.4f} (10% of spread {score_spread:.4f}), too ambiguous")
                return None
        
        gap_str = f"{confidence_gap:.4f}" if confidence_gap is not None else "N/A"
        print(f"[DEBUG] select_anchor SUCCESS: Anchor found with score={best_thread['thread_score']:.4f}, confidence_gap={gap_str}")
        return best_thread['best_candidate']
    elif mode == "FALLBACK":
        print(f"[DEBUG] select_anchor SUCCESS (FALLBACK): Returning best thread")
        return best_thread['best_candidate']
    else:
        return None