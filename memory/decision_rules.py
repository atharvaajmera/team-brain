def decide_label(
    signal_norm,
    abs_ratio,
    rel_gap,
    entropy,
    coherence,
    pop_stats,
    thresholds,
    domain_confidence=None,
    support_ratio=None,
):
    rg = rel_gap if isinstance(rel_gap, (int, float)) else 1.0
    dc = domain_confidence if isinstance(domain_confidence, (int, float)) else None
    sr = support_ratio if isinstance(support_ratio, (int, float)) else None

    rel_gap_std = pop_stats.get("rel_gap_std", 0.0)
    entropy_std = pop_stats.get("entropy_std", 0.0)
    coherence_std = pop_stats.get("coherence_std", 0.0)

    z_rg = (
        (rg - pop_stats.get("rel_gap_mean", 0.0)) / rel_gap_std
        if rel_gap_std > 0 else 0.0
    )
    z_ent = (
        (entropy - pop_stats.get("entropy_mean", 0.0)) / entropy_std
        if entropy_std > 0 else 0.0
    )
    z_coh = (
        (coherence - pop_stats.get("coherence_mean", 0.0)) / coherence_std
        if coherence_std > 0 else 0.0
    )

    if dc is not None and sr is not None:
        if dc <= 0.18 and sr <= 0.20 and abs_ratio >= 0.72:
            return "REJECT"

        if dc >= 0.42 and abs_ratio <= 0.68 and coherence >= 0.26 and entropy <= 0.42:
            return "NARROW"

    if abs_ratio >= 0.90 and entropy >= 0.78 and coherence <= 0.19:
        return "REJECT"

    if coherence <= 0.18 and entropy >= 0.70 and rg <= 0.22:
        return "REJECT"

    if z_coh <= -0.9 and z_ent >= 0.8 and abs_ratio >= 0.84:
        return "REJECT"

    if abs_ratio <= 0.62 and rg >= 0.34 and entropy <= 0.32:
        return "NARROW"

    if rg <= 0.15 and entropy >= 0.48 and entropy < 0.72 and coherence >= 0.20:
        return "AMBIGUOUS"

    if rg <= 0.24 and entropy >= 0.62 and coherence >= 0.18 and abs_ratio < 0.90:
        return "BROAD"

    if z_rg > thresholds["Z_REL_GAP_HIGH"] and z_ent < thresholds["Z_ENTROPY_LOW"]:
        return "NARROW"

    if (
        thresholds["Z_REL_GAP_AMB_LO"] < z_rg < thresholds["Z_REL_GAP_AMB_HI"]
        and thresholds["Z_ENTROPY_AMB_LO"] < z_ent < thresholds["Z_ENTROPY_AMB_HI"]
    ):
        return "AMBIGUOUS"

    if z_ent >= 0.7 and z_coh >= -0.3 and abs_ratio < 0.92:
        return "BROAD"

    if entropy >= 0.45 and coherence >= 0.21:
        return "AMBIGUOUS"

    return "REJECT"
