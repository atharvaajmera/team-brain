def decide_label(signal_norm, abs_ratio, rel_gap, entropy, coherence, pop_stats, thresholds):
    rg = rel_gap if isinstance(rel_gap, (int, float)) else 1.0

    rel_gap_std = pop_stats.get("rel_gap_std", 0.0)
    entropy_std = pop_stats.get("entropy_std", 0.0)
    signal_norm_std = pop_stats.get("signal_norm_std", 0.0)
    coherence_std = pop_stats.get("coherence_std", 0.0)

    z_rg = (
        (rg - pop_stats.get("rel_gap_mean", 0.0)) / rel_gap_std
        if rel_gap_std > 0 else 0.0
    )
    z_ent = (
        (entropy - pop_stats.get("entropy_mean", 0.0)) / entropy_std
        if entropy_std > 0 else 0.0
    )
    z_sig = (
        (signal_norm - pop_stats.get("signal_norm_mean", 0.0)) / signal_norm_std
        if signal_norm_std > 0 else 0.0
    )
    z_coh = (
        (coherence - pop_stats.get("coherence_mean", 0.0)) / coherence_std
        if coherence_std > 0 else 0.0
    )

    if abs_ratio >= 0.88 and entropy >= 0.75 and rg <= 0.05:
        return "REJECT"

    if coherence <= 0.35 and entropy >= 0.65 and rg <= 0.20:
        return "REJECT"

    if z_coh <= -1.0 and entropy >= 0.60 and signal_norm <= 1.5:
        return "REJECT"

    if signal_norm >= 2.4 and rg >= 0.30 and entropy <= 0.55:
        return "NARROW"

    if rg <= 0.12 and entropy >= 0.65 and signal_norm < 1.4:
        return "AMBIGUOUS"

    if rg <= 0.15 and entropy >= 0.60 and signal_norm >= 1.4 and abs_ratio <= 0.85:
        return "BROAD"

    if z_rg > thresholds["Z_REL_GAP_HIGH"] and z_ent < thresholds["Z_ENTROPY_LOW"]:
        return "NARROW"

    if (
        thresholds["Z_REL_GAP_AMB_LO"] < z_rg < thresholds["Z_REL_GAP_AMB_HI"]
        and thresholds["Z_ENTROPY_AMB_LO"] < z_ent < thresholds["Z_ENTROPY_AMB_HI"]
    ):
        return "AMBIGUOUS"

    if z_sig >= thresholds["Z_SIGNAL_BROAD"]:
        return "BROAD"

    if signal_norm >= 1.0 and entropy >= 0.45:
        return "AMBIGUOUS"

    return "REJECT"
