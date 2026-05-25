# Baseline Lock

Date: 2026-05-23

Commit hash: `193f6606fbc29ab90846d4cec681deaade272562`

Source run:
- `scripts/benchmark.py` completed with `150/150` calibration queries used
- `scripts/diagnostics.py` baseline and PRF runs from `diagnostics_output.txt`

Locked metrics:
- `Recall@5 = 93.75%`
- `MRR = 0.8278`
- `Rule-based classifier accuracy = 34/64 = 53.13%`
- `Rule-based classifier errors = 30/64`
- `LogReg (LOO) accuracy = 40/64 = 62.50%`
- `LogReg (LOO) errors = 24/64`

Frozen stack:
- `retrieval.py`
- `ranking.py`
- `query expansion / PRF merge behavior`
- embedding model / embedding flow

Notes:
- Baseline and PRF aggregate retrieval metrics were identical on the latest diagnostics run.
- Retrieval remains strong enough to freeze for now.
- The primary bottleneck is classifier behavior and potentially unstable labels.
