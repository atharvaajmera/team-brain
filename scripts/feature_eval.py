import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scripts.diagnostics import _evaluate_queries, _load_test_queries, _load_thread_corpus


FEATURE_SPECS = [
    ("entropy", lambda m: m.get("ent_score_T0.1", 0.0)),
    ("coherence", lambda m: m.get("semantic_coherence_top5", 1.0)),
    ("signal", lambda m: m.get("signal", 0.0)),
    ("rel_gap", lambda m: m.get("rel_gap", 1.0) if isinstance(m.get("rel_gap"), (int, float)) else 1.0),
    ("signal_norm", lambda m: (m.get("signal", 0.0) / m.get("std_distance", 1.0)) if m.get("std_distance", 0.0) > 0 else 0.0),
    ("abs_ratio", lambda m: m.get("abs_ratio", 1.0)),
]


def _format_float(value):
    return f"{value:.4f}"


def _feature_vector(result, feature_names):
    feature_map = {name: fn(result) for name, fn in FEATURE_SPECS}
    return [feature_map[name] for name in feature_names]


def _evaluate_logreg_loo(all_results, feature_names):
    if not all_results:
        return 0.0

    X = np.array([_feature_vector(result, feature_names) for result in all_results], dtype=float)
    labels = [result["expected"] for result in all_results]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    loo = LeaveOneOut()
    correct = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)
        correct += int(pred[0] == y_test[0])

    return correct / len(all_results)


def _print_class_stats(all_results):
    classes = ["NARROW", "AMBIGUOUS", "BROAD", "REJECT"]
    print("Per-class feature summary (mean +- std)\n")

    header = (
        f"{'feature':<22} {'narrow':<24} {'ambiguous':<24} "
        f"{'broad':<24} {'reject':<24}"
    )
    print(header)
    print("-" * len(header))

    for feature_name, feature_fn in FEATURE_SPECS:
        row = [f"{feature_name:<22}"]
        for label in classes:
            subset = [feature_fn(result) for result in all_results if result["expected"] == label]
            if subset:
                mean = np.mean(subset)
                std = np.std(subset)
                row.append(f"{_format_float(mean)} +- {_format_float(std):<12}")
            else:
                row.append(f"{'-':<24}")
        print(" ".join(row))


def _print_ablation(all_results):
    all_feature_names = [name for name, _ in FEATURE_SPECS]
    baseline_acc = _evaluate_logreg_loo(all_results, all_feature_names)

    print("\nLeave-one-out feature ablation (logistic regression)\n")
    print(f"All features accuracy: {_format_float(baseline_acc)}")
    print(f"{'removed_feature':<24} {'features_left':<14} {'accuracy':<10} {'delta':<10}")
    print("-" * 62)

    for feature_name in all_feature_names:
        kept = [name for name in all_feature_names if name != feature_name]
        acc = _evaluate_logreg_loo(all_results, kept)
        delta = acc - baseline_acc
        print(
            f"{feature_name:<24} {len(kept):<14} "
            f"{_format_float(acc):<10} {_format_float(delta):<10}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diagnostics feature usefulness and simple leave-one-out ablations."
    )
    parser.add_argument(
        "--use-prf",
        action="store_true",
        help="Evaluate on PRF-enabled retrieval instead of baseline retrieval.",
    )
    args = parser.parse_args()

    test_queries = _load_test_queries()
    if not test_queries:
        raise RuntimeError("No diagnostics queries found.")

    thread_corpus = _load_thread_corpus()
    all_results = _evaluate_queries(test_queries, thread_corpus, use_prf=args.use_prf)
    if not all_results:
        raise RuntimeError("No evaluation results produced.")

    print(f"Feature evaluation set size: {len(all_results)} queries")
    print(f"Retrieval mode: {'PRF' if args.use_prf else 'BASELINE'}\n")

    _print_class_stats(all_results)
    _print_ablation(all_results)


if __name__ == "__main__":
    main()
