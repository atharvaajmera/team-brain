import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scripts.diagnostics import _evaluate_queries, _load_test_queries, _load_thread_corpus


CLASS_ORDER = ["NARROW", "AMBIGUOUS", "BROAD", "REJECT"]

FEATURE_SPECS = {
    "entropy": lambda m: m.get("ent_score_T0.1", 0.0),
    "coherence": lambda m: m.get("semantic_coherence_top5", 1.0),
    "signal": lambda m: m.get("signal", 0.0),
    "rel_gap": lambda m: m.get("rel_gap", 1.0) if isinstance(m.get("rel_gap"), (int, float)) else 1.0,
    "signal_norm": lambda m: (m.get("signal", 0.0) / m.get("std_distance", 1.0)) if m.get("std_distance", 0.0) > 0 else 0.0,
    "abs_ratio": lambda m: m.get("abs_ratio", 1.0),
}

MODEL_SPECS = {
    "A_full": ["entropy", "coherence", "signal", "rel_gap", "signal_norm", "abs_ratio"],
    "B_minimal": ["entropy", "coherence", "rel_gap"],
    "C_middle": ["entropy", "coherence", "rel_gap", "abs_ratio"],
    "D_signal": ["entropy", "coherence", "rel_gap", "signal"],
}


def _evaluate_model(all_results, feature_names):
    X = np.array(
        [
            [FEATURE_SPECS[name](result) for name in feature_names]
            for result in all_results
        ],
        dtype=float,
    )
    labels = [result["expected"] for result in all_results]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    loo = LeaveOneOut()
    predictions = np.empty(len(all_results), dtype=int)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X_train_s, y_train)
        predictions[test_idx[0]] = clf.predict(X_test_s)[0]

    predicted_labels = encoder.inverse_transform(predictions)
    accuracy = float(np.mean(predicted_labels == np.array(labels)))
    macro_f1 = float(f1_score(labels, predicted_labels, labels=CLASS_ORDER, average="macro"))
    matrix = confusion_matrix(labels, predicted_labels, labels=CLASS_ORDER)
    return accuracy, macro_f1, matrix


def _print_confusion_matrix(matrix):
    header = "true\\pred".ljust(12) + "".join(f"{label:<12}" for label in CLASS_ORDER)
    print(header)
    print("-" * len(header))
    for label, row in zip(CLASS_ORDER, matrix):
        print(f"{label:<12}" + "".join(f"{int(value):<12}" for value in row))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed classifier feature subsets with LOO accuracy, macro-F1, and confusion matrices."
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

    print(f"Feature subset evaluation set size: {len(all_results)} queries")
    print(f"Retrieval mode: {'PRF' if args.use_prf else 'BASELINE'}\n")

    for model_name, feature_names in MODEL_SPECS.items():
        accuracy, macro_f1, matrix = _evaluate_model(all_results, feature_names)
        print(f"{model_name}")
        print(f"  features   : {', '.join(feature_names)}")
        print(f"  accuracy   : {accuracy:.4f}")
        print(f"  macro_f1   : {macro_f1:.4f}")
        print("  confusion_matrix:")
        _print_confusion_matrix(matrix)
        print()


if __name__ == "__main__":
    main()
