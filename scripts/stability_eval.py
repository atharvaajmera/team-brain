import argparse
import gc
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory import decision as decision_module
from memory import retrieval as retrieval_module
from memory import storage as storage_module
from scripts import diagnostics as diagnostics_module
from scripts import slack_convos_generator


def _threads_to_records(threads):
    texts = []
    ids = []
    metadatas = []

    for thread in threads:
        for message in thread.get("messages", []):
            text = (message.get("text") or "").strip()
            if not text:
                continue

            author = message.get("user", "unknown")
            ts = float(message.get("ts", 0))
            thread_id = float(message.get("thread_ts", ts))
            texts.append(text)
            ids.append(f"{author}_{ts}")
            metadatas.append({
                "author": author,
                "ts": ts,
                "text": text,
                "thread_id": thread_id,
            })

    return texts, ids, metadatas


def _swap_collection(temp_collection):
    originals = {
        "storage": storage_module.collection,
        "retrieval": retrieval_module.collection,
        "decision": decision_module.collection,
        "diagnostics": diagnostics_module.collection,
    }

    storage_module.collection = temp_collection
    retrieval_module.collection = temp_collection
    decision_module.collection = temp_collection
    diagnostics_module.collection = temp_collection
    return originals


def _restore_collection(originals):
    storage_module.collection = originals["storage"]
    retrieval_module.collection = originals["retrieval"]
    decision_module.collection = originals["decision"]
    diagnostics_module.collection = originals["diagnostics"]


def _cleanup_temp_dir(path, attempts=5, delay=0.2):
    last_error = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            time.sleep(delay)
    if last_error is not None:
        print(f"Warning: could not remove temp stability dir '{path}': {last_error}")
    return False


def _run_single_corpus(seed, reseed, base_ts, use_prf):
    run_seed = seed * 1000 + reseed
    random.seed(run_seed)
    threads = slack_convos_generator.generate_threads(base_ts=base_ts)
    texts, ids, metadatas = _threads_to_records(threads)

    temp_dir = tempfile.mkdtemp(prefix="team_brain_stability_")
    client = None
    temp_collection = None
    originals = None
    try:
        client = chromadb.PersistentClient(
            path=temp_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        temp_collection = client.get_or_create_collection(name="slack_archive")
        temp_collection.upsert(documents=texts, ids=ids, metadatas=metadatas)

        originals = _swap_collection(temp_collection)
        try:
            test_queries = diagnostics_module._load_test_queries()
            thread_corpus = diagnostics_module._load_thread_corpus()
            all_results = diagnostics_module._evaluate_queries(
                test_queries,
                thread_corpus,
                use_prf=use_prf,
            )
            if not all_results:
                raise RuntimeError("No diagnostics results produced for stability run.")

            summary = diagnostics_module._summarize_results(all_results)
            accuracy = float(np.mean([m["correct"] for m in all_results]))
            return {
                "seed": seed,
                "reseed": reseed,
                "run_seed": run_seed,
                "accuracy": accuracy,
                "recall@5": summary["recall@5"],
                "mrr": summary["mrr"],
                "rule_errors": summary["rule_errors"],
            }
        finally:
            _restore_collection(originals)
    finally:
        temp_collection = None
        client = None
        originals = None
        gc.collect()
        if os.path.exists(temp_dir):
            _cleanup_temp_dir(temp_dir)


def _mean_std(values):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def main():
    parser = argparse.ArgumentParser(
        description="Measure classifier and retrieval stability across synthetic corpus reseeds."
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of primary seeds to evaluate.")
    parser.add_argument("--reseeds", type=int, default=5, help="Number of reseeds per primary seed.")
    parser.add_argument("--base-ts", type=int, default=1780000000, help="Base timestamp for synthetic corpora.")
    parser.add_argument("--use-prf", action="store_true", help="Evaluate with PRF enabled.")
    args = parser.parse_args()

    rows = []
    for seed in range(1, args.seeds + 1):
        for reseed in range(1, args.reseeds + 1):
            base_ts = args.base_ts + (seed * 100000) + (reseed * 1000)
            result = _run_single_corpus(seed, reseed, base_ts, use_prf=args.use_prf)
            rows.append(result)
            print(
                f"seed={seed} reseed={reseed} "
                f"accuracy={result['accuracy']:.4f} "
                f"recall@5={result['recall@5']:.4f} "
                f"mrr={result['mrr']:.4f} "
                f"rule_errors={result['rule_errors']}"
            )

    accuracy_mean, accuracy_std = _mean_std([row["accuracy"] for row in rows])
    recall_mean, recall_std = _mean_std([row["recall@5"] for row in rows])
    mrr_mean, mrr_std = _mean_std([row["mrr"] for row in rows])

    print("\nSTABILITY SUMMARY")
    print(f"runs              : {len(rows)}")
    print(f"retrieval mode    : {'PRF' if args.use_prf else 'BASELINE'}")
    print(f"accuracy mean/std : {accuracy_mean:.4f} / {accuracy_std:.4f}")
    print(f"Recall@5 mean/std : {recall_mean:.4f} / {recall_std:.4f}")
    print(f"MRR mean/std      : {mrr_mean:.4f} / {mrr_std:.4f}")


if __name__ == "__main__":
    main()
