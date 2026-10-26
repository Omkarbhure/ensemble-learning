import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import load_model, LABEL_MAP, CANONICAL_MAP, SPECIAL_SYMBOLS_SET, get_character_category
from dataset_builder import DATASET_PATH

BASE_DIR = os.path.dirname(CURRENT_DIR)
BENCHMARK_FILE = os.path.join(BASE_DIR, "models", "benchmark_metrics.json")


def generate_benchmark_metrics(sample_limit=2400, random_state=42):
    """
    Evaluates individual base classifiers and the soft voting ensemble on a held-out test set.
    Caches the results to models/benchmark_metrics.json for fast rendering in the Streamlit UI.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

    data = np.load(DATASET_PATH)
    X = data["X"]
    y = data["y"]

    # Stratified split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Subsample if requested for fast benchmark calculation
    if sample_limit and sample_limit < len(y_test):
        _, X_eval, _, y_eval = train_test_split(
            X_test, y_test, test_size=sample_limit, random_state=random_state, stratify=y_test
        )
    else:
        X_eval, y_eval = X_test, y_test

    model = load_model()

    estimators = model.named_estimators_
    lr = estimators.get("lr")
    rf = estimators.get("rf")
    mlp = estimators.get("mlp")

    results = {}
    time_records = {}

    # 1. Base Classifiers Evaluation
    for name, clf, label_name in [
        ("lr", lr, "Logistic Regression"),
        ("rf", rf, "Random Forest (120 Trees)"),
        ("mlp", mlp, "Deep Neural Net (MLP)"),
    ]:
        if clf is not None:
            t0 = time.time()
            y_pred = clf.predict(X_eval)
            dur = time.time() - t0

            acc = float(accuracy_score(y_eval, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_eval, y_pred, average="weighted", zero_division=0
            )

            results[name] = {
                "name": label_name,
                "accuracy": round(acc * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1_score": round(float(f1) * 100, 2),
                "latency_ms_per_sample": round((dur / len(y_eval)) * 1000, 3),
            }

    # 2. Ensemble Evaluation
    t0 = time.time()
    y_ens = model.predict(X_eval)
    dur = time.time() - t0
    acc_ens = float(accuracy_score(y_eval, y_ens))
    prec_ens, rec_ens, f1_ens, _ = precision_recall_fscore_support(
        y_eval, y_ens, average="weighted", zero_division=0
    )

    results["ensemble"] = {
        "name": "Soft Voting Ensemble (Weighted 1:3:5)",
        "accuracy": round(acc_ens * 100, 2),
        "precision": round(float(prec_ens) * 100, 2),
        "recall": round(float(rec_ens) * 100, 2),
        "f1_score": round(float(f1_ens) * 100, 2),
        "latency_ms_per_sample": round((dur / len(y_eval)) * 1000, 3),
    }

    # 3. Category Breakdown (Digits vs Letters vs Symbols)
    category_metrics = {}
    digit_mask = y_eval < 10
    letter_mask = (y_eval >= 10) & (y_eval < 36)
    symbol_mask = y_eval >= 36

    for cat_name, mask in [
        ("Digits (0–9)", digit_mask),
        ("Alphabets (A–Z / a–z)", letter_mask),
        ("Special Symbols", symbol_mask),
    ]:
        if np.any(mask):
            cat_y = y_eval[mask]
            cat_pred = y_ens[mask]
            cat_acc = float(accuracy_score(cat_y, cat_pred))
            category_metrics[cat_name] = {
                "sample_count": int(np.sum(mask)),
                "accuracy": round(cat_acc * 100, 2),
            }

    # 4. Top Confused Pairs in Ensemble
    cm = confusion_matrix(y_eval, y_ens)
    np.fill_diagonal(cm, 0)  # Ignore correct predictions
    top_confused = []
    indices = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0]

    for true_idx, pred_idx in indices[:8]:
        cnt = int(cm[true_idx, pred_idx])
        if cnt > 0:
            top_confused.append({
                "true_label": LABEL_MAP.get(int(true_idx), str(true_idx)),
                "predicted_label": LABEL_MAP.get(int(pred_idx), str(pred_idx)),
                "count": cnt,
            })

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_eval_samples": len(y_eval),
        "total_classes": len(np.unique(y_eval)),
        "models": results,
        "categories": category_metrics,
        "top_confused_pairs": top_confused,
    }

    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Benchmark saved successfully to {BENCHMARK_FILE}")
    return output_payload


def load_benchmark_metrics():
    """Load cached benchmark metrics, generating them if not yet created."""
    if not os.path.exists(BENCHMARK_FILE):
        return generate_benchmark_metrics()
    try:
        with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return generate_benchmark_metrics()


if __name__ == "__main__":
    metrics = generate_benchmark_metrics()
    print("Ensemble Accuracy:", metrics["models"]["ensemble"]["accuracy"], "%")
