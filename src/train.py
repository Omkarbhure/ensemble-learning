import os
import sys
import argparse
import joblib
import numpy as np

# Ensure src directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from dataset_builder import build_character_dataset, LABEL_MAP, DATASET_PATH

BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "digit_ensemble.pkl")
CHARACTER_MODEL_PATH = os.path.join(MODELS_DIR, "character_ensemble.pkl")


def train_ensemble(test_size=0.2, random_state=42):
    print("=" * 65)
    print("  High-Accuracy Character & Symbol Ensemble Training (48 Classes)")
    print("  Supported: Digits (0-9), Alphabets (A-Z & a-z), Symbols (+, -, *, /, =, ?, !, #, @, $, %, &)")
    print("=" * 65)

    # 1. Load or Build Dataset
    print("\n[1/5] Loading calibrated 48-class character & symbol dataset...")
    if not os.path.exists(DATASET_PATH):
        print("  -> Precompiled dataset not found. Building now...")
        X, y = build_character_dataset(samples_per_class=700, random_state=random_state)
    else:
        data = np.load(DATASET_PATH)
        X = data["X"]
        y = data["y"]
        print(f"  -> Loaded existing dataset: {len(y):,} samples, {len(np.unique(y))} classes.")

    # 2. Strict Supervised Featurization: Train / Test Split
    print("\n[2/5] Creating stratified train/test split (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  * Training samples : {len(X_train):,}")
    print(f"  * Testing samples  : {len(X_test):,}")

    # 3. Base Classifiers
    print("\n[3/5] Initializing base models for high-accuracy 43-class ensemble:")
    print("  - Logistic Regression (L-BFGS, max_iter=200) - Weight: 1")
    print("  - Random Forest (120 trees, multi-threaded)  - Weight: 3")
    print("  - Deep Neural Network MLP (384x192, Adam)    - Weight: 5")

    lr = LogisticRegression(max_iter=200, solver="lbfgs", random_state=random_state)
    rf = RandomForestClassifier(n_estimators=120, max_depth=30, random_state=random_state, n_jobs=-1)
    mlp = MLPClassifier(
        hidden_layer_sizes=(384, 192),
        activation="relu",
        solver="adam",
        max_iter=75,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=random_state,
    )

    # Soft Voting Classifier with Calibrated Weights
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("mlp", mlp)],
        voting="soft",
        weights=[1, 3, 5],
        n_jobs=1,
    )

    # 4. Training Ensemble
    print("\n[4/5] Training Weighted Soft Voting Ensemble...")
    ensemble.fit(X_train, y_train)
    print("Training complete!")

    # 5. Evaluation
    print("\n[5/5] Evaluating Models on Test Set:")
    print("-" * 65)

    model_names = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "mlp": "Neural Network (MLP)",
    }

    for key, name in model_names.items():
        estimator = ensemble.named_estimators_[key]
        pred = estimator.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  * {name:<26}: {acc * 100:.2f}% accuracy")

    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"  >>> {'Weighted Ensemble (Final)':<24}: {ensemble_acc * 100:.2f}% accuracy")
    print("-" * 65)

    target_names = [LABEL_MAP.get(i, str(i)) for i in sorted(np.unique(y_test))]
    print("\nEnsemble Detailed Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=target_names, digits=3))

    # Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(ensemble, MODEL_PATH)
    joblib.dump(ensemble, CHARACTER_MODEL_PATH)
    print(f"\nTrained ensemble model successfully saved to:")
    print(f"  -> {MODEL_PATH}")
    print(f"  -> {CHARACTER_MODEL_PATH}")

    return ensemble_acc


if __name__ == "__main__":
    train_ensemble()