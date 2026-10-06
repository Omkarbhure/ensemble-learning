import os
import sys
import argparse
import joblib
import numpy as np

# Ensure src directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_training_data

BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "digit_ensemble.pkl")


def train_ensemble(sample_size=30000, test_size=0.2, random_state=42):
    print("=" * 60)
    print("  Handwritten Digit Recognition - Ensemble Training")
    print("=" * 60)

    print("\n[1/5] Loading MNIST dataset from OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data
    y = mnist.target.astype(int)

    total_samples = len(X)
    print(f"Dataset loaded: {total_samples} total samples available.")

    # Apply sample limit if specified for balanced speed and accuracy
    if sample_size and sample_size < total_samples:
        print(f"Using {sample_size:,} samples for high-accuracy training...")
        np.random.seed(random_state)
        indices = np.random.choice(total_samples, sample_size, replace=False)
        X = X[indices]
        y = y[indices]

    # Preprocessing
    print("\n[2/5] Normalizing pixel values (0-255 -> 0.0-1.0)...")
    X = preprocess_training_data(X)

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    # Define Base Classifiers
    print("\n[3/5] Initializing base models:")
    print("  - Logistic Regression (lbfgs, max_iter=300)")
    print("  - Random Forest (150 trees, n_jobs=-1)")
    print("  - Support Vector Machine (RBF kernel, probability=True, C=5.0)")

    lr = LogisticRegression(max_iter=300, solver="lbfgs", random_state=random_state)
    rf = RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
    svm = SVC(probability=True, kernel="rbf", C=5.0, gamma="scale", random_state=random_state)

    # Soft Voting Classifier with Calibrated Weights
    # SVM has superior non-linear boundary separation on MNIST -> higher weight
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
        voting="soft",
        weights=[1, 2, 4],
        n_jobs=-1,
    )

    # Training Ensemble
    print("\n[4/5] Training Weighted Soft Voting Ensemble...")
    ensemble.fit(X_train, y_train)
    print("Training complete!")

    # Evaluation
    print("\n[5/5] Evaluating Models on Test Set:")
    print("-" * 60)

    model_names = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "svm": "Support Vector Machine",
    }

    for key, name in model_names.items():
        estimator = ensemble.named_estimators_[key]
        pred = estimator.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"  * {name:<26}: {acc * 100:.2f}% accuracy")

    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"  >>> {'Weighted Ensemble (Final)':<24}: {ensemble_acc * 100:.2f}% accuracy")
    print("-" * 60)

    print("\nEnsemble Detailed Classification Report:")
    print(classification_report(y_test, ensemble_pred, digits=4))

    # Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(ensemble, MODEL_PATH)
    print(f"\nTrained ensemble model successfully saved to:")
    print(f"  -> {MODEL_PATH}")

    return ensemble_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST Digit Ensemble Classifier")
    parser.add_argument(
        "--samples",
        type=int,
        default=30000,
        help="Number of samples to use (default: 30000, 0 or 70000 for full dataset)",
    )
    args = parser.parse_args()

    sample_size = None if args.samples in (0, 70000) else args.samples
    train_ensemble(sample_size=sample_size)