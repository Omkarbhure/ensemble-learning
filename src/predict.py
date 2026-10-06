import os
import sys
import joblib
import numpy as np

# Ensure src directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_image

BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "digit_ensemble.pkl")

_cached_model = None


def load_model():
    """
    Load the trained voting ensemble model with caching.
    """
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_PATH}'. "
                f"Please run 'python src/train.py' first to train the ensemble model."
            )
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model


def predict_digit(image):
    """
    Predict handwritten digit from a PIL image or numpy array.
    Returns:
        digit (int): predicted digit (0-9)
        confidence (float): prediction confidence (0.0 to 1.0)
        probabilities (list/ndarray): probabilities for classes 0-9
        individual_predictions (dict): predictions from individual models in ensemble
    """
    model = load_model()
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    probabilities = model.predict_proba(processed_image)[0]
    confidence = float(probabilities.max())

    # Extract individual estimator predictions if available
    individual_preds = {}
    if hasattr(model, "named_estimators_"):
        for name, estimator in model.named_estimators_.items():
            try:
                pred = estimator.predict(processed_image)[0]
                proba = estimator.predict_proba(processed_image)[0].max()
                individual_preds[name] = {"digit": int(pred), "confidence": float(proba)}
            except Exception:
                pass

    return int(prediction[0]), confidence, probabilities, individual_preds