import os
import sys
import importlib
import joblib
import numpy as np

# Ensure src directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import disambiguation
importlib.reload(disambiguation)
from disambiguation import disambiguate_homoglyphs
from preprocessing import preprocess_image

BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "digit_ensemble.pkl")
CHARACTER_MODEL_PATH = os.path.join(BASE_DIR, "models", "character_ensemble.pkl")

# 48 Classes mapping (Digits, Alphabets A/a through Z/z, 12 Special Symbols)
LABEL_MAP = {
    # 0 - 9: Digits
    **{i: str(i) for i in range(10)},
    # 10 - 35: Alphabets (Matching both capital and small forms)
    **{10 + i: f"{chr(65 + i)} / {chr(97 + i)}" for i in range(26)},
    # 36 - 47: Special Symbols
    36: "+",
    37: "-",
    38: "*",
    39: "/",
    40: "=",
    41: "?",
    42: "!",
    43: "#",
    44: "@",
    45: "$",
    46: "%",
    47: "&",
}

# Single canonical character representation
CANONICAL_MAP = {
    **{i: str(i) for i in range(10)},
    **{10 + i: chr(65 + i) for i in range(26)},
    36: "+",
    37: "-",
    38: "*",
    39: "/",
    40: "=",
    41: "?",
    42: "!",
    43: "#",
    44: "@",
    45: "$",
    46: "%",
    47: "&",
}

SPECIAL_SYMBOLS_SET = {"+", "-", "*", "/", "=", "?", "!", "#", "@", "$", "%", "&"}


def get_character_category(char_str: str) -> str:
    """Return descriptive category for a predicted character."""
    if char_str.isdigit():
        return "Digit (0–9)"
    elif "/" in char_str or char_str.isalpha():
        return "Alphabet (Capital & Small)"
    elif char_str in SPECIAL_SYMBOLS_SET:
        return "Math & Special Symbol"
    return "Character"


_cached_model = None


def load_model():
    """
    Load the trained voting ensemble model with caching.
    """
    global _cached_model
    if _cached_model is None:
        target_path = CHARACTER_MODEL_PATH if os.path.exists(CHARACTER_MODEL_PATH) else MODEL_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Trained model not found at '{target_path}'. "
                f"Please run 'python src/train.py' first to train the ensemble model."
            )
        _cached_model = joblib.load(target_path)
    return _cached_model


def predict_digit(image, weights=None, voting_mode="soft"):
    """
    Unified prediction interface for digits, alphabets (both capital and small), and 12 special characters.
    Supports dynamic weights and soft vs. hard voting mode for ensemble exploration.

    Parameters:
        image: PIL Image or ndarray
        weights (dict, optional): e.g. {'lr': 1, 'rf': 3, 'mlp': 5}. Defaults to 1:3:5.
        voting_mode (str): "soft" (probability weighted sum) or "hard" (majority voting).

    Returns:
        prediction_label (str): e.g. '7', 'A / a', '#', '@'
        confidence (float): prediction confidence (0.0 to 1.0)
        probabilities (ndarray): full class probabilities
        individual_predictions (dict): predictions from each base estimator
        top_candidates (list): list of top 5 (label, probability) tuples
        category (str): character category description
    """
    model = load_model()
    processed_image = preprocess_image(image)

    # Base estimator probabilities
    individual_preds = {}
    base_probs = {}
    base_classes = {}

    if hasattr(model, "named_estimators_"):
        for name, estimator in model.named_estimators_.items():
            try:
                proba = estimator.predict_proba(processed_image)[0]
                base_probs[name] = proba
                pred_raw = int(np.argmax(proba))
                base_classes[name] = pred_raw
                est_label = LABEL_MAP.get(pred_raw, str(pred_raw))
                individual_preds[name] = {
                    "digit": est_label,
                    "confidence": float(proba[pred_raw]),
                    "category": get_character_category(est_label),
                }
            except Exception:
                pass

    # Dynamic Weight Resolution
    if weights is None:
        weights = {"lr": 1, "rf": 3, "mlp": 5}
    w_lr = max(0, float(weights.get("lr", 1)))
    w_rf = max(0, float(weights.get("rf", 3)))
    w_mlp = max(0, float(weights.get("mlp", 5)))
    w_total = w_lr + w_rf + w_mlp
    if w_total <= 0:
        w_lr, w_rf, w_mlp, w_total = 1.0, 1.0, 1.0, 3.0

    # Probability & Consensus Computation
    if base_probs and "lr" in base_probs and "rf" in base_probs and "mlp" in base_probs:
        p_lr = base_probs["lr"]
        p_rf = base_probs["rf"]
        p_mlp = base_probs["mlp"]
        soft_probs = (w_lr * p_lr + w_rf * p_rf + w_mlp * p_mlp) / w_total

        if voting_mode.lower() == "hard":
            # Hard majority voting
            vote_tally = {}
            for name, w in [("lr", w_lr), ("rf", w_rf), ("mlp", w_mlp)]:
                cls = base_classes[name]
                vote_tally[cls] = vote_tally.get(cls, 0.0) + w
            raw_prediction = max(vote_tally.items(), key=lambda item: item[1])[0]
            probabilities = soft_probs
        else:
            probabilities = soft_probs
            raw_prediction = int(np.argmax(probabilities))
    else:
        # Fallback to model's default prediction
        raw_prediction = int(model.predict(processed_image)[0])
        probabilities = model.predict_proba(processed_image)[0]

    # Refine ambiguous homoglyphs (e.g. B vs 8, D vs 0, 6 vs 0, ! vs 1, P vs 9)
    refined_prediction = disambiguate_homoglyphs(raw_prediction, probabilities, image, LABEL_MAP)

    if refined_prediction != raw_prediction:
        # Boost refined candidate so probabilities and confidence match morphological truth
        raw_prob = float(probabilities[raw_prediction])
        probabilities = probabilities.copy()
        probabilities[refined_prediction] = max(raw_prob, 0.72)
        probabilities[raw_prediction] = min(float(probabilities[raw_prediction]), 0.12)
        probabilities = probabilities / np.sum(probabilities)

    confidence = float(probabilities[refined_prediction])

    predicted_label = LABEL_MAP.get(refined_prediction, str(refined_prediction))
    category = get_character_category(predicted_label)

    # Extract Top-5 predictions with refined candidate on top
    top_5_indices = [idx for idx in np.argsort(probabilities)[::-1] if idx != refined_prediction][:4]
    top_5_indices = [refined_prediction] + top_5_indices
    top_candidates = [
        (LABEL_MAP.get(idx, str(idx)), float(probabilities[idx]))
        for idx in top_5_indices
    ]

    return predicted_label, confidence, probabilities, individual_preds, top_candidates, category


# Alias
predict_character = predict_digit