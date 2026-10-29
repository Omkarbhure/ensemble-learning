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
from disambiguation import disambiguate_homoglyphs, is_five_pointed_star
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
    """Return concise category for a predicted character: Alphabet, Digit, or Symbol."""
    if char_str.isdigit():
        return "Digit"
    elif "/" in char_str or char_str.isalpha():
        return "Alphabet"
    elif char_str in SPECIAL_SYMBOLS_SET:
        return "Symbol"
    return "Character"


def get_character_category_detail(char_str: str) -> str:
    """Return descriptive badge with icon."""
    if char_str.isdigit():
        return "🔢 Digit (0–9)"
    elif "/" in char_str or char_str.isalpha():
        return "🔤 Alphabet (A–Z / a–z)"
    elif char_str in SPECIAL_SYMBOLS_SET:
        return "➕ Math & Special Symbol"
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

    # 0. Check for 5-pointed star:
    # Users drawing the math symbol '*' (star) frequently draw a 5-pointed star (★ / pentagram).
    if is_five_pointed_star(image):
        star_class = 38  # Class 38 is '*'
        star_probs = np.full_like(probabilities, 0.14 / (len(probabilities) - 1))
        star_probs[star_class] = 0.86
        probabilities = star_probs
        raw_prediction = star_class
        refined_prediction = star_class
        for k in individual_preds:
            individual_preds[k]["digit"] = "*"
            individual_preds[k]["category"] = "Symbol"
            individual_preds[k]["confidence"] = 0.86
    else:
        # 1. Algorithm Consensus Determination
        # If the base classifiers agree (majority >= 2), or if there is a decisive probability margin,
        # we strictly respect the algorithms' prediction without heuristic tampering.
        classes_voted = list(base_classes.values()) if base_classes else []
        majority_count = max([classes_voted.count(c) for c in set(classes_voted)]) if classes_voted else 0

        sorted_indices = np.argsort(probabilities)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top1_idx
        prob_margin = float(probabilities[top1_idx] - probabilities[top2_idx])

        has_algorithm_consensus = (
            majority_count >= 2
            or prob_margin >= 0.12
            or probabilities[raw_prediction] >= 0.35
        )

        if has_algorithm_consensus:
            refined_prediction = raw_prediction
        else:
            # Only use morphological disambiguation as an edge-case tie-breaker for ambiguous pairs
            refined_prediction = disambiguate_homoglyphs(raw_prediction, probabilities, image, LABEL_MAP)

    confidence = float(probabilities[refined_prediction])
    predicted_label = LABEL_MAP.get(refined_prediction, str(refined_prediction))
    category = get_character_category(predicted_label)

    # Extract Top-5 predictions with predicted candidate on top
    sorted_indices = np.argsort(probabilities)[::-1]
    top_5_indices = list(sorted_indices[:5])
    if refined_prediction not in top_5_indices:
        top_5_indices = [refined_prediction] + top_5_indices[:4]
    elif top_5_indices[0] != refined_prediction:
        top_5_indices.remove(refined_prediction)
        top_5_indices = [refined_prediction] + top_5_indices[:4]

    top_candidates = [
        (LABEL_MAP.get(idx, str(idx)), float(probabilities[idx]))
        for idx in top_5_indices
    ]

    return predicted_label, confidence, probabilities, individual_preds, top_candidates, category


# Alias
predict_character = predict_digit