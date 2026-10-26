import os
import sys
import numpy as np
from PIL import Image


def disambiguate_homoglyphs(raw_pred_idx, probabilities, image, label_map):
    """
    Refine ambiguous homoglyph twin pairs using explicit structural morphology:
      - 'P' vs '9' vs 'R': 'P' has descending stem on LEFT and completely empty bottom-right;
                           '9' has descending stem on RIGHT;
                           'R' has diagonal leg in bottom-right.
      - 'B' vs '8': 'B' has straight vertical left spine; '8' has double indented waist.
      - 'D' vs '0': 'D' has straight vertical left spine; '0' is curved oval on left.
      - '6' vs '0': '6' has open top right; '0' is a closed loop.
      - '!' vs '1': '!' has a detached bottom dot separated by a blank gap.
      - 'E' vs '5': 'E' has straight vertical left spine with horizontal bars.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 and np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)

    # Convert to grayscale
    img_gray = image.convert("L")
    arr = np.array(img_gray)
    if np.mean(arr) > 130:
        arr = 255 - arr

    # Tight bounding box
    bin_img = (arr > 30).astype(np.uint8)
    if not np.any(bin_img):
        return raw_pred_idx

    y_indices, x_indices = np.where(bin_img)
    ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
    xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
    h = max(1, ymax - ymin)
    w = max(1, xmax - xmin)

    crop = bin_img[ymin:ymax+1, xmin:xmax+1]
    ch, cw = crop.shape

    # Helper: analyze left profile linearity (for B vs 8, D vs 0, E vs 5)
    r_start, r_end = int(ch * 0.15), int(ch * 0.85)
    left_cols = []
    for r in range(r_start, r_end):
        cols = np.where(crop[r] > 0)[0]
        if len(cols) > 0:
            left_cols.append(cols[0])

    left_span = (max(left_cols) - min(left_cols)) if left_cols else 999
    left_is_straight_spine = (left_span <= max(3, cw * 0.20))

    # Helper: analyze lower 40% ink distribution (below the upper loop of P/9/R)
    bot_crop = crop[int(ch * 0.58):, :]
    if bot_crop.size > 0:
        bot_mid = cw // 2
        bot_left = int(np.sum(bot_crop[:, :bot_mid] > 0))
        bot_right = int(np.sum(bot_crop[:, bot_mid:] > 0))
        bot_total = bot_left + bot_right
        left_ratio = (bot_left / (bot_total + 1e-5))
        right_ratio = (bot_right / (bot_total + 1e-5))
    else:
        left_ratio, right_ratio, bot_total = 0.5, 0.5, 0

    top_candidates = list(np.argsort(probabilities)[-6:][::-1])

    # -------------------------------------------------------------
    # 1. 'P' (class 25) vs '9' (class 9) vs 'R' (class 27) vs 'F' (class 15)
    # -------------------------------------------------------------
    # A letter 'P' has its descending vertical stem on the LEFT (lower 40% has >65% on left).
    # Its bottom-right quadrant is completely empty (<18% of bottom ink).
    # A digit '9' has its descending stroke on the RIGHT or center (lower 40% has >65% on right).
    # A letter 'R' has a left stem AND a diagonal leg extending to the bottom right (>22% on right).
    if raw_pred_idx in [9, 25, 27, 15] or any(c in [9, 25, 27] for c in top_candidates[:3]):
        # Decisive check on lower 40% ink location:
        if bot_total > 15:
            # If the lower stem is overwhelmingly on the LEFT and the right is EMPTY:
            if left_ratio >= 0.65 and right_ratio < 0.20:
                return 25  # Unambiguously 'P / p'

            # If the lower stem is on the RIGHT and the left is EMPTY:
            elif right_ratio >= 0.65 and left_ratio < 0.25:
                return 9   # Unambiguously '9'

            # If the lower portion has both a left stem AND a prominent right leg:
            elif left_ratio >= 0.35 and right_ratio >= 0.22:
                if raw_pred_idx in [25, 27, 9]:
                    return 27  # 'R / r'

    # -------------------------------------------------------------
    # 2. 'B' (class 11) vs '8' (class 8)
    # -------------------------------------------------------------
    if raw_pred_idx in [8, 11] or (8 in top_candidates[:2] and 11 in top_candidates[:2]):
        if left_is_straight_spine:
            return 11  # 'B'
        elif raw_pred_idx == 11 and not left_is_straight_spine:
            return 8   # '8'

    # -------------------------------------------------------------
    # 3. 'D' (class 13) vs '0' (class 0)
    # -------------------------------------------------------------
    if raw_pred_idx in [0, 13] or (0 in top_candidates[:2] and 13 in top_candidates[:2]):
        if left_is_straight_spine:
            return 13  # 'D'
        elif raw_pred_idx == 13 and not left_is_straight_spine:
            return 0   # '0'

    # -------------------------------------------------------------
    # 4. 'E' (class 14) vs '5' (class 5)
    # -------------------------------------------------------------
    if raw_pred_idx in [5, 14] or (5 in top_candidates[:2] and 14 in top_candidates[:2]):
        if left_is_straight_spine:
            return 14  # 'E'

    # -------------------------------------------------------------
    # 5. '!' (class 42) vs '1' (class 1)
    # -------------------------------------------------------------
    if raw_pred_idx in [1, 42] or (1 in top_candidates[:2] and 42 in top_candidates[:2]):
        lower_crop = crop[int(ch * 0.55):, :]
        row_density = np.sum(lower_crop > 0, axis=1)
        has_gap = np.any(row_density == 0) and np.sum(row_density[-max(3, int(ch * 0.1)):]) > 0
        if has_gap:
            return 42  # '!'
        else:
            return 1   # '1'

    # -------------------------------------------------------------
    # 6. '6' (class 6) vs '0' (class 0)
    # -------------------------------------------------------------
    if raw_pred_idx in [0, 6] or (0 in top_candidates[:2] and 6 in top_candidates[:2]):
        top_right_density = np.sum(crop[:int(ch * 0.35), int(cw * 0.60):])
        if top_right_density < (ch * 0.35 * cw * 0.40) * 0.12:
            return 6  # Open top right -> '6'
        else:
            return 0  # Closed oval -> '0'

    return raw_pred_idx
