import os
import sys
import numpy as np
from PIL import Image


def is_five_pointed_star(image) -> bool:
    """
    Detect if the user drew a 5-pointed star (★ / pentagram).
    Users drawing the math symbol '*' (star) frequently draw a 5-pointed star.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 and np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)

    # Composite transparency/alpha onto solid black background
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")

    img_gray = image.convert("L")
    arr = np.array(img_gray)
    if np.mean(arr) > 130:
        arr = 255 - arr

    bin_img = (arr > 35).astype(np.uint8)
    y_idx, x_idx = np.where(bin_img)
    if len(y_idx) == 0:
        return False
    ymin, ymax = int(np.min(y_idx)), int(np.max(y_idx))
    xmin, xmax = int(np.min(x_idx)), int(np.max(x_idx))

    crop = bin_img[ymin:ymax+1, xmin:xmax+1]
    ch, cw = crop.shape
    aspect = cw / max(1, ch)
    # A star is roughly square (aspect ratio between 0.60 and 1.55)
    if not (0.60 <= aspect <= 1.55):
        return False
    if ch < 16 or cw < 16:
        return False

    # 1. Topmost profile: single apex in central band (0.28 to 0.72)
    top_band = crop[:max(2, int(ch * 0.12)), :]
    if not np.any(top_band):
        return False
    top_y, top_x = np.where(top_band > 0)
    top_center = np.mean(top_x) / cw
    if not (0.28 <= top_center <= 0.72):
        return False

    # Extreme corners of top band must be empty (unlike 5, 7, E, F, T, Z, etc.)
    top_left_corner = np.sum(top_band[:, :int(cw * 0.22)])
    top_right_corner = np.sum(top_band[:, int(cw * 0.78):])
    if top_left_corner > 0 or top_right_corner > 0:
        if (top_left_corner + top_right_corner) > 0.15 * len(top_x):
            return False

    # 2. Bottom-most rows (bottom 4% / 3 rows):
    # Star has open notch between feet: middle 30% of bottom rows has ZERO ink
    bot_slice = crop[-max(2, int(ch * 0.04)):, :]
    bot_mid_empty = np.sum(bot_slice[:, int(cw * 0.35):int(cw * 0.65)])
    if bot_mid_empty > 2:
        return False

    # 3. Bottom band (bottom 25%): must have two separated legs
    bot_band = crop[-max(4, int(ch * 0.25)):, :]
    bot_left = np.sum(bot_band[:, :int(cw * 0.42)])
    bot_right = np.sum(bot_band[:, int(cw * 0.58):])
    bot_mid = np.sum(bot_band[:, int(cw * 0.40):int(cw * 0.60)])
    if bot_left == 0 or bot_right == 0:
        return False
    # Notch check: middle of bottom 25% has significantly less ink than the legs
    if bot_mid > 0.25 * (bot_left + bot_right):
        return False

    # 4. Middle arms: Star has horizontal arms extending to edges in middle 25%-65% band
    mid_band = crop[int(ch * 0.25):int(ch * 0.65), :]
    mid_left = np.sum(mid_band[:, :max(2, int(cw * 0.15))])
    mid_right = np.sum(mid_band[:, -max(2, int(cw * 0.15)):])
    if mid_left == 0 or mid_right == 0:
        return False

    # 5. Distinction from A: In A, width at mid-height (y=0.40*ch) is much smaller than bottom width.
    mid_widths = []
    for r in range(int(ch * 0.30), int(ch * 0.60)):
        cols = np.where(crop[r] > 0)[0]
        if len(cols) > 0:
            mid_widths.append(cols[-1] - cols[0])
    max_mid_w = max(mid_widths) if mid_widths else 0
    if (max_mid_w / cw) < 0.82:
        return False

    # Arms indentation: above the arms (y=0.15 to 0.28), width is narrower than mid arms
    neck_band = crop[int(ch * 0.15):int(ch * 0.28), :]
    neck_widths = []
    for r in range(neck_band.shape[0]):
        cols = np.where(neck_band[r] > 0)[0]
        if len(cols) > 0:
            neck_widths.append(cols[-1] - cols[0])
    min_neck_w = min(neck_widths) if neck_widths else max_mid_w
    if min_neck_w > max_mid_w * 0.85:
        return False

    # 6. Center crossing strokes
    center_box = crop[int(ch * 0.30):int(ch * 0.70), int(cw * 0.25):int(cw * 0.75)]
    if np.sum(center_box > 0) < 12:
        return False

    return True


def disambiguate_homoglyphs(raw_pred_idx, probabilities, image, label_map):
    """
    Refines ambiguous homoglyph twin pairs ONLY when the ML models face genuine
    ambiguity (i.e. both twin candidates are closely competing in the top 2 predictions).

    Recognizes dual-case morphology (both uppercase and lowercase forms):
      - 'D / d' (13) vs '0' (0):
          * Capital 'D' has a straight vertical left spine.
          * Lowercase 'd' has a tall vertical ascender on the right (top 40% ink is on the right).
          * '0' is a closed loop without an isolated top-right ascender or left straight spine.
          * If the model already predicts 'D / d', it is preserved.
      - 'B / b' (11) vs '8' (8):
          * 'B' and 'b' have a straight vertical left spine or left ascender.
          * '8' has double indented waist curves on both left and right edges.
      - 'P / p' (25) vs '9' (9) vs 'R / r' (27):
          * 'P' has a descending left stem and empty lower-right.
          * '9' has descending ink on the right.
          * 'R' has a left stem and diagonal bottom-right leg.
      - '!' (42) vs '1' (1):
          * '!' features a detached bottom dot separated by a blank vertical gap.
          * '1' features a continuous line/flag without a separated dot.
      - '6' (6) vs '0' (0):
          * '6' has an open gap at top right.
          * '0' is a closed convex loop.
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

    crop = bin_img[ymin:ymax+1, xmin:xmax+1]
    ch, cw = crop.shape
    if ch < 8 or cw < 8:
        return raw_pred_idx

    top_candidates = list(np.argsort(probabilities)[-6:][::-1])
    top_2 = top_candidates[:2]

    # Helper: analyze left profile linearity (for capital B, capital D, E, etc.)
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

    # -------------------------------------------------------------
    # 1. 'D / d' (class 13) vs '0' (class 0)
    # -------------------------------------------------------------
    # Only evaluate if 0 and 13 are the top competing pair in the ensemble
    if 0 in top_2 and 13 in top_2:
        # Check for lowercase 'd': tall vertical ascender on the right
        top_crop = crop[:int(ch * 0.42), :]
        top_mid = cw // 2
        top_left_ink = int(np.sum(top_crop[:, :top_mid] > 0))
        top_right_ink = int(np.sum(top_crop[:, top_mid:] > 0))
        top_total = top_left_ink + top_right_ink

        is_lower_d = (top_total > 5 and (top_right_ink / (top_total + 1e-5) > 0.60))

        if left_is_straight_spine or is_lower_d:
            return 13  # Confirmed 'D / d'
        elif raw_pred_idx == 0 and not is_lower_d and not left_is_straight_spine:
            return 0   # Confirmed '0'
        # Otherwise, trust the algorithm's raw prediction
        return raw_pred_idx

    # -------------------------------------------------------------
    # 2. 'P / p' (class 25) vs '9' (class 9) vs 'R / r' (class 27)
    # -------------------------------------------------------------
    if raw_pred_idx in [9, 25, 27] and any(c in [9, 25, 27] for c in top_2):
        if bot_total > 15:
            # Overwhelmingly on left with empty right -> 'P / p'
            if left_ratio >= 0.65 and right_ratio < 0.20:
                return 25
            # Overwhelmingly on right with empty left -> '9'
            elif right_ratio >= 0.65 and left_ratio < 0.25:
                return 9
            # Left stem AND diagonal right leg -> 'R / r'
            elif left_ratio >= 0.35 and right_ratio >= 0.22:
                return 27

    # -------------------------------------------------------------
    # 3. 'B / b' (class 11) vs '8' (class 8)
    # -------------------------------------------------------------
    if 8 in top_2 and 11 in top_2:
        if left_is_straight_spine:
            return 11  # Straight spine -> 'B / b'
        return raw_pred_idx

    # -------------------------------------------------------------
    # 4. 'E / e' (class 14) vs '5' (class 5)
    # -------------------------------------------------------------
    if 5 in top_2 and 14 in top_2:
        if left_is_straight_spine:
            return 14  # Straight left spine -> 'E / e'
        return raw_pred_idx

    # -------------------------------------------------------------
    # 5. '!' (class 42) vs '1' (class 1)
    # -------------------------------------------------------------
    if 1 in top_2 and 42 in top_2:
        lower_crop = crop[int(ch * 0.55):, :]
        row_density = np.sum(lower_crop > 0, axis=1)
        has_gap = np.any(row_density == 0) and np.sum(row_density[-max(3, int(ch * 0.1)):]) > 0
        if has_gap:
            return 42  # Separated dot -> '!'
        elif raw_pred_idx == 42 and not has_gap:
            return 1   # Continuous stroke -> '1'
        return raw_pred_idx

    # -------------------------------------------------------------
    # 6. '6' (class 6) vs '0' (class 0)
    # -------------------------------------------------------------
    if 0 in top_2 and 6 in top_2:
        top_right_density = np.sum(crop[:int(ch * 0.35), int(cw * 0.60):])
        if top_right_density < (ch * 0.35 * cw * 0.40) * 0.10:
            return 6  # Open top right -> '6'
        else:
            return 0  # Closed oval -> '0'

    return raw_pred_idx
