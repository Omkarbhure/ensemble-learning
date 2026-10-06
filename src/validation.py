import numpy as np
from PIL import Image, ImageOps
import scipy.ndimage


def analyze_image_properties(image):
    """
    Extract geometric, topological, and stroke properties of the input image.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 and np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)

    # Composite alpha onto black background
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(bg, image.convert("RGBA")).convert("RGB")

    image = image.convert("L")
    img_array = np.array(image)

    # Auto-invert light background to dark background
    corner_pixels = [
        img_array[0, 0],
        img_array[0, -1],
        img_array[-1, 0],
        img_array[-1, -1],
    ]
    if np.mean(corner_pixels) > 128 or np.mean(img_array) > 130:
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # Threshold binary ink
    binary = (img_array > 35).astype(np.uint8)
    ink_pixel_count = int(np.sum(binary))

    bbox = image.getbbox()
    if bbox is None or ink_pixel_count < 25:
        return {
            "is_empty": True,
            "ink_pixel_count": ink_pixel_count,
            "total_pixels": img_array.size,
            "bbox": None,
        }

    total_w = bbox[2] - bbox[0]
    total_h = bbox[3] - bbox[1]
    bbox_area = max(1, total_w * total_h)
    fill_ratio = ink_pixel_count / bbox_area

    # Connected Component Analysis
    structure = np.ones((5, 5), dtype=int)
    dilated = scipy.ndimage.binary_dilation(binary, structure=structure)
    labeled, num_components = scipy.ndimage.label(dilated)

    # Holes / loops detection (Euler number analysis via inverted background)
    # Digits have at most 2 closed holes (e.g. '8' has 2, '0','6','9','4' have 1, others have 0)
    cropped_binary = binary[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Pad cropped binary with zeros to ensure background is connected
    padded = np.pad(cropped_binary, pad_width=2, mode='constant', constant_values=0)
    inv_padded = (padded == 0).astype(np.uint8)
    _, num_bg_regions = scipy.ndimage.label(inv_padded)
    # Background minus the outer border region gives internal holes
    holes_count = max(0, num_bg_regions - 1)

    # Aspect ratio
    aspect_ratio = total_w / max(1, total_h)

    # Edge density / complexity (to detect complex photos/drawings)
    dx = np.diff(img_array.astype(float), axis=1)
    dy = np.diff(img_array.astype(float), axis=0)
    edge_energy = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    return {
        "is_empty": False,
        "ink_pixel_count": ink_pixel_count,
        "total_pixels": img_array.size,
        "bbox": bbox,
        "width": total_w,
        "height": total_h,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "num_components": num_components,
        "holes_count": holes_count,
        "edge_energy": edge_energy,
        "binary": binary,
        "cropped_binary": cropped_binary,
    }


def detect_alphabet_patterns(crop):
    """
    Detect explicit structural patterns of common alphabet letters (e.g., B, D, E, F, H, T, A, K, X).
    Returns (is_alphabet: bool, letter_name: str, explanation: str)
    """
    h, w = crop.shape
    if h < 20 or w < 10:
        return False, "", ""

    # Extract vertical left and right profiles
    left_profile, right_profile = [], []
    for y in range(int(h * 0.08), int(h * 0.92)):
        row = np.where(crop[y, :])[0]
        if len(row) > 0:
            left_profile.append(row[0])
            right_profile.append(row[-1])

    if len(left_profile) < 10:
        return False, "", ""

    left_profile = np.array(left_profile)
    right_profile = np.array(right_profile)
    n = len(left_profile)

    # Vertical linearity of the left side
    y_vals = np.arange(n)
    slope_l, intercept_l = np.polyfit(y_vals, left_profile, 1)
    res_left = np.mean(np.abs(left_profile - (slope_l * y_vals + intercept_l)))
    left_span = np.max(left_profile) - np.min(left_profile)

    mid_start, mid_end = int(n * 0.35), int(n * 0.65)
    left_waist_indent = np.mean(left_profile[mid_start:mid_end]) - np.mean(
        np.concatenate([left_profile[:int(n * 0.25)], left_profile[int(n * 0.75):]])
    )
    right_waist_indent = np.mean(
        np.concatenate([right_profile[:int(n * 0.25)], right_profile[int(n * 0.75):]])
    ) - np.mean(right_profile[mid_start:mid_end])

    # 1. Letter 'B' Check
    # (Straight vertical left spine + double right lobes with middle indentation)
    top_span = np.max(right_profile[:int(n * 0.45)] - left_profile[:int(n * 0.45)])
    bot_span = np.max(right_profile[int(n * 0.55):] - left_profile[int(n * 0.55):])
    if top_span > w * 0.55 and bot_span > w * 0.55:
        if (res_left < w * 0.09 or left_span < w * 0.18) and right_waist_indent > w * 0.06:
            if left_waist_indent < max(2.0, right_waist_indent * 0.35):
                return (
                    True,
                    "Alphabet Letter 'B'",
                    "Detected a straight vertical left spine with double right lobes characteristic of the letter 'B'. Only single digits (0–9) are allowed.",
                )

    # 2. Letter 'D' Check
    # (Straight vertical left spine + single rounded right bulge with no top/bottom gaps)
    if (res_left < w * 0.08 or left_span < w * 0.14) and (np.max(right_profile) - np.min(right_profile) > w * 0.15):
        right_bulge = np.mean(right_profile[mid_start:mid_end]) - np.mean(
            np.concatenate([right_profile[:int(n * 0.20)], right_profile[int(n * 0.80):]])
        )
        if right_bulge > w * 0.08:
            return (
                True,
                "Alphabet Letter 'D'",
                "Detected a straight vertical left spine with a rounded right boundary characteristic of the letter 'D'.",
            )

    # 3. Letter 'T' Check
    # (Full-width horizontal top crossbar with a centered vertical stem)
    top_w = np.sum(crop[:int(h * 0.25), :] > 0, axis=1).max()
    bot_w = np.sum(crop[int(h * 0.50):, :] > 0, axis=1).mean()
    if top_w > w * 0.85 and bot_w < w * 0.35:
        stem_cols = np.where(np.sum(crop[int(h * 0.40):, :], axis=0) > 0)[0]
        if len(stem_cols) > 0 and (stem_cols.min() > w * 0.20 and stem_cols.max() < w * 0.80):
            return (
                True,
                "Alphabet Letter 'T'",
                "Detected a top horizontal crossbar and centered stem characteristic of the letter 'T'.",
            )

    # 4. Letter 'H' Check
    # (Two vertical parallel uprights connected by a central horizontal crossbar, with open top and bottom gaps)
    left_col_density = np.sum(crop[:, :int(w * 0.25)]) / (h * int(w * 0.25))
    right_col_density = np.sum(crop[:, -int(w * 0.25):]) / (h * int(w * 0.25))
    top_mid_density = np.sum(crop[:int(h * 0.35), int(w * 0.35):int(w * 0.65)]) / (int(h * 0.35) * int(w * 0.30))
    bot_mid_density = np.sum(crop[int(h * 0.65):, int(w * 0.35):int(w * 0.65)]) / (int(h * 0.35) * int(w * 0.30))
    if left_col_density > 0.25 and right_col_density > 0.25 and top_mid_density < 0.08 and bot_mid_density < 0.08:
        return (
            True,
            "Alphabet Letter 'H'",
            "Detected dual parallel uprights with open top/bottom gaps characteristic of the letter 'H'.",
        )

    # 5. Letter 'A' Check
    # (Triangular apex + horizontal crossbar + open bottom between two legs)
    apex_span = np.max(right_profile[:int(n * 0.18)] - left_profile[:int(n * 0.18)])
    base_span = np.max(right_profile[int(n * 0.80):] - left_profile[int(n * 0.80):])
    if apex_span < w * 0.35 and base_span > w * 0.75:
        # Check crossbar in middle and gap at bottom
        mid_row_sum = np.sum(crop[int(h * 0.45):int(h * 0.60), :] > 0, axis=1).max()
        bot_center_density = np.sum(crop[int(h * 0.75):, int(w * 0.35):int(w * 0.65)]) / (int(h * 0.25) * int(w * 0.30))
        if mid_row_sum > w * 0.45 and bot_center_density < 0.15:
            return (
                True,
                "Alphabet Letter 'A'",
                "Detected a triangular apex, horizontal crossbar, and bifurcated base characteristic of the letter 'A'.",
            )

    return False, "", ""


def validate_digit_input(image, ensemble_pred=None):
    """
    Validate that the input image is strictly a single valid digit (0-9).
    Rejects:
      - Empty canvas / blanks
      - Multi-digit inputs (e.g. '10', '99')
      - Alphabets / Text letters (e.g. 'A', 'B', 'D', 'H', 'T', 'X')
      - Non-digit drawings / complex scribbles / shapes / geometric figures
      - Photos / cluttered images

    Returns:
      (is_valid: bool, error_type: str, title: str, description: str)
    """
    props = analyze_image_properties(image)

    # 1. Check for empty canvas
    if props.get("is_empty", False):
        return (
            False,
            "EMPTY_INPUT",
            "Empty Input Detected",
            "The canvas or uploaded image is blank. Please draw a single digit (0–9) or upload an image containing a digit.",
        )

    # 2. Check for solid blocks / excessive ink / photo clutter
    if props["fill_ratio"] > 0.78 and props["ink_pixel_count"] > 3000:
        return (
            False,
            "SOLID_OR_PHOTO",
            "Non-Digit Graphic or Photo Detected",
            "The input has excessive solid coverage or photo texture. Only clean, single handwritten digit strokes are supported.",
        )

    # 3. Check for too many disconnected strokes/letters (e.g. words or complex scribbles)
    if props["num_components"] >= 5:
        return (
            False,
            "COMPLEX_DRAWING",
            "Complex Drawing or Multiple Characters Detected",
            f"Detected {props['num_components']} disconnected parts. Please draw only ONE single digit with connected or standard strokes.",
        )

    # 4. Check for excessive closed loops (> 2 holes is invalid for single digits)
    # Digits have at most 2 holes ('8'). Letters/drawings like '%', '&', complex shapes often have 3+
    if props["holes_count"] >= 3:
        return (
            False,
            "INVALID_TOPOLOGY",
            "Non-Digit Symbol or Shape Detected",
            f"The drawing contains {props['holes_count']} enclosed loops. Single digits (0–9) have at most 2 loops (e.g., '8').",
        )

    # 5. Check for extreme aspect ratio (wide multi-digits like '72', '10')
    if props["aspect_ratio"] > 1.45 and props["width"] > 60:
        return (
            False,
            "MULTI_DIGIT",
            "Multi-Digit Number Detected",
            "Detected a wide aspect ratio typical of multi-digit numbers (like 10, 72, 99). Only single digits (0–9) are supported.",
        )

    # 6. Explicit Alphabet Geometric Pattern Recognition (B, D, E, F, H, T, A, etc.)
    if "cropped_binary" in props:
        is_alpha, alpha_title, alpha_desc = detect_alphabet_patterns(props["cropped_binary"])
        if is_alpha:
            return (
                False,
                "ALPHABET_DETECTED",
                f"Alphabet Letter Detected ({alpha_title})",
                alpha_desc,
            )

    # 7. Ensemble Consensus & Out-of-Distribution (OOD) Validation
    if ensemble_pred is not None:
        confidence = ensemble_pred.get("confidence", 0.0)
        probabilities = ensemble_pred.get("probabilities", [])
        individual_preds = ensemble_pred.get("individual_preds", {})

        votes = [v["digit"] for v in individual_preds.values() if "digit" in v]
        unique_votes = set(votes)

        # Probabilities margin analysis
        if len(probabilities) > 0:
            sorted_probs = sorted(probabilities, reverse=True)
            top1_prob = sorted_probs[0]
            top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            margin = top1_prob - top2_prob

            # Ambiguous input or classifier divergence (common for alphabets like A, M, K, X, doodles)
            if top1_prob < 0.45 and (len(unique_votes) >= 2 or margin < 0.12):
                return (
                    False,
                    "OOD_NON_DIGIT",
                    "Unrecognized / Non-Digit Input Detected",
                    "The model cannot identify a clear handwritten digit (0–9). Non-digit characters (like alphabet letters A-Z, symbols, or doodles) are not supported.",
                )

            # Strict lower confidence bound
            if confidence < 0.40:
                return (
                    False,
                    "LOW_CONFIDENCE",
                    "Ambiguous Non-Digit Drawing",
                    f"Prediction confidence ({confidence*100:.1f}%) is too low to match a valid digit. Please draw a clear, single digit (0–9).",
                )

    return (True, "VALID", "Valid Digit", "Input is a valid single digit.")
