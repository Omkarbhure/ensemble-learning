import numpy as np
from PIL import Image, ImageOps
import scipy.ndimage


def analyze_image_properties(image):
    """
    Extract geometric, topological, and stroke properties of the input image.
    Supports single digits (0-9), alphabets (A-Z), and symbols (+, -, *, /, =, ?, !).
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
    if bbox is None or ink_pixel_count < 20:
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

    # Aspect ratio
    aspect_ratio = total_w / max(1, total_h)

    # Check horizontal distribution of components (to detect multi-character strings like "AB" or "12")
    components = []
    min_area = (total_w * total_h) * 0.03
    for i in range(1, num_components + 1):
        mask = (labeled == i)
        area = np.sum(mask)
        if area >= max(25, min_area):
            y_indices, x_indices = np.where(mask)
            components.append({
                "x_min": int(np.min(x_indices)),
                "x_max": int(np.max(x_indices)),
                "w": int(np.max(x_indices) - np.min(x_indices)),
                "h": int(np.max(y_indices) - np.min(y_indices)),
            })

    # Sort components left to right
    components = sorted(components, key=lambda c: c["x_min"])
    distinct_horizontal = 1
    if len(components) >= 2:
        last_c = components[0]
        for c in components[1:]:
            overlap_x = max(0, min(last_c["x_max"], c["x_max"]) - max(last_c["x_min"], c["x_min"]))
            min_w = min(last_c["w"], c["w"])
            if overlap_x < min_w * 0.20:
                distinct_horizontal += 1
                last_c = c

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
        "distinct_horizontal": distinct_horizontal,
    }


def validate_digit_input(image, ensemble_pred=None):
    """
    Validate that the input is a single character (digit, alphabet, or symbol).
    Rejects:
      - Empty canvas / blanks
      - Multi-character sequences (e.g. '10', 'AB', '++')
      - Non-drawing graphics or photos (solid blocks, photographic textures)
      - Unrecognizable scribbles (extreme low confidence)

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
            "The canvas or uploaded image is blank. Please draw a single character (digit 0–9, letter A–Z, or symbol +, -, *, /, =, ?, !) or upload an image.",
        )

    # 2. Check for solid blocks / excessive ink / photo clutter
    if props["fill_ratio"] > 0.82 and props["ink_pixel_count"] > 4000:
        return (
            False,
            "SOLID_OR_PHOTO",
            "Graphic or Photo Detected",
            "The input has excessive solid coverage or photo texture. Please draw or upload clean handwritten character strokes.",
        )

    # 3. Check for multiple distinct side-by-side characters
    if props.get("distinct_horizontal", 1) >= 2 and props["width"] > 70 and props["aspect_ratio"] > 1.6:
        return (
            False,
            "MULTI_CHARACTER",
            "Multiple Characters Detected",
            "Detected multiple side-by-side characters (e.g., multi-digit numbers or words). Please draw only ONE single character or symbol at a time.",
        )

    # 4. Check for extreme clutter / scribbles (> 6 disconnected stroke fragments)
    if props["num_components"] > 6:
        return (
            False,
            "COMPLEX_SCRIBBLE",
            "Complex Scribble Detected",
            f"Detected {props['num_components']} disconnected stroke fragments. Please draw a single clean character or symbol.",
        )

    # 5. Out-of-Distribution (OOD) Validation with Model Probabilities
    if ensemble_pred is not None:
        confidence = ensemble_pred.get("confidence", 0.0)

        # Confidence sanity threshold for 43 classes (random guessing = 1/43 ≈ 2.3%)
        if confidence < 0.16:
            return (
                False,
                "LOW_CONFIDENCE",
                "Unrecognized Drawing",
                f"The ensemble confidence ({confidence*100:.1f}%) is too low. Please draw a clearer single digit (0–9), letter (A–Z), or symbol (+, -, *, /, =, ?, !).",
            )

    return (True, "VALID", "Valid Character", "Input is a valid single character or symbol.")
