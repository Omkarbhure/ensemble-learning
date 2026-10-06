import numpy as np
from PIL import Image, ImageOps
import scipy.ndimage


def preprocess_training_data(X):
    """
    Normalize training dataset pixel values from 0-255 to 0.0-1.0.
    """
    X = X.astype("float32")
    X = X / 255.0
    return X


def detect_multiple_digits(image):
    """
    Detect if the user drew or uploaded a multi-digit number (e.g., 10, 72, 99).

    Uses Connected Component Analysis (CCA) and aspect-ratio validation:
      - Returns (is_multi_digit: bool, count: int, reason: str)
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

    # Invert light background if needed
    corner_pixels = [
        img_array[0, 0],
        img_array[0, -1],
        img_array[-1, 0],
        img_array[-1, -1],
    ]
    if np.mean(corner_pixels) > 128 or np.mean(img_array) > 130:
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # Threshold binary ink strokes
    binary = (img_array > 35).astype(np.uint8)
    if not np.any(binary):
        return False, 0, "No drawing detected"

    bbox = image.getbbox()
    if bbox is None:
        return False, 0, "Empty canvas"

    total_w = bbox[2] - bbox[0]
    total_h = bbox[3] - bbox[1]

    # Dilate slightly to group strokes of the same character
    structure = np.ones((6, 6), dtype=int)
    dilated = scipy.ndimage.binary_dilation(binary, structure=structure)
    labeled, num_features = scipy.ndimage.label(dilated)

    components = []
    min_area = (total_w * total_h) * 0.04
    for i in range(1, num_features + 1):
        mask = (labeled == i)
        area = np.sum(mask)
        if area >= max(30, min_area):
            y_indices, x_indices = np.where(mask)
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            components.append({
                "id": i,
                "area": area,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "w": x_max - x_min,
                "h": y_max - y_min,
            })

    # Sort components horizontally from left to right
    components = sorted(components, key=lambda c: c["x_min"])

    # Check 1: Multiple distinct horizontally separated characters
    if len(components) >= 2:
        distinct_digits = 1
        last_c = components[0]
        for c in components[1:]:
            overlap_x = max(
                0, min(last_c["x_max"], c["x_max"]) - max(last_c["x_min"], c["x_min"])
            )
            min_w = min(last_c["w"], c["w"])
            # If horizontal overlap is small, these are distinct side-by-side digits
            if overlap_x < min_w * 0.35:
                distinct_digits += 1
                last_c = c
        if distinct_digits >= 2:
            return (
                True,
                distinct_digits,
                f"Multiple separate digits detected ({distinct_digits} digits side-by-side).",
            )

    # Check 2: Wide bounding box aspect ratio (typical of multi-digit numbers like 10, 72, 99)
    if total_w > total_h * 1.35 and total_w > 50:
        return (
            True,
            2,
            "Wide drawing aspect ratio detected. Multi-digit numbers (e.g. 2-digit numbers) are restricted.",
        )

    return False, 1, "Single digit"


def preprocess_image(image, invert_if_needed=True):
    """
    Enhanced MNIST-standard Preprocessing Pipeline:
      1. Converts to Grayscale & composites alpha channel.
      2. Auto-detects light background and inverts to dark background (MNIST format).
      3. Cleans stray background noise.
      4. Bounding-box cropping to isolate the digit regardless of canvas position.
      5. Aspect-ratio preserving scaling into a 20x20 bounding box.
      6. Center-of-Mass (Centroid) alignment into the 28x28 spatial grid.
      7. Gaussian ink-stroke antialiasing filter to match scanned pen stroke gradients.
      8. Dynamic range normalization [0.0, 1.0] and flatten to (1, 784).
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

    # Convert to grayscale
    image = image.convert("L")

    # Check background brightness: MNIST digits are white (255) on black (0) background
    img_array = np.array(image)
    corner_pixels = [
        img_array[0, 0],
        img_array[0, -1],
        img_array[-1, 0],
        img_array[-1, -1],
    ]
    if invert_if_needed and (np.mean(corner_pixels) > 128 or np.mean(img_array) > 130):
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # Filter out faint background noise/compression artifacts
    img_array[img_array < 30] = 0
    image = Image.fromarray(img_array)

    # Find the tight bounding box of the drawn digit
    bbox = image.getbbox()
    if bbox is None:
        # Empty canvas
        return np.zeros((1, 784), dtype=np.float32)

    # Tightly crop to digit boundaries
    cropped = image.crop(bbox)
    width, height = cropped.size

    # Scale to fit inside 20x20 box preserving aspect ratio (Yann LeCun MNIST specification)
    if width > height:
        new_width = 20
        new_height = max(1, int(round((height * 20.0) / width)))
    else:
        new_height = 20
        new_width = max(1, int(round((width * 20.0) / height)))

    resized = cropped.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Place inside standard 28x28 canvas centered
    canvas_28 = Image.new("L", (28, 28), color=0)
    offset_x = (28 - new_width) // 2
    offset_y = (28 - new_height) // 2
    canvas_28.paste(resized, (offset_x, offset_y))

    # Center-of-Mass Alignment (matches MNIST dataset creation)
    arr_28 = np.array(canvas_28, dtype=np.float32)
    cy, cx = scipy.ndimage.center_of_mass(arr_28)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = 14.0 - cy
        shift_x = 14.0 - cx
        arr_28 = scipy.ndimage.shift(arr_28, (shift_y, shift_x), mode="constant", cval=0.0)

    # Gaussian smoothing to simulate scanned pen ink gradients matching MNIST
    arr_28 = scipy.ndimage.gaussian_filter(arr_28, sigma=0.65)

    # Peak normalization
    if arr_28.max() > 0:
        arr_28 = (arr_28 / arr_28.max()) * 255.0

    # Clip values and normalize to 0.0 - 1.0
    arr_28 = np.clip(arr_28, 0.0, 255.0) / 255.0

    return arr_28.reshape(1, 784)