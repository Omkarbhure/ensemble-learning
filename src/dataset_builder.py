import os
import sys
import io
import random
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.datasets import fetch_openml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_image

BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
DATASET_PATH = os.path.join(MODELS_DIR, "character_dataset.npz")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 12 Special Symbols
SYMBOLS = ["+", "-", "*", "/", "=", "?", "!", "#", "@", "$", "%", "&"]

# 48 Classes mapping (Digits, Alphabets A/a through Z/z, Special Symbols)
LABEL_MAP = {
    # 0 - 9: Digits
    **{i: str(i) for i in range(10)},
    # 10 - 35: Alphabets (Dual-case)
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


def generate_font_glyph(char_str, font_path, size=55, canvas_size=(110, 110)):
    """Render a character (letter or symbol) with a font and apply handwriting-like augmentations."""
    img = Image.new("L", canvas_size, color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char_str, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = canvas_size
    x = (w - tw) // 2 + random.randint(-4, 4)
    y = (h - th) // 2 + random.randint(-4, 4)
    draw.text((x, y), char_str, font=font, fill=255)

    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

    # Random stroke dilation/blur
    if random.random() > 0.4:
        img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
    return img


def generate_handwritten_variant(char, canvas_size=(100, 100)):
    """
    Generate realistic procedural strokes for characters prone to confusion:
    'A' (with and without crossbar ^), '1' (plain line vs flag/base),
    '!' (stem + distinct gap + dot), 'D', 'U', 'M'.
    """
    w, h = canvas_size
    img = Image.new("L", canvas_size, color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2 + random.randint(-4, 4), h // 2 + random.randint(-4, 4)
    stroke = random.randint(5, 9)

    if char in ("A", "a"):
        top_y = cy - random.randint(28, 38)
        bot_y = cy + random.randint(28, 38)
        left_x = cx - random.randint(18, 30)
        right_x = cx + random.randint(18, 30)
        apex_x = cx + random.randint(-4, 4)

        draw.line([(left_x, bot_y), (apex_x, top_y)], fill=255, width=stroke)
        draw.line([(apex_x, top_y), (right_x, bot_y)], fill=255, width=stroke)

        if random.random() > 0.30:
            bar_y = (top_y + bot_y) // 2 + random.randint(-4, 6)
            bx0 = left_x + (apex_x - left_x) * (bar_y - bot_y) / (top_y - bot_y)
            bx1 = right_x + (apex_x - right_x) * (bar_y - bot_y) / (top_y - bot_y)
            draw.line([(bx0, bar_y), (bx1, bar_y)], fill=255, width=stroke)

    elif char == "1":
        top_y = cy - random.randint(28, 38)
        bot_y = cy + random.randint(28, 38)
        draw.line([(cx, top_y), (cx, bot_y)], fill=255, width=stroke)
        if random.random() > 0.4:
            draw.line([(cx - random.randint(10, 18), top_y + random.randint(8, 16)), (cx, top_y)], fill=255, width=stroke)
        if random.random() > 0.6:
            base_w = random.randint(12, 22)
            draw.line([(cx - base_w, bot_y), (cx + base_w, bot_y)], fill=255, width=stroke)

    elif char == "!":
        top_y = cy - random.randint(28, 38)
        bot_stem_y = cy + random.randint(6, 14)
        draw.line([(cx, top_y), (cx, bot_stem_y)], fill=255, width=stroke)
        dot_cy = bot_stem_y + random.randint(12, 18)
        dot_r = max(2, stroke // 2 + 1)
        draw.ellipse([(cx - dot_r, dot_cy - dot_r), (cx + dot_r, dot_cy + dot_r)], fill=255)

    elif char in ("D", "d"):
        top_y = cy - random.randint(28, 36)
        bot_y = cy + random.randint(28, 36)
        left_x = cx - random.randint(14, 22)
        draw.line([(left_x, top_y), (left_x, bot_y)], fill=255, width=stroke)
        draw.arc([(left_x - 10, top_y), (cx + random.randint(16, 26), bot_y)], start=270, end=90, fill=255, width=stroke)

    elif char in ("U", "u"):
        top_y = cy - random.randint(28, 36)
        bot_y = cy + random.randint(24, 34)
        left_x = cx - random.randint(16, 26)
        right_x = cx + random.randint(16, 26)
        draw.line([(left_x, top_y), (left_x, bot_y - 12)], fill=255, width=stroke)
        draw.line([(right_x, top_y), (right_x, bot_y - 12)], fill=255, width=stroke)
        draw.arc([(left_x, bot_y - 24), (right_x, bot_y)], start=0, end=180, fill=255, width=stroke)

    elif char in ("M", "m"):
        top_y = cy - random.randint(22, 34)
        bot_y = cy + random.randint(22, 34)
        w_span = random.randint(24, 38)
        x0 = cx - w_span // 2
        x1 = cx + w_span // 2
        xm = cx
        draw.line([(x0, bot_y), (x0, top_y)], fill=255, width=stroke)
        draw.line([(x0, top_y), (xm, cy + random.randint(0, 10))], fill=255, width=stroke)
        draw.line([(xm, cy + random.randint(0, 10)), (x1, top_y)], fill=255, width=stroke)
        draw.line([(x1, top_y), (x1, bot_y)], fill=255, width=stroke)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def generate_procedural_symbol(sym, canvas_size=(64, 64)):
    """Draw a natural handwritten-like symbol with random variations."""
    w, h = canvas_size
    img = Image.new("L", canvas_size, color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2 + random.randint(-4, 4), h // 2 + random.randint(-4, 4)
    stroke = random.randint(4, 8)

    if sym == "+":
        arm_h = random.randint(14, 24)
        arm_v = random.randint(14, 24)
        angle = random.uniform(-10, 10) * np.pi / 180.0
        x0, y0 = cx - arm_h * np.cos(angle), cy - arm_h * np.sin(angle)
        x1, y1 = cx + arm_h * np.cos(angle), cy + arm_h * np.sin(angle)
        draw.line([(x0, y0), (x1, y1)], fill=255, width=stroke)
        draw.line([(cx, cy - arm_v), (cx, cy + arm_v)], fill=255, width=stroke)

    elif sym == "-":
        arm = random.randint(16, 26)
        angle = random.uniform(-8, 8) * np.pi / 180.0
        x0, y0 = cx - arm * np.cos(angle), cy - arm * np.sin(angle)
        x1, y1 = cx + arm * np.cos(angle), cy + arm * np.sin(angle)
        draw.line([(x0, y0), (x1, y1)], fill=255, width=stroke)

    elif sym == "=":
        arm = random.randint(16, 24)
        gap = random.randint(7, 12)
        angle = random.uniform(-6, 6) * np.pi / 180.0
        dx = arm * np.cos(angle)
        dy = arm * np.sin(angle)
        draw.line([(cx - dx, cy - gap - dy), (cx + dx, cy - gap + dy)], fill=255, width=stroke)
        draw.line([(cx - dx, cy + gap - dy), (cx + dx, cy + gap + dy)], fill=255, width=stroke)

    elif sym == "/":
        arm = random.randint(18, 28)
        tilt = random.uniform(0.75, 1.15)
        draw.line([(cx - arm * tilt, cy + arm), (cx + arm * tilt, cy - arm)], fill=255, width=stroke)

    elif sym == "*":
        arm = random.randint(12, 20)
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=255, width=stroke)
        draw.line([(cx - arm * 0.7, cy - arm * 0.7), (cx + arm * 0.7, cy + arm * 0.7)], fill=255, width=stroke)
        draw.line([(cx - arm * 0.7, cy + arm * 0.7), (cx + arm * 0.7, cy - arm * 0.7)], fill=255, width=stroke)
        if random.random() > 0.5:
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill=255, width=stroke)

    elif sym == "!":
        arm = random.randint(16, 24)
        draw.line([(cx, cy - arm), (cx, cy + arm // 4)], fill=255, width=stroke)
        dot_r = max(2, stroke // 2 + 1)
        draw.ellipse([(cx - dot_r, cy + arm - dot_r), (cx + dot_r, cy + arm + dot_r)], fill=255)

    elif sym == "?":
        r = random.randint(10, 16)
        draw.arc([(cx - r, cy - r * 1.6), (cx + r, cy - r * 0.2)], start=180, end=0, fill=255, width=stroke)
        draw.line([(cx + r, cy - r * 0.8), (cx, cy)], fill=255, width=stroke)
        draw.line([(cx, cy), (cx, cy + int(r * 0.4))], fill=255, width=stroke)
        dot_r = max(2, stroke // 2 + 1)
        draw.ellipse([(cx - dot_r, cy + int(r * 1.3) - dot_r), (cx + dot_r, cy + int(r * 1.3) + dot_r)], fill=255)

    elif sym == "#":
        span = random.randint(14, 22)
        gap = random.randint(5, 9)
        tilt = random.uniform(-0.15, 0.15)
        draw.line([(cx - gap + tilt * span, cy - span), (cx - gap - tilt * span, cy + span)], fill=255, width=stroke)
        draw.line([(cx + gap + tilt * span, cy - span), (cx + gap - tilt * span, cy + span)], fill=255, width=stroke)
        draw.line([(cx - span, cy - gap), (cx + span, cy - gap)], fill=255, width=stroke)
        draw.line([(cx - span, cy + gap), (cx + span, cy + gap)], fill=255, width=stroke)

    elif sym == "@":
        inner_r = random.randint(6, 9)
        draw.ellipse([(cx - inner_r, cy - inner_r), (cx + inner_r, cy + inner_r)], outline=255, width=stroke)
        draw.line([(cx + inner_r, cy - inner_r), (cx + inner_r, cy + inner_r)], fill=255, width=stroke)
        outer_r = inner_r + random.randint(7, 12)
        draw.arc([(cx - outer_r, cy - outer_r), (cx + outer_r, cy + outer_r)], start=50, end=320, fill=255, width=stroke)

    elif sym == "$":
        span = random.randint(16, 24)
        draw.arc([(cx - span * 0.6, cy - span), (cx + span * 0.6, cy)], start=90, end=300, fill=255, width=stroke)
        draw.arc([(cx - span * 0.6, cy), (cx + span * 0.6, cy + span)], start=270, end=120, fill=255, width=stroke)
        draw.line([(cx, cy - span - 4), (cx, cy + span + 4)], fill=255, width=stroke)

    elif sym == "%":
        span = random.randint(16, 24)
        draw.line([(cx - span * 0.7, cy + span), (cx + span * 0.7, cy - span)], fill=255, width=stroke)
        dot_r = random.randint(3, 5)
        draw.ellipse([(cx - span * 0.5 - dot_r, cy - span * 0.6 - dot_r), (cx - span * 0.5 + dot_r, cy - span * 0.6 + dot_r)], outline=255, width=max(2, stroke - 1))
        draw.ellipse([(cx + span * 0.5 - dot_r, cy + span * 0.6 - dot_r), (cx + span * 0.5 + dot_r, cy + span * 0.6 + dot_r)], outline=255, width=max(2, stroke - 1))

    elif sym == "&":
        span = random.randint(14, 22)
        draw.arc([(cx - span * 0.5, cy - span), (cx + span * 0.5, cy - span * 0.1)], start=0, end=360, fill=255, width=stroke)
        draw.arc([(cx - span * 0.7, cy - span * 0.2), (cx + span * 0.7, cy + span)], start=0, end=360, fill=255, width=stroke)
        draw.line([(cx - span * 0.4, cy + span * 0.8), (cx + span * 0.8, cy + span)], fill=255, width=stroke)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def build_character_dataset(samples_per_class=700, random_state=42):
    """
    Build a balanced 48-class dataset:
      - 10 digits (0-9) from MNIST (fully normalized through preprocess_image)
      - 26 letters (A-Z and a-z combined) from EMNIST + procedural & font augmentations
      - 12 symbols (+, -, *, /, =, ?, !, #, @, $, %, &) from procedural & font generation
    All samples are normalized via preprocess_image to ensure consistent feature space.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    print("=" * 65)
    print("  Building Calibrated 48-Class Dataset (Digits, Letters, 12 Symbols)")
    print("=" * 65)

    all_X = []
    all_y = []

    available_fonts = []
    font_names = ["arial.ttf", "calibri.ttf", "comic.ttf", "times.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf", "cour.ttf"]
    for fn in font_names:
        fp = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(fp):
            available_fonts.append(fp)

    # 1. Load MNIST Digits (0 - 9) and pass each through preprocess_image
    print(f"\n[1/3] Loading {samples_per_class} samples for each Digit (0-9)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    mnist_X = mnist.data.astype("float32")
    mnist_y = mnist.target.astype(int)

    for digit in range(10):
        idx = np.where(mnist_y == digit)[0]
        selected = np.random.choice(idx, samples_per_class, replace=False)
        digit_imgs = []
        for s_idx in selected:
            raw_28 = mnist_X[s_idx].reshape(28, 28).astype(np.uint8)
            pil_img = Image.fromarray(raw_28, mode="L")
            feat = preprocess_image(pil_img, invert_if_needed=False)
            digit_imgs.append(feat[0])

        all_X.append(np.array(digit_imgs))
        all_y.extend([digit] * samples_per_class)
        print(f"  * Digit '{digit}': {samples_per_class} samples")

    # 2. Load EMNIST Letters (A-Z and a-z combined -> class 10 - 35)
    parquet_path = os.path.join(DATA_CACHE_DIR, "emnist_letters_train.parquet")
    print(f"\n[2/3] Loading {samples_per_class} samples for each Letter (Capital + Small + Procedural)...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing EMNIST parquet file at {parquet_path}")

    df_emnist = pd.read_parquet(parquet_path)
    emnist_count = 320
    procedural_count = 80
    font_upper_count = 150
    font_lower_count = samples_per_class - emnist_count - procedural_count - font_upper_count

    for letter_idx in range(26):
        upper_char = chr(65 + letter_idx)
        lower_char = chr(97 + letter_idx)
        class_id = 10 + letter_idx

        letter_imgs = []

        sub_df = df_emnist[df_emnist["label"] == letter_idx]
        selected_rows = sub_df.sample(emnist_count, replace=False, random_state=random_state)
        for _, row in selected_rows.iterrows():
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
            feat = preprocess_image(img, invert_if_needed=False)
            letter_imgs.append(feat[0])

        for _ in range(procedural_count):
            img = generate_handwritten_variant(upper_char)
            feat = preprocess_image(img, invert_if_needed=False)
            letter_imgs.append(feat[0])

        for _ in range(font_upper_count):
            fp = random.choice(available_fonts) if available_fonts else None
            fsize = random.randint(46, 68)
            img = generate_font_glyph(upper_char, fp, size=fsize)
            feat = preprocess_image(img, invert_if_needed=False)
            letter_imgs.append(feat[0])

        for _ in range(font_lower_count):
            fp = random.choice(available_fonts) if available_fonts else None
            fsize = random.randint(42, 64)
            img = generate_font_glyph(lower_char, fp, size=fsize)
            feat = preprocess_image(img, invert_if_needed=False)
            letter_imgs.append(feat[0])

        all_X.append(np.array(letter_imgs))
        all_y.extend([class_id] * len(letter_imgs))
        print(f"  * Letter '{upper_char}/{lower_char}' (Class {class_id}): {len(letter_imgs)} samples")

    # 3. Generate 12 Symbols (+, -, *, /, =, ?, !, #, @, $, %, & -> class 36 - 47)
    print(f"\n[3/3] Generating {samples_per_class} samples for each Symbol (12 symbols total)...")
    for s_idx, sym in enumerate(SYMBOLS):
        class_id = 36 + s_idx
        sym_imgs = []
        for _ in range(samples_per_class):
            if sym == "!" and random.random() > 0.4:
                raw_img = generate_handwritten_variant("!")
            elif sym == "@" and available_fonts and random.random() > 0.15:
                fp = random.choice(available_fonts)
                fsize = random.randint(48, 72)
                raw_img = generate_font_glyph(sym, fp, size=fsize)
            elif available_fonts and random.random() > 0.35:
                fp = random.choice(available_fonts)
                fsize = random.randint(48, 70)
                raw_img = generate_font_glyph(sym, fp, size=fsize)
            else:
                raw_img = generate_procedural_symbol(sym)

            feat = preprocess_image(raw_img, invert_if_needed=False)
            sym_imgs.append(feat[0])

        all_X.append(np.array(sym_imgs))
        all_y.extend([class_id] * samples_per_class)
        print(f"  * Symbol '{sym}' (Class {class_id}): {samples_per_class} samples")

    # Combine into single arrays
    X = np.vstack(all_X).astype("float32")
    y = np.array(all_y, dtype=int)

    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    print("\n" + "-" * 65)
    print(f"Total balanced dataset: {len(y):,} samples across {len(LABEL_MAP)} classes.")
    print(f"Feature matrix shape  : {X.shape}")
    print(f"Saving to {DATASET_PATH}...")

    np.savez_compressed(DATASET_PATH, X=X, y=y)
    print(f"Dataset successfully saved! ({round(os.path.getsize(DATASET_PATH) / (1024*1024), 2)} MB)")
    return X, y


if __name__ == "__main__":
    build_character_dataset(samples_per_class=700)
