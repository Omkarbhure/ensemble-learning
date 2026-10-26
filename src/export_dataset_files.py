import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_NPZ = os.path.join(MODELS_DIR, "character_dataset.npz")
PREVIEWS_DIR = os.path.join(DATA_DIR, "sample_images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREVIEWS_DIR, exist_ok=True)

# 48 class mapping
CANONICAL_NAMES = {
    **{i: str(i) for i in range(10)},
    **{10 + i: chr(65 + i) for i in range(26)},
    36: "plus",
    37: "minus",
    38: "multiply",
    39: "divide",
    40: "equal",
    41: "question",
    42: "exclamation",
    43: "hashtag",
    44: "at_sign",
    45: "dollar",
    46: "percent",
    47: "ampersand",
}

SYMBOL_CHAR = {
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


def export_files():
    print("=" * 60)
    print("  Exporting 48-Class Dataset to Visible Project Files")
    print("=" * 60)

    if not os.path.exists(DATASET_NPZ):
        raise FileNotFoundError(f"Missing compiled dataset at {DATASET_NPZ}")

    data = np.load(DATASET_NPZ)
    X = data["X"]
    y = data["y"]

    total_samples = len(y)
    num_classes = len(np.unique(y))
    print(f"Loaded {total_samples:,} samples across {num_classes} classes from {DATASET_NPZ}")

    csv_sample_path = os.path.join(DATA_DIR, "dataset_preview.csv")
    csv_full_path = os.path.join(DATA_DIR, "dataset_48_classes.csv")

    pixel_cols = [f"pixel_{i}" for i in range(784)]
    X_int = (X * 255.0).astype(np.uint8)

    chars = []
    categories = []
    for label in y:
        if label < 10:
            chars.append(str(label))
            categories.append("Digit")
        elif label < 36:
            chars.append(chr(65 + (label - 10)))
            categories.append("Alphabet")
        else:
            chars.append(SYMBOL_CHAR.get(label, "Symbol"))
            categories.append("Symbol")

    df = pd.DataFrame(X_int, columns=pixel_cols)
    df.insert(0, "category", categories)
    df.insert(0, "character", chars)
    df.insert(0, "label", y)

    # Save preview CSV (50 samples per class)
    sample_df = df.groupby("label", group_keys=False).apply(lambda g: g.head(50))
    sample_df.to_csv(csv_sample_path, index=False)
    print(f"  * Created preview CSV: {csv_sample_path} ({round(os.path.getsize(csv_sample_path) / (1024*1024), 2)} MB)")

    # Save full CSV
    df.to_csv(csv_full_path, index=False)
    print(f"  * Created full CSV   : {csv_full_path} ({round(os.path.getsize(csv_full_path) / (1024*1024), 2)} MB)")

    # Export sample PNGs for all classes
    for class_id in range(num_classes):
        folder_name = CANONICAL_NAMES.get(class_id, f"class_{class_id}")
        class_folder = os.path.join(PREVIEWS_DIR, folder_name)
        os.makedirs(class_folder, exist_ok=True)

        class_indices = np.where(y == class_id)[0][:5]
        for idx_num, sample_idx in enumerate(class_indices):
            img_arr = X_int[sample_idx].reshape(28, 28)
            img = Image.fromarray(img_arr, mode="L")
            img_path = os.path.join(class_folder, f"sample_{idx_num + 1}.png")
            img.save(img_path)

    print(f"  * Exported PNG images for all {num_classes} classes to: {PREVIEWS_DIR}")

    # Create README.md in data folder
    data_readme = os.path.join(DATA_DIR, "README.md")
    with open(data_readme, "w", encoding="utf-8") as f:
        f.write(f"""# 📁 Dataset Files

This directory contains the dataset files for the Handwritten Character & Symbol Recognition project.

## 📄 Files & Folders

1. **`dataset_48_classes.csv`**: Full dataset containing {total_samples:,} samples across 48 classes.
   - Format: `label, character, category, pixel_0 ... pixel_783`
   - Pixel values: normalized 0–255 grayscale (28×28 pixels).
2. **`dataset_preview.csv`**: Balanced subset (50 samples per class) for quick opening and inspection in Excel / VS Code.
3. **`sample_images/`**: Subfolders for each character and symbol containing actual 28×28 PNG images.

## 🔤 Classes Supported (48 Total)

- **Digits (0–9)**: Classes 0 to 9.
- **Alphabets (A–Z & a–z)**: Classes 10 to 35.
- **Symbols (+, -, *, /, =, ?, !, #, @, $, %, &)**: Classes 36 to 47.
""")
    print(f"  * Created documentation: {data_readme}")
    print("\nDone! Dataset files are up to date.")


if __name__ == "__main__":
    export_files()
