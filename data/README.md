# 📁 Dataset Files

This directory contains the dataset files for the Handwritten Character & Symbol Recognition project.

## 📄 Files & Folders

1. **`dataset_48_classes.csv`**: Full dataset containing 33,600 samples across 48 classes.
   - Format: `label, character, category, pixel_0 ... pixel_783`
   - Pixel values: normalized 0–255 grayscale (28×28 pixels).
2. **`dataset_preview.csv`**: Balanced subset (50 samples per class) for quick opening and inspection in Excel / VS Code.
3. **`sample_images/`**: Subfolders for each character and symbol containing actual 28×28 PNG images.

## 🔤 Classes Supported (48 Total)

- **Digits (0–9)**: Classes 0 to 9.
- **Alphabets (A–Z & a–z)**: Classes 10 to 35.
- **Symbols (+, -, *, /, =, ?, !, #, @, $, %, &)**: Classes 36 to 47.
