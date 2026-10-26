# 🔤 Handwritten Character & Symbol Recognition Using Soft Voting Ensemble

A machine learning web application for recognizing handwritten characters ($43$ classes total: Digits $0–9$, Uppercase & Lowercase Letters $A–Z$, and Math/Special Symbols $+$, $-$, $*$, $/$, $=$, $?$, $!$) using a **Calibrated Soft Voting Ensemble** combining Logistic Regression, Random Forest, and a Deep Multi-Layer Perceptron Neural Network (MLP), with dedicated UI/UX separation (HTML/CSS) and real-time input validation.

---

## 🌟 Key Features

- **🧠 Calibrated Multi-Class Ensemble Model**:
  - 🔹 **Logistic Regression** (L-BFGS) - *Default Weight: 1*
  - 🌲 **Random Forest** (120 Trees) - *Default Weight: 3*
  - ⚡ **Deep Neural Network (MLP)** (384×192 with ReLU and Adam) - *Default Weight: 5*
- **🎛️ Live Ensemble Sandbox**:
  - Dynamically adjust classifier weights ($0–10$) on the fly with real-time recalculation (< 5ms).
  - Instant toggle between **Soft Voting** (weighted probabilities) and **Hard Voting** (majority rule).
- **📊 Model Performance & Benchmark Analytics Dashboard**:
  - Comparative metrics: Accuracy, Precision, Recall, F1 Score, and per-sample latency for all models.
  - Interactive comparison bar charts and category accuracy breakdowns (Digits vs Letters vs Symbols).
  - Ambiguous twin / confused pairs table based on empirical test set evaluation.
- **🔤 48 Supported Classes**:
  - 🔢 **Digits (0–9)**: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
  - 🔡 **Alphabets (A–Z & a–z)**: Dual-case recognition for all 26 letters (`A / a` through `Z / z`)
  - ➕ **Symbols**: `+`, `-`, `*`, `/`, `=`, `?`, `!`, `#`, `@`, `$`, `%`, `&`
- **🎨 Modular UI/UX Architecture**:
  - Externalized CSS (`src/static/css/style.css`)
  - Clean HTML component templates (`src/templates/*.html`)
- **🛡️ Multi-Layer Validation & Disambiguation**:
  - Single Character Enforcement for single-character mode.
  - Morphological Disambiguation resolving homoglyphs (`B` vs `8`, `D` vs `0`, `6` vs `0`, `!` vs `1`, `P` vs `9`).
  - Out-of-Distribution (OOD) Filtering for unrecognizable scribbles.

---

## 📂 Project Structure

```
handwritten-digit-ensemble/
├── data/
│   ├── dataset_48_classes.csv    # Full dataset in CSV format (33,600 rows x 787 columns)
│   ├── dataset_preview.csv       # Preview dataset (2,400 rows) for quick opening in Excel/VS Code
│   ├── sample_images/            # 48 character folders with PNG samples for visual inspection
│   └── README.md                 # Dataset documentation and schema
├── models/
│   ├── benchmark_metrics.json    # Precomputed empirical evaluation & benchmark metrics
│   ├── character_dataset.npz     # Balanced 48-class compiled numpy dataset (33,600 samples)
│   ├── character_ensemble.pkl    # Trained 48-class ensemble model
│   └── digit_ensemble.pkl        # Compatibility model artifact
├── src/
│   ├── static/
│   │   └── css/
│   │       └── style.css         # Frontend UI/UX styling
│   ├── templates/
│   │   ├── header.html           # Main header template
│   │   ├── model_card.html       # Individual classifier vote card
│   │   ├── prediction_box.html   # Main ensemble prediction display
│   │   ├── restriction_box.html  # Multi-character warning card
│   │   ├── use_cases_guide.html  # Supported use cases guide
│   │   └── validation_error_box.html # Validation rejection card
│   ├── app.py                    # Streamlit web application
│   ├── benchmark.py              # Performance evaluation and benchmark generator
│   ├── dataset_builder.py        # Compiles balanced 48-class dataset
│   ├── disambiguation.py         # Morphological twin character disambiguation
│   ├── export_dataset_files.py   # Exports dataset into data/ CSV and image files
│   ├── predict.py                # Prediction, dynamic weights & label decoder
│   ├── preprocessing.py          # 28x28 centering & normalization pipeline
│   ├── train.py                  # Training pipeline for 3 classifiers + voting
│   ├── ui_helpers.py             # Template & CSS loader
│   └── validation.py             # Stroke, geometric, loop & OOD validator
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Dataset & Train the Ensemble Model
```bash
python src/dataset_builder.py
python src/train.py
```

### 3. (Optional) Export Dataset to CSV & Image Files
```bash
python src/export_dataset_files.py
```

### 4. Run the Web Application
```bash
streamlit run src/app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 📜 License
MIT License.
