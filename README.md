# 🔢 Handwritten Digit Recognition Using Soft Voting Ensemble

A machine learning web application for handwritten digit recognition ($0–9$) using a **Soft Voting Ensemble** combining Logistic Regression, Random Forest, and Support Vector Machines (SVM) trained on the MNIST dataset, with dedicated UI/UX separation (HTML/CSS) and real-time input validation.

---

## 🌟 Key Features

- **🧠 Calibrated Ensemble Model**:
  - 🔹 **Logistic Regression** (L-BFGS) - *Weight: 1*
  - 🌲 **Random Forest** (150 Trees) - *Weight: 2*
  - ⚡ **Support Vector Machine** (RBF Kernel with probability calibration) - *Weight: 4*
- **🎨 Modular UI/UX Architecture**:
  - Externalized CSS (`src/static/css/style.css`)
  - Clean HTML component templates (`src/templates/*.html`)
- **🛡️ Strict Multi-Layer Validation**:
  - **Single Digit Enforcement**: Rejects multi-digit inputs (e.g., 10, 72, 99).
  - **Alphabet & Non-Digit Rejection**: Rejects alphabet letters (A-Z, such as `B`, `D`, `H`, `T`, `A`), geometric shapes, doodles, and photos.
  - **Out-of-Distribution (OOD) Detection**: Classifier divergence and entropy analysis to filter non-digit drawings.
- **✏️ Multiple Input Modes**:
  - Interactive Canvas for freehand drawing with brush size controls.
  - Image file uploader (PNG, JPG, JPEG).
  - Quick test sample buttons for instant verification ($0–9$).
- **📊 Real-time Visualizations**:
  - Preprocessed 28×28 MNIST view.
  - Individual classifier breakdown cards.
  - Full class probability distribution bar charts.

---

## 📂 Project Structure

```
handwritten-digit-ensemble/
├── models/
│   └── digit_ensemble.pkl        # Trained ensemble model artifact
├── src/
│   ├── static/
│   │   └── css/
│   │       └── style.css          # Frontend UI/UX styling
│   ├── templates/
│   │   ├── header.html           # Main header template
│   │   ├── model_card.html       # Individual classifier vote card
│   │   ├── prediction_box.html   # Main ensemble prediction display
│   │   ├── restriction_box.html  # Multi-digit warning card
│   │   ├── use_cases_guide.html  # Supported use cases guide
│   │   └── validation_error_box.html # Validation rejection card
│   ├── app.py                    # Streamlit application
│   ├── predict.py                # Prediction & model loading
│   ├── preprocessing.py          # MNIST centering & normalization pipeline
│   ├── train.py                  # Training pipeline for 3 classifiers + voting
│   ├── ui_helpers.py             # Template & CSS loader
│   └── validation.py             # Stroke, geometric, loop & OOD validator
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Omkarbhure/ensemble-learning.git
cd ensemble-learning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (if needed)
```bash
python src/train.py
```

### 4. Run the Web Application
```bash
streamlit run src/app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 📜 License
MIT License.
