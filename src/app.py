import os
import sys
import importlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import disambiguation
importlib.reload(disambiguation)

import predict
importlib.reload(predict)
from predict import predict_digit, load_model, MODEL_PATH, CHARACTER_MODEL_PATH, LABEL_MAP

import preprocessing
importlib.reload(preprocessing)
from preprocessing import preprocess_image

import validation
importlib.reload(validation)
from validation import validate_digit_input

from ui_helpers import apply_custom_css, render_template

import benchmark
importlib.reload(benchmark)
from benchmark import load_benchmark_metrics, generate_benchmark_metrics

# -----------------------------
# Page Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Handwritten Character & Math Recognition Ensemble",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply external CSS from static/css/style.css
apply_custom_css("style.css")

# -----------------------------
# Sidebar: System & Architecture
# -----------------------------
with st.sidebar:
    st.header("🧠 Ensemble Architecture")
    st.markdown(
        """
        Combines 3 heterogeneous classifiers:
        1. 🔹 **Logistic Regression** (Linear)
        2. 🌲 **Random Forest** (120 Trees)
        3. ⚡ **Neural Network (MLP)** (384×192)
        
        ---
        **Supported Classes (48 total)**:
        - 🔢 **Digits**: `0` to `9` (10 classes)
        - 🔤 **Alphabets**: `A–Z` & `a–z` (26 dual-case classes)
        - ➕ **Symbols**: `+`, `-`, `*`, `/`, `=`, `?`, `!`, `#`, `@`, `$`, `%`, `&` (12 classes)
        """
    )

    active_model_path = CHARACTER_MODEL_PATH if os.path.exists(CHARACTER_MODEL_PATH) else MODEL_PATH
    if os.path.exists(active_model_path):
        st.success("✅ 48-Class Ensemble Model loaded and ready.")
    else:
        st.error("⚠️ Model file not found. Please train first.")

    st.markdown("---")
    st.header("🎛️ Live Ensemble Sandbox")
    st.caption("Dynamically adjust model weights & consensus rule in real-time:")

    voting_mode_choice = st.radio(
        "Consensus Voting Rule",
        ["Soft Voting (Probabilistic)", "Hard Voting (Majority)"],
        index=0,
        help="Soft voting computes weighted class probabilities. Hard voting tallies discrete model predictions."
    )
    voting_mode = "soft" if "Soft" in voting_mode_choice else "hard"

    st.markdown("**Classifier Weights:**")
    w_lr = st.slider("🔹 Logistic Regression Weight", min_value=0, max_value=10, value=1, step=1)
    w_rf = st.slider("🌲 Random Forest Weight", min_value=0, max_value=10, value=3, step=1)
    w_mlp = st.slider("⚡ Deep Neural Net Weight", min_value=0, max_value=10, value=5, step=1)

    total_w = w_lr + w_rf + w_mlp
    if total_w > 0:
        pct_lr = int((w_lr / total_w) * 100)
        pct_rf = int((w_rf / total_w) * 100)
        pct_mlp = int((w_mlp / total_w) * 100)
        st.info(f"📊 **Weight Ratio:** `LR {pct_lr}%` | `RF {pct_rf}%` | `MLP {pct_mlp}%`")
    else:
        st.warning("⚠️ All weights are 0! Defaulting to equal weighting (1:1:1).")

    custom_weights = {"lr": w_lr, "rf": w_rf, "mlp": w_mlp}

    st.markdown("---")
    st.markdown("### ⚙️ Drawing Settings")
    stroke_width = st.slider("Brush Stroke Width", min_value=10, max_value=32, value=18, step=2)

# -----------------------------
# Header & Use Cases Guide
# -----------------------------
render_template("header.html")

with st.expander("📖 View Supported Characters & Validation Rules", expanded=False):
    render_template("use_cases_guide.html")

# -----------------------------
# Model Verification Check
# -----------------------------
try:
    _ = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.warning(
        f"⚠️ **Trained model not found!**\n\n"
        f"Please run the training script from the terminal:\n"
        f"```bash\npython src/train.py\n```"
    )

if model_ready:
    # Top-Level Navigation Tabs
    app_tab_single, app_tab_bench = st.tabs([
        "✏️ Single Character Recognition",
        "📊 Model Benchmarks & Analytics"
    ])

    # =========================================================
    # TAB 1: SINGLE CHARACTER RECOGNITION
    # =========================================================
    with app_tab_single:
        # Input Tabs: Draw, Upload, Test Samples
        tab_draw, tab_upload, tab_samples = st.tabs([
            "✏️ Draw Character / Symbol",
            "📤 Upload Image",
            "🎲 Test Samples"
        ])

        selected_image = None
        input_source = None

        # --- Sub-tab 1: Draw ---
        with tab_draw:
            st.write("Draw a single **digit (0-9)**, **letter (Capital or Small)**, or **symbol (+, -, *, /, =, ?, !, #, @, $, %, &)**:")
            col_canvas, col_canvas_info = st.columns([1, 1], gap="medium")

            with col_canvas:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=stroke_width,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    return_image_data=True,
                    update_streamlit=True,
                    key="char_draw_canvas",
                )

            with col_canvas_info:
                st.info(
                    "💡 **Drawing Tips:**\n"
                    "- 🔢 **Digits**: `0` through `9`\n"
                    "- 🔤 **Letters**: Both **Capital** (`A–Z`) and **Small** (`a–z`) are recognized!\n"
                    "- ➕ **Symbols**: `+`, `-`, `*`, `/`, `=`, `?`, `!`, `#`, `@`, `$`, `%`, `&`\n"
                    "- Draw a single character centered in the box.\n"
                    "- Multi-character strings (e.g. words like `cat` or numbers like `72`) are restricted."
                )

                if canvas_result is not None:
                    try:
                        img_data = canvas_result.image_data
                        if img_data is not None and np.max(img_data[:, :, :3]) > 20:
                            selected_image = Image.fromarray(img_data.astype("uint8"), "RGBA")
                            input_source = "Drawn Canvas"
                            if "active_sample" in st.session_state:
                                del st.session_state["active_sample"]
                    except Exception:
                        pass

        # --- Sub-tab 2: Upload ---
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload an image containing a single character (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                key="char_file_uploader",
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                input_source = "Uploaded Image"
                if "active_sample" in st.session_state:
                    del st.session_state["active_sample"]

        # --- Sub-tab 3: Test Samples ---
        with tab_samples:
            st.write("Click any sample button below to test immediate recognition:")

            st.caption("🔢 **Sample Digits (0–9)**")
            d_cols = st.columns(10)
            sample_digits = [str(i) for i in range(10)]
            for i, d in enumerate(sample_digits):
                with d_cols[i]:
                    if st.button(f"**{d}**", key=f"sample_digit_{d}", use_container_width=True):
                        img = Image.new("L", (140, 140), 0)
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 90)
                        except Exception:
                            font = ImageFont.load_default()
                        bbox = draw.textbbox((0, 0), d, font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(((140 - w) // 2, (140 - h) // 2 - 10), d, font=font, fill=255)
                        st.session_state["active_sample"] = img
                        st.session_state["sample_label"] = f"Digit '{d}'"

            st.caption("🔠 **Sample Capital Letters (A–Z)**")
            sample_upper = ["A", "B", "C", "D", "E", "H", "M", "R", "S", "Z"]
            u_cols = st.columns(len(sample_upper))
            for i, l in enumerate(sample_upper):
                with u_cols[i]:
                    if st.button(f"**{l}**", key=f"sample_upper_{l}", use_container_width=True):
                        img = Image.new("L", (140, 140), 0)
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 85)
                        except Exception:
                            font = ImageFont.load_default()
                        bbox = draw.textbbox((0, 0), l, font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(((140 - w) // 2, (140 - h) // 2 - 10), l, font=font, fill=255)
                        st.session_state["active_sample"] = img
                        st.session_state["sample_label"] = f"Capital Letter '{l}'"

            st.caption("🔡 **Sample Small Letters (a–z)**")
            sample_lower = ["a", "b", "c", "d", "e", "h", "m", "r", "s", "z"]
            low_cols = st.columns(len(sample_lower))
            for i, l in enumerate(sample_lower):
                with low_cols[i]:
                    if st.button(f"**{l}**", key=f"sample_lower_{l}", use_container_width=True):
                        img = Image.new("L", (140, 140), 0)
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 85)
                        except Exception:
                            font = ImageFont.load_default()
                        bbox = draw.textbbox((0, 0), l, font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(((140 - w) // 2, (140 - h) // 2 - 10), l, font=font, fill=255)
                        st.session_state["active_sample"] = img
                        st.session_state["sample_label"] = f"Small Letter '{l}'"

            st.caption("➕ **Sample Symbols (12 total)**")
            sample_symbols = ["+", "-", "*", "/", "=", "?", "!", "#", "@", "$", "%", "&"]
            s_cols = st.columns(len(sample_symbols))
            for i, s in enumerate(sample_symbols):
                with s_cols[i]:
                    btn_label = f"**\\***" if s == "*" else f"**{s}**"
                    if st.button(btn_label, key=f"sample_symbol_{i}", use_container_width=True):
                        img = Image.new("L", (140, 140), 0)
                        draw = ImageDraw.Draw(img)
                        try:
                            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 90)
                        except Exception:
                            font = ImageFont.load_default()
                        bbox = draw.textbbox((0, 0), s, font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        draw.text(((140 - w) // 2, (140 - h) // 2 - 10), s, font=font, fill=255)
                        st.session_state["active_sample"] = img
                        st.session_state["sample_label"] = f"Symbol '{s}'"

            if "active_sample" in st.session_state and selected_image is None:
                selected_image = st.session_state["active_sample"]
                input_source = st.session_state.get("sample_label", "Sample Image")

        # --- Prediction Pipeline ---
        if selected_image is not None:
            st.markdown("---")

            # Step 1: Pre-prediction structural validation
            is_struct_valid, err_type, err_title, err_desc = validate_digit_input(selected_image)

            if not is_struct_valid:
                col_img, col_warn = st.columns([1, 2.5], gap="large")
                with col_img:
                    st.subheader("🖼️ Input Image")
                    st.image(selected_image, caption=input_source or "Input Image", width=200)

                with col_warn:
                    render_template("validation_error_box.html", title=err_title, description=err_desc)
            else:
                # Step 2: Compute Ensemble Predictions with Dynamic Weights & Mode
                predicted_label, confidence, probabilities, individual_preds, top_candidates, category = predict_digit(
                    selected_image,
                    weights=custom_weights,
                    voting_mode=voting_mode
                )

                # Step 3: Confidence / OOD Validation
                is_valid_char, ood_type, ood_title, ood_desc = validate_digit_input(
                    selected_image,
                    ensemble_pred={"confidence": confidence, "probabilities": probabilities}
                )

                if not is_valid_char:
                    col_img, col_warn = st.columns([1, 2.5], gap="large")
                    with col_img:
                        st.subheader("🖼️ Input Image")
                        st.image(selected_image, caption=input_source or "Input Image", width=200)

                    with col_warn:
                        render_template("validation_error_box.html", title=ood_title, description=ood_desc)
                else:
                    col_img, col_pred, col_prob = st.columns([1, 1.2, 1.8], gap="large")

                    with col_img:
                        st.subheader("🖼️ Input Preview")
                        st.image(selected_image, caption=input_source or "Input Image", width=200)

                        # Show 28x28 Preprocessed view
                        preprocessed = preprocess_image(selected_image)
                        preprocessed_view = (preprocessed.reshape(28, 28) * 255).astype(np.uint8)
                        st.caption("28×28 Preprocessed Feature Space")
                        st.image(preprocessed_view, width=110)

                    with col_pred:
                        st.subheader("🎯 Ensemble Result")
                        cat_upper = category.upper()
                        cat_lower = category.lower()
                        cat_detail = "🔤 Alphabet Letter (A–Z / a–z)" if category == "Alphabet" else ("🔢 Single Digit (0–9)" if category == "Digit" else "➕ Math & Special Symbol")
                        render_template(
                            "prediction_box.html",
                            digit=predicted_label,
                            category=cat_upper,
                            category_upper=cat_upper,
                            category_lower=cat_lower,
                            category_detail=cat_detail,
                            confidence=confidence * 100,
                        )

                        # Individual Classifier Breakdown
                        if individual_preds:
                            st.markdown(f"**Individual Classifier Votes ({voting_mode.title()} Voting):**")
                            c1, c2, c3 = st.columns(3)
                            names = {"lr": "Logistic Reg", "rf": "Random Forest", "mlp": "Neural Net"}
                            cols_map = {"lr": c1, "rf": c2, "mlp": c3}
                            for k, col in cols_map.items():
                                if k in individual_preds:
                                    item = individual_preds[k]
                                    with col:
                                        render_template(
                                            "model_card.html",
                                            name=names[k],
                                            digit=item["digit"],
                                            confidence=item["confidence"] * 100,
                                            category=item.get("category", ""),
                                        )

                    with col_prob:
                        st.subheader("📊 Top Candidates")
                        top8_idx = np.argsort(probabilities)[-8:][::-1]
                        top8_labels = [LABEL_MAP.get(i, str(i)) for i in top8_idx]
                        top8_probs = [round(float(probabilities[i]) * 100, 2) for i in top8_idx]

                        df_top = pd.DataFrame({
                            "Character": top8_labels,
                            "Probability (%)": top8_probs
                        })
                        st.bar_chart(df_top.set_index("Character"), height=260)

                        top_text = " | ".join([
                            f"**#{r+1}** `{char}` ({prob*100:.1f}%)"
                            for r, (char, prob) in enumerate(top_candidates[:4])
                        ])
                        st.markdown(f"**Top Candidates:** {top_text}")
        else:
            st.info("👆 Draw a character/symbol, upload an image, or click any sample button above to test recognition!")

    # =========================================================
    # TAB 2: BENCHMARK & PERFORMANCE ANALYTICS
    # =========================================================
    with app_tab_bench:
        st.subheader("📊 Model Performance & Benchmark Analytics")
        st.write(
            "Empirical evaluation and comparative analysis across all base classifiers and the "
            "Soft Voting Ensemble on held-out test data."
        )

        # Load cached benchmark metrics
        try:
            bench_data = load_benchmark_metrics()
        except Exception:
            with st.spinner("Generating benchmark evaluation metrics..."):
                bench_data = generate_benchmark_metrics()

        models_meta = bench_data.get("models", {})

        # KPI Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        lr_info = models_meta.get("lr", {})
        with col_m1:
            st.markdown(
                f"""
                <div class="bench-card">
                    <div class="bench-title">Logistic Regression</div>
                    <div class="bench-val">{lr_info.get('accuracy', 0)}%</div>
                    <div class="bench-sub">Latency: {lr_info.get('latency_ms_per_sample', 0)} ms/sample</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        rf_info = models_meta.get("rf", {})
        with col_m2:
            st.markdown(
                f"""
                <div class="bench-card">
                    <div class="bench-title">Random Forest (120 Trees)</div>
                    <div class="bench-val">{rf_info.get('accuracy', 0)}%</div>
                    <div class="bench-sub">Latency: {rf_info.get('latency_ms_per_sample', 0)} ms/sample</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        mlp_info = models_meta.get("mlp", {})
        with col_m3:
            st.markdown(
                f"""
                <div class="bench-card">
                    <div class="bench-title">Deep Neural Net (MLP)</div>
                    <div class="bench-val">{mlp_info.get('accuracy', 0)}%</div>
                    <div class="bench-sub">Latency: {mlp_info.get('latency_ms_per_sample', 0)} ms/sample</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        ens_info = models_meta.get("ensemble", {})
        with col_m4:
            st.markdown(
                f"""
                <div class="bench-card bench-card-highlight">
                    <div class="bench-title">⭐ Soft Voting Ensemble</div>
                    <div class="bench-val" style="color: #4ADE80;">{ens_info.get('accuracy', 0)}%</div>
                    <div class="bench-sub">Precision: {ens_info.get('precision', 0)}% | F1: {ens_info.get('f1_score', 0)}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        col_chart1, col_chart2 = st.columns([1.5, 1], gap="large")

        with col_chart1:
            st.markdown("##### 📈 Model Accuracy & Metric Comparison")
            df_compare = pd.DataFrame({
                "Model": [
                    "Logistic Regression",
                    "Random Forest",
                    "Neural Net (MLP)",
                    "Soft Voting Ensemble"
                ],
                "Accuracy (%)": [
                    lr_info.get("accuracy", 0),
                    rf_info.get("accuracy", 0),
                    mlp_info.get("accuracy", 0),
                    ens_info.get("accuracy", 0),
                ],
                "Precision (%)": [
                    lr_info.get("precision", 0),
                    rf_info.get("precision", 0),
                    mlp_info.get("precision", 0),
                    ens_info.get("precision", 0),
                ],
                "F1 Score (%)": [
                    lr_info.get("f1_score", 0),
                    rf_info.get("f1_score", 0),
                    mlp_info.get("f1_score", 0),
                    ens_info.get("f1_score", 0),
                ]
            })
            st.bar_chart(df_compare.set_index("Model")["Accuracy (%)"], height=280)

        with col_chart2:
            st.markdown("##### 🏷️ Category Accuracy Breakdown")
            cats = bench_data.get("categories", {})
            if cats:
                df_cats = pd.DataFrame([
                    {"Category": k, "Accuracy (%)": v["accuracy"]}
                    for k, v in cats.items()
                ])
                st.bar_chart(df_cats.set_index("Category"), height=280)

        st.markdown("---")
        st.markdown("##### 🔍 Top Ambiguous & Confused Pairs")
        st.write("Visual twins and homoglyphs with the highest confusion frequency in evaluation:")

        confused_pairs = bench_data.get("top_confused_pairs", [])
        if confused_pairs:
            df_conf = pd.DataFrame(confused_pairs)
            df_conf.columns = ["True Character", "Mispredicted Character", "Error Count"]
            st.dataframe(df_conf, use_container_width=True)

        if st.button("🔄 Recompute Benchmarks on Test Set", key="btn_recompute_bench"):
            with st.spinner("Running full evaluation across test set..."):
                new_data = generate_benchmark_metrics()
                st.success("✅ Benchmarks updated successfully!")
                st.rerun()