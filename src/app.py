import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import predict_digit, load_model, MODEL_PATH
from preprocessing import preprocess_image
from validation import validate_digit_input
from ui_helpers import apply_custom_css, render_template

# -----------------------------
# Page Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition Ensemble",
    page_icon="🔢",
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
        Combines 3 classifiers using **Calibrated Soft Voting**:
        
        1. 🔹 **Logistic Regression** (L-BFGS) - *Weight: 1*
        2. 🌲 **Random Forest** (150 Trees) - *Weight: 2*
        3. ⚡ **Support Vector Machine** (RBF Kernel) - *Weight: 4*
        
        ---
        **Input Format**: Single digit (0 to 9)  
        **Multi-digit & Non-digit Restriction**: Enabled  
        **Features**: 784 normalized pixel values  
        **Model File**: `models/digit_ensemble.pkl`
        """
    )
    
    if os.path.exists(MODEL_PATH):
        st.success("✅ Model loaded and ready.")
    else:
        st.error("⚠️ Model file not found. Please train first.")

    st.markdown("---")
    st.markdown("### ⚙️ Drawing Settings")
    stroke_width = st.slider("Brush Stroke Width", min_value=10, max_value=32, value=18, step=2)

# -----------------------------
# Header & Use Cases Guide
# -----------------------------
render_template("header.html")

with st.expander("📖 View Supported Use Cases & Validation Rules", expanded=False):
    render_template("use_cases_guide.html")

# -----------------------------
# Model Check
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
    # -----------------------------
    # Input Tabs: Draw, Upload, Test Samples
    # -----------------------------
    tab_draw, tab_upload, tab_samples = st.tabs([
        "✏️ Draw Digit",
        "📤 Upload Image",
        "🎲 Test Sample Digits"
    ])

    selected_image = None
    input_source = None

    # --- TAB 1: DRAW DIGIT ---
    with tab_draw:
        st.write("Draw a **single digit (0-9)** in the black canvas below:")
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
                key="digit_draw_canvas",
            )

        with col_canvas_info:
            st.info(
                "💡 **Drawing Rules & Tips:**\n"
                "- ⚠️ **Restriction**: Only **single digits (0–9)** are accepted.\n"
                "- 🚫 **Disallowed**: Letters/alphabets (A-Z), multi-digits (10, 72), symbols, or doodles.\n"
                "- Draw your single digit centered on the canvas.\n"
                "- Adjust brush thickness in the sidebar.\n"
                "- Use the trash icon below canvas to reset."
            )
            
            # Check if canvas has drawing
            if canvas_result is not None:
                try:
                    img_data = canvas_result.image_data
                    if img_data is not None and np.max(img_data[:, :, :3]) > 20:
                        selected_image = Image.fromarray(img_data.astype("uint8"), "RGBA")
                        input_source = "Drawn Digit Canvas"
                except Exception:
                    pass

    # --- TAB 2: UPLOAD IMAGE ---
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a single digit image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            key="digit_file_uploader",
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            input_source = "Uploaded Image"

    # --- TAB 3: TEST SAMPLES ---
    with tab_samples:
        st.write("Click any sample digit button to test immediately:")
        cols = st.columns(10)
        
        digit_paths = {
            0: [(70, 30), (45, 70), (70, 110), (95, 70), (70, 30)],
            1: [(55, 45), (70, 30), (70, 110)],
            2: [(45, 45), (70, 30), (95, 50), (45, 110), (95, 110)],
            3: [(45, 35), (95, 35), (70, 68), (95, 85), (60, 110)],
            4: [(85, 110), (85, 30), (40, 85), (105, 85)],
            5: [(95, 35), (45, 35), (45, 68), (95, 75), (60, 110)],
            6: [(90, 35), (50, 70), (50, 95), (90, 95), (90, 70), (50, 70)],
            7: [(45, 35), (95, 35), (60, 110)],
            8: [(70, 30), (50, 50), (70, 70), (90, 90), (70, 110), (50, 90), (70, 70), (90, 50), (70, 30)],
            9: [(90, 70), (50, 70), (50, 45), (90, 45), (90, 110), (55, 110)],
        }

        for i in range(10):
            with cols[i]:
                if st.button(f"**{i}**", key=f"sample_btn_{i}", use_container_width=True):
                    canvas = Image.new("L", (140, 140), color=0)
                    draw = ImageDraw.Draw(canvas)
                    path = digit_paths.get(i, [(70, 30), (70, 110)])
                    for p_idx in range(len(path) - 1):
                        draw.line([path[p_idx], path[p_idx + 1]], fill=255, width=14)
                    
                    st.session_state["active_sample"] = canvas
                    st.session_state["sample_label"] = i

        if "active_sample" in st.session_state and tab_samples:
            selected_image = st.session_state["active_sample"]
            input_source = f"Sample Digit {st.session_state.get('sample_label', '')}"

    # -----------------------------
    # Validation & Prediction Pipeline
    # -----------------------------
    if selected_image is not None:
        st.markdown("---")
        
        # Step 1: Pre-prediction structural validation (geometric, stroke, topology)
        is_struct_valid, err_type, err_title, err_desc = validate_digit_input(selected_image)

        if not is_struct_valid:
            col_img, col_warn = st.columns([1, 2.5], gap="large")
            with col_img:
                st.subheader("🖼️ Input Image")
                st.image(selected_image, caption=input_source or "Input Image", width=200)
            
            with col_warn:
                render_template("validation_error_box.html", title=err_title, description=err_desc)
        else:
            # Step 2: Compute ML predictions
            digit, confidence, probabilities, individual_preds = predict_digit(selected_image)

            # Step 3: Out-of-Distribution (OOD) & Ensemble Consensus Validation
            is_valid_digit, ood_type, ood_title, ood_desc = validate_digit_input(
                selected_image,
                ensemble_pred={
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "individual_preds": individual_preds,
                }
            )

            if not is_valid_digit:
                col_img, col_warn = st.columns([1, 2.5], gap="large")
                with col_img:
                    st.subheader("🖼️ Input Image")
                    st.image(selected_image, caption=input_source or "Input Image", width=200)
                
                with col_warn:
                    render_template("validation_error_box.html", title=ood_title, description=ood_desc)
            else:
                # Valid Single Digit Flow
                col_img, col_pred, col_prob = st.columns([1, 1.2, 1.8], gap="large")

                with col_img:
                    st.subheader("🖼️ Input Preview")
                    st.image(selected_image, caption=input_source or "Input Image", width=200)
                    
                    # Show 28x28 Preprocessed Grayscale view (what the ML model actually receives)
                    preprocessed = preprocess_image(selected_image)
                    preprocessed_view = (preprocessed.reshape(28, 28) * 255).astype(np.uint8)
                    st.caption("28×28 Preprocessed (MNIST format)")
                    st.image(preprocessed_view, width=110)

                with col_pred:
                    st.subheader("🎯 Ensemble Result")
                    render_template(
                        "prediction_box.html",
                        digit=digit,
                        confidence=confidence * 100
                    )

                    # Individual Classifier Breakdown
                    if individual_preds:
                        st.markdown("**Individual Classifier Votes:**")
                        c1, c2, c3 = st.columns(3)
                        names = {"lr": "Logistic Reg", "rf": "Random Forest", "svm": "SVM"}
                        cols_map = {"lr": c1, "rf": c2, "svm": c3}
                        for k, col in cols_map.items():
                            if k in individual_preds:
                                item = individual_preds[k]
                                with col:
                                    render_template(
                                        "model_card.html",
                                        name=names[k],
                                        digit=item['digit'],
                                        confidence=item['confidence'] * 100
                                    )

                with col_prob:
                    st.subheader("📊 Class Probabilities (0-9)")
                    df_prob = pd.DataFrame({
                        "Digit": [str(d) for d in range(10)],
                        "Probability (%)": [round(p * 100, 2) for p in probabilities]
                    })
                    st.bar_chart(df_prob.set_index("Digit"), height=260)
                    
                    # Top 3 ranked candidates
                    top3_idx = np.argsort(probabilities)[::-1][:3]
                    top3_text = " | ".join([
                        f"**#{rank+1}** Digit **{idx}** ({probabilities[idx]*100:.1f}%)" 
                        for rank, idx in enumerate(top3_idx)
                    ])
                    st.markdown(f"**Top Predictions:** {top3_text}")
    else:
        st.info("👆 Draw a single digit on the canvas, upload an image, or click a sample digit above to see the ensemble prediction!")