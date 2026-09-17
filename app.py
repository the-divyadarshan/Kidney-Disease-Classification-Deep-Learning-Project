import os
import json
import streamlit as st
from PIL import Image

from cnnClassifier.pipeline.prediction import PredictionPipeline


# =========================================================
# ENVIRONMENT
# =========================================================

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="KidneyVision AI | Kidney Disease Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html {
        scroll-behavior: smooth;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(14, 165, 233, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(168, 85, 247, 0.10),
                transparent 35%
            ),
            #060914;

        color: #ffffff;
        font-family: "Inter", sans-serif;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 0.5rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* =====================================================
       NAVBAR
    ===================================================== */

    .navbar-custom {
        padding: 16px 0;
        margin-bottom: 20px;

        background: rgba(6, 9, 20, 0.82);

        border-bottom:
            1px solid rgba(255, 255, 255, 0.07);

        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);

        border-radius: 0 0 18px 18px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-logo {
        width: 44px;
        height: 44px;

        border-radius: 14px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 21px;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

        box-shadow:
            0 8px 30px
            rgba(99, 102, 241, 0.35);
    }

    .brand-name {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-name span {
        color: #818cf8;
    }

    .nav-pill {
        display: inline-block;

        padding: 8px 13px;

        margin-left: 6px;

        border-radius: 30px;

        color: #8d98ad;

        font-size: 10px;
        font-weight: 700;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid rgba(255,255,255,0.06);
    }

    .nav-pill-icon {
        color: #818cf8;
    }

    .nav-status {
        display: inline-flex;

        align-items: center;
        gap: 7px;

        padding: 8px 13px;

        border-radius: 30px;

        color: #86efac;

        background:
            rgba(34, 197, 94, 0.07);

        border:
            1px solid rgba(34, 197, 94, 0.16);

        font-size: 10px;
        font-weight: 700;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:
            0 0 12px #22c55e;
    }


    /* =====================================================
       HERO
    ===================================================== */

    .hero {
        padding: 55px 0 45px;
    }

    .hero-badge {
        display: inline-flex;

        align-items: center;
        gap: 8px;

        padding: 8px 15px;

        border-radius: 30px;

        background:
            rgba(99, 102, 241, 0.08);

        border:
            1px solid rgba(129, 140, 248, 0.20);

        color: #a5b4fc;

        font-size: 10px;
        font-weight: 800;

        letter-spacing: 1px;

        text-transform: uppercase;

        margin-bottom: 20px;
    }

    .hero-title {
        max-width: 900px;

        font-size: clamp(42px, 6vw, 70px);

        line-height: 1.02;

        font-weight: 800;

        letter-spacing: -4px;

        margin: 0;
    }

    .gradient-text {
        background:
            linear-gradient(
                90deg,
                #818cf8,
                #c084fc,
                #38bdf8
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        background-clip: text;
    }

    .hero-description {
        max-width: 680px;

        color: #8c96aa;

        font-size: 14px;

        line-height: 1.8;

        margin-top: 22px;
    }

    .hero-stats {
        display: flex;

        gap: 9px;

        margin-top: 27px;

        flex-wrap: wrap;
    }

    .hero-stat {
        padding: 10px 14px;

        border-radius: 12px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid rgba(255,255,255,0.06);

        color: #9ca7bb;

        font-size: 10px;

        font-weight: 600;
    }

    .hero-stat-icon {
        color: #818cf8;
        margin-right: 6px;
    }


    /* =====================================================
       MAIN GLASS CARD
    ===================================================== */

    .glass-card {

        background:
            rgba(13, 18, 34, 0.75);

        border:
            1px solid rgba(255,255,255,0.08);

        border-radius: 28px;

        box-shadow:
            0 30px 90px rgba(0,0,0,0.35),
            inset 0 1px 0
            rgba(255,255,255,0.04);

        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);

        padding: 28px;

    }


    /* =====================================================
       SECTION HEADINGS
    ===================================================== */

    .section-label {

        color: #818cf8;

        font-size: 9px;

        font-weight: 800;

        letter-spacing: 1.5px;

        text-transform: uppercase;

        margin-bottom: 7px;
    }

    .section-title {

        font-size: 21px;

        font-weight: 700;

        margin-bottom: 5px;
    }

    .section-subtitle {

        color: #737e94;

        font-size: 11px;

        margin-bottom: 20px;
    }


    /* =====================================================
       UPLOAD AREA
    ===================================================== */

    .upload-container {

        min-height: 420px;

        border-radius: 21px;

        border:
            1.5px dashed
            rgba(129,140,248,0.28);

        background:

            radial-gradient(
                circle at center,
                rgba(99,102,241,0.08),
                transparent 65%
            ),

            rgba(3,7,18,0.40);

        display: flex;

        align-items: center;

        justify-content: center;

        text-align: center;

        padding: 25px;

    }

    .upload-icon {

        width: 76px;
        height: 76px;

        border-radius: 23px;

        display: flex;

        align-items: center;
        justify-content: center;

        margin: 0 auto 18px;

        background:
            rgba(99,102,241,0.10);

        border:
            1px solid
            rgba(129,140,248,0.18);

        color: #818cf8;

        font-size: 30px;
    }

    .upload-title {

        font-size: 17px;

        font-weight: 700;

        margin-bottom: 8px;
    }

    .upload-description {

        color: #68748a;

        font-size: 11px;

        margin-bottom: 20px;
    }


    /* =====================================================
       STREAMLIT FILE UPLOADER
    ===================================================== */

    [data-testid="stFileUploader"] {

        width: 100%;
    }

    [data-testid="stFileUploaderDropzone"] {

        background:
            rgba(3,7,18,0.30) !important;

        border:
            1.5px dashed
            rgba(129,140,248,0.30) !important;

        border-radius: 18px !important;

        padding: 20px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {

        color: #8c96aa !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {

        color: #a5b4fc !important;
    }

    [data-testid="stFileUploaderDropzone"] button {

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            ) !important;

        color: white !important;

        border: none !important;

        border-radius: 11px !important;
    }


    /* =====================================================
       IMAGE PREVIEW
    ===================================================== */

    .image-preview-card {

        background:
            rgba(3,7,18,0.55);

        border:
            1px solid
            rgba(255,255,255,0.055);

        border-radius: 21px;

        padding: 12px;

        min-height: 420px;

        display: flex;

        align-items: center;

        justify-content: center;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    .stButton > button {

        width: 100%;

        border: none !important;

        border-radius: 13px !important;

        padding: 13px 21px !important;

        color: #ffffff !important;

        font-size: 12px !important;

        font-weight: 700 !important;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            ) !important;

        box-shadow:
            0 10px 28px
            rgba(99,102,241,0.25);

        transition: 0.25s ease;
    }

    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 15px 35px
            rgba(99,102,241,0.38);
    }


    /* =====================================================
       RESULT CARD
    ===================================================== */

    .result-container {

        min-height: 420px;

        border-radius: 21px;

        background:
            rgba(3,7,18,0.42);

        border:
            1px solid
            rgba(255,255,255,0.055);

        padding: 22px;

        overflow: auto;
    }

    .result-placeholder {

        min-height: 350px;

        display: flex;

        align-items: center;

        justify-content: center;

        text-align: center;

        color: #5f6a80;
    }

    .result-placeholder-icon {

        width: 70px;
        height: 70px;

        border-radius: 20px;

        display: flex;

        align-items: center;
        justify-content: center;

        margin: 0 auto 16px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid
            rgba(255,255,255,0.06);

        font-size: 26px;
    }

    .result-placeholder-text {

        max-width: 260px;

        font-size: 11px;

        line-height: 1.7;

        margin: auto;
    }


    /* =====================================================
       PREDICTION RESULT
    ===================================================== */

    .prediction-result {

        padding: 20px;

        border-radius: 15px;

        background:
            rgba(99,102,241,0.07);

        border:
            1px solid
            rgba(129,140,248,0.13);

        margin-top: 15px;
    }

    .prediction-heading {

        color: #a5b4fc;

        font-size: 10px;

        font-weight: 800;

        letter-spacing: 1px;

        text-transform: uppercase;

        margin-bottom: 12px;
    }

    .prediction-value {

        color: #ffffff;

        font-size: 22px;

        font-weight: 700;

        margin-bottom: 10px;
    }

    .prediction-json {

        color: #d1d5db;

        font-family: monospace;

        font-size: 11px;

        line-height: 1.7;

        white-space: pre-wrap;

        word-break: break-word;
    }


    /* =====================================================
       DEVELOPER CARD
    ===================================================== */

    .developer-card {

        margin-top: 25px;

        padding: 18px 21px;

        border-radius: 20px;

        display: flex;

        align-items: center;

        justify-content: space-between;

        gap: 20px;

        background:

            linear-gradient(
                135deg,
                rgba(99,102,241,0.07),
                rgba(14,165,233,0.04)
            );

        border:
            1px solid
            rgba(255,255,255,0.07);
    }

    .developer-info {

        display: flex;

        align-items: center;

        gap: 15px;
    }

    .developer-avatar {

        width: 64px;
        height: 64px;

        min-width: 64px;

        border-radius: 50%;

        display: flex;

        align-items: center;

        justify-content: center;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

        border:
            2px solid
            rgba(129,140,248,0.45);

        color: white;

        font-size: 20px;

        font-weight: 800;

        box-shadow:
            0 8px 25px
            rgba(0,0,0,0.35);
    }

    .developer-label {

        color: #69758b;

        font-size: 9px;

        font-weight: 700;

        letter-spacing: 1px;

        text-transform: uppercase;

        margin-bottom: 3px;
    }

    .developer-name {

        font-size: 14px;

        font-weight: 700;

        margin-bottom: 3px;
    }

    .developer-role {

        color: #818cf8;

        font-size: 10px;

        font-weight: 600;
    }

    .developer-tech {

        display: flex;

        gap: 7px;

        flex-wrap: wrap;

        justify-content: flex-end;
    }

    .tech-tag {

        padding: 7px 10px;

        border-radius: 8px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid
            rgba(255,255,255,0.06);

        color: #8c96aa;

        font-size: 9px;

        font-weight: 600;
    }


    /* =====================================================
       INFO CARDS
    ===================================================== */

    .info-section {

        margin-top: 25px;
    }

    .info-card {

        height: 100%;

        padding: 21px;

        border-radius: 18px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.055);

        transition: 0.25s ease;
    }

    .info-card:hover {

        transform: translateY(-3px);

        background:
            rgba(255,255,255,0.04);
    }

    .info-icon {

        width: 38px;
        height: 38px;

        border-radius: 11px;

        display: flex;

        align-items: center;
        justify-content: center;

        background:
            rgba(99,102,241,0.10);

        color: #818cf8;

        margin-bottom: 13px;
    }

    .info-card-title {

        font-size: 13px;

        font-weight: 700;

        margin-bottom: 7px;
    }

    .info-card-text {

        color: #68748a;

        font-size: 10px;

        line-height: 1.7;

        margin: 0;
    }


    /* =====================================================
       DISCLAIMER
    ===================================================== */

    .disclaimer {

        margin-top: 22px;

        padding: 14px 17px;

        border-radius: 13px;

        background:
            rgba(245,158,11,0.045);

        border:
            1px solid
            rgba(245,158,11,0.10);

        color: #7f7a70;

        font-size: 9px;

        line-height: 1.7;
    }

    .disclaimer strong {

        color: #d6a93a;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .custom-footer {

        margin-top: 35px;

        padding: 25px 0;

        border-top:
            1px solid
            rgba(255,255,255,0.06);

        text-align: center;

        color: #606b80;

        font-size: 10px;
    }

    .footer-name {

        color: #a5b4fc;

        font-weight: 700;
    }

    .heart {

        display: inline-block;

        color: #f43f5e;

        animation:
            heartbeat 1.4s infinite;
    }

    @keyframes heartbeat {

        0%, 100% {
            transform: scale(1);
        }

        50% {
            transform: scale(1.18);
        }
    }


    /* =====================================================
       RESPONSIVE
    ===================================================== */

    @media (max-width: 767px) {

        .hero-title {

            font-size: 42px;

            letter-spacing: -2.5px;
        }

        .glass-card {

            padding: 17px;
        }

        .developer-card {

            align-items: flex-start;

            flex-direction: column;
        }

        .developer-tech {

            justify-content: flex-start;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# NAVBAR
# =========================================================

st.markdown(
    """
    <div class="navbar-custom">

        <div class="brand">

            <div class="brand-logo">
                🧬
            </div>

            <div class="brand-name">
                Kidney<span>Vision</span> AI
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✨ &nbsp; AI-Powered Medical Image Classification
        </div>

        <div class="hero-title">

            Kidney Disease

            <br>

            <span class="gradient-text">
                Classification
            </span>

        </div>

        <div class="hero-description">

            An end-to-end Deep Learning application powered by
            Convolutional Neural Networks to analyze kidney images
            and generate an AI-assisted classification result.

        </div>

        <div class="hero-stats">

            <div class="hero-stat">
                🧠 &nbsp; CNN Architecture
            </div>

            <div class="hero-stat">
                🖼️ &nbsp; Computer Vision
            </div>

            <div class="hero-stat">
                ⚡ &nbsp; Real-Time Prediction
            </div>

            <div class="hero-stat">
                🐍 &nbsp; TensorFlow
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_prediction_pipeline():

    filename = "inputImage.jpg"

    classifier = PredictionPipeline(filename)

    return classifier


try:

    classifier = load_prediction_pipeline()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.exception(e)

    st.stop()


# =========================================================
# MAIN APPLICATION CARD
# =========================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)


col1, col2 = st.columns(2, gap="large")


# =========================================================
# LEFT SIDE - UPLOAD
# =========================================================

with col1:

    st.markdown(
        """
        <div class="section-label">
            Step 01
        </div>

        <div class="section-title">
            Upload Kidney Image
        </div>

        <div class="section-subtitle">
            Select an image or drag and drop it here.
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


    if uploaded_file is None:

        st.markdown(
            """
            <div class="upload-container">

                <div>

                    <div class="upload-icon">
                        ☁️
                    </div>

                    <div class="upload-title">
                        Drop your image here
                    </div>

                    <div class="upload-description">
                        PNG, JPG or JPEG • Maximum 10 MB
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        try:

            image = Image.open(uploaded_file)

            st.markdown(
                '<div class="image-preview-card">',
                unsafe_allow_html=True
            )

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception:

            st.error("Unable to read this image.")

            uploaded_file = None


    st.write("")


    if uploaded_file is not None:

        analyze_button = st.button(
            "✨  Analyze Image",
            use_container_width=True
        )

    else:

        analyze_button = st.button(
            "✨  Analyze Image",
            use_container_width=True,
            disabled=True
        )


# =========================================================
# RIGHT SIDE - RESULT
# =========================================================

with col2:

    st.markdown(
        """
        <div class="section-label">
            Step 02
        </div>

        <div class="section-title">
            AI Prediction
        </div>

        <div class="section-subtitle">
            Your CNN model output will appear here.
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    if uploaded_file is None:

        st.markdown(
            """
            <div class="result-container">

                <div class="result-placeholder">

                    <div>

                        <div class="result-placeholder-icon">
                            📊
                        </div>

                        <div class="result-placeholder-text">

                            Your prediction will appear here
                            after the image has been analyzed.

                        </div>

                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    elif analyze_button:

        try:

            # Save image exactly as expected by
            # your existing PredictionPipeline

            image = Image.open(uploaded_file)

            image.save(
                "inputImage.jpg"
            )


            # -------------------------------------------------
            # MODEL INFERENCE
            # -------------------------------------------------

            with st.spinner(
                "Analyzing Kidney Image..."
            ):

                result = classifier.predict()


            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            st.markdown(
                """
                <div class="result-container">

                    <div class="prediction-result">

                        <div class="prediction-heading">
                            ✓ &nbsp; Model Prediction
                        </div>

                """,
                unsafe_allow_html=True
            )


            # Try to display a clean prediction

            if isinstance(result, (list, tuple)):

                prediction_text = result[0] if len(result) > 0 else result

            else:

                prediction_text = result


            st.markdown(
                f"""
                        <div class="prediction-value">
                            {prediction_text}
                        </div>
                """,
                unsafe_allow_html=True
            )


            # Full result for debugging/details

            st.markdown(
                """
                        <div class="prediction-heading">
                            Prediction Details
                        </div>
                """,
                unsafe_allow_html=True
            )


            try:

                formatted_result = json.dumps(
                    result,
                    indent=2,
                    default=str
                )

            except Exception:

                formatted_result = str(result)


            st.code(
                formatted_result,
                language="json"
            )


            st.markdown(
                """
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                "❌ Unable to process the image."
            )

            st.exception(e)


    else:

        st.markdown(
            """
            <div class="result-container">

                <div class="result-placeholder">

                    <div>

                        <div class="result-placeholder-icon">
                            📊
                        </div>

                        <div class="result-placeholder-text">

                            Your prediction will appear here
                            after the image has been analyzed.

                        </div>

                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# DEVELOPER CARD
# =========================================================

st.markdown(
    """
    <div class="developer-card">

        <div class="developer-info">

            <div class="developer-avatar">
                DS
            </div>

            <div>

                <div class="developer-label">
                    Project Developer
                </div>

                <div class="developer-name">
                    Divyadarshan Srivastava
                </div>

                <div class="developer-role">
                    AI/ML Engineer • Deep Learning
                </div>

            </div>

        </div>


        <div class="developer-tech">

            <span class="tech-tag">
                Python
            </span>

            <span class="tech-tag">
                TensorFlow
            </span>

            <span class="tech-tag">
                CNN
            </span>

            <span class="tech-tag">
                Streamlit
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION CARDS
# =========================================================

st.markdown(
    '<div class="info-section">',
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3, gap="medium")


with info1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🧠
            </div>

            <div class="info-card-title">
                Deep Learning Model
            </div>

            <p class="info-card-text">

                A Convolutional Neural Network processes
                visual features from the uploaded image
                for classification.

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                ⚡
            </div>

            <div class="info-card-title">
                Real-Time Analysis
            </div>

            <p class="info-card-text">

                Upload an image and run inference
                directly through the deployed
                Deep Learning application.

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🧩
            </div>

            <div class="info-card-title">
                End-to-End Project
            </div>

            <p class="info-card-text">

                Demonstrates image preprocessing,
                deep learning inference,
                application integration
                and deployment.

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">

        <strong>Important:</strong>

        This application is an educational Deep Learning
        project and is not intended to provide medical diagnosis,
        treatment recommendations, or professional medical advice.
        Model predictions should not be used as a substitute for
        evaluation by a qualified healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CLOSE MAIN CARD
# =========================================================

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="custom-footer">

        Made with

        <span class="heart">
            ♥
        </span>

        by

        <span class="footer-name">
            Divyadarshan Srivastava
        </span>

    </div>
    """,
    unsafe_allow_html=True
)

