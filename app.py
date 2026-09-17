import os
import streamlit as st
from PIL import Image

from cnnClassifier.pipeline.prediction import PredictionPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KidneyVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       IMPORT FONT
    -------------------------------------------------------- */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');


    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.18), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.15), transparent 30%),
            radial-gradient(circle at 50% 90%, rgba(6, 182, 212, 0.10), transparent 35%),
            linear-gradient(135deg, #070b18 0%, #0b1024 45%, #080c19 100%);
        color: #ffffff;
    }


    /* --------------------------------------------------------
       REMOVE STREAMLIT DEFAULT SPACING
    -------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }


    /* --------------------------------------------------------
       HIDE STREAMLIT MENU / FOOTER
    -------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* --------------------------------------------------------
       BRAND / NAVBAR
    -------------------------------------------------------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 55px;
        padding: 4px 0;
    }

    .brand-logo {
        width: 48px;
        height: 48px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6,
            #06b6d4
        );
        box-shadow:
            0 10px 30px rgba(99, 102, 241, 0.30);
    }

    .brand-name {
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }

    .brand-name span {
        background: linear-gradient(
            90deg,
            #8b5cf6,
            #06b6d4
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero {
        text-align: center;
        margin-bottom: 50px;
    }

    .hero-badge {
        display: inline-block;
        padding: 9px 17px;
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 50px;
        background: rgba(99, 102, 241, 0.10);
        color: #c4b5fd;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 22px;
        box-shadow: 0 5px 20px rgba(99, 102, 241, 0.08);
    }

    .hero-title {
        font-size: clamp(42px, 6vw, 72px);
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -3px;
        margin-bottom: 25px;
        color: #ffffff;
    }

    .gradient-text {
        background: linear-gradient(
            90deg,
            #818cf8,
            #a78bfa,
            #22d3ee
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 720px;
        margin: auto;
        color: #a7b0c5;
        font-size: 16px;
        line-height: 1.8;
    }

    .hero-tags {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 28px;
    }

    .hero-tag {
        padding: 7px 13px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 500;
    }


    /* --------------------------------------------------------
       MAIN GLASS CARD
    -------------------------------------------------------- */

    .main-card {
        position: relative;
        padding: 32px;
        border-radius: 28px;
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(255, 255, 255, 0.09);
        box-shadow:
            0 25px 80px rgba(0, 0, 0, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        margin-bottom: 35px;
    }


    /* --------------------------------------------------------
       SECTION HEADINGS
    -------------------------------------------------------- */

    .section-title {
        font-size: 19px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #8994aa;
        font-size: 13px;
        margin-bottom: 22px;
    }


    /* --------------------------------------------------------
       UPLOAD AREA
    -------------------------------------------------------- */

    .upload-info {
        padding: 22px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.07);
        border: 1px solid rgba(99, 102, 241, 0.18);
        margin-bottom: 18px;
    }

    .upload-icon {
        font-size: 35px;
        margin-bottom: 10px;
    }

    .upload-heading {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .upload-text {
        color: #8994aa;
        font-size: 13px;
        line-height: 1.6;
    }


    /* --------------------------------------------------------
       STREAMLIT FILE UPLOADER
    -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        width: 100%;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px dashed rgba(139, 92, 246, 0.45) !important;
        border-radius: 18px !important;
        padding: 20px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #8b5cf6 !important;
        background: rgba(139, 92, 246, 0.06) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a7b0c5 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #a7b0c5 !important;
    }


    /* --------------------------------------------------------
       IMAGE PREVIEW
    -------------------------------------------------------- */

    .preview-title {
        font-size: 13px;
        font-weight: 600;
        color: #9ca3af;
        margin-top: 20px;
        margin-bottom: 10px;
    }


    /* --------------------------------------------------------
       PREDICTION RESULT
    -------------------------------------------------------- */

    .result-card {
        padding: 25px;
        min-height: 300px;
        border-radius: 22px;
        background:
            linear-gradient(
                145deg,
                rgba(99, 102, 241, 0.10),
                rgba(6, 182, 212, 0.05)
            );
        border: 1px solid rgba(139, 92, 246, 0.18);
    }

    .result-icon {
        font-size: 42px;
        margin-bottom: 12px;
    }

    .result-heading {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .result-description {
        font-size: 13px;
        line-height: 1.7;
        color: #8f9ab0;
    }


    /* --------------------------------------------------------
       BUTTON
    -------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: 14px !important;
        padding: 13px 20px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6,
            #06b6d4
        ) !important;
        box-shadow:
            0 12px 30px rgba(99, 102, 241, 0.22);
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 16px 35px rgba(99, 102, 241, 0.32);
    }


    /* --------------------------------------------------------
       INFO CARDS
    -------------------------------------------------------- */

    .info-card {
        height: 100%;
        padding: 25px;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.20);
        transition: transform 0.25s ease;
    }

    .info-card:hover {
        transform: translateY(-4px);
    }

    .info-icon {
        width: 45px;
        height: 45px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        background: rgba(99, 102, 241, 0.12);
        margin-bottom: 17px;
    }

    .info-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .info-text {
        font-size: 13px;
        line-height: 1.7;
        color: #8994aa;
    }


    /* --------------------------------------------------------
       DEVELOPER CARD
    -------------------------------------------------------- */

    .developer-card {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 27px;
        border-radius: 23px;
        background:
            linear-gradient(
                135deg,
                rgba(99, 102, 241, 0.09),
                rgba(6, 182, 212, 0.05)
            );
        border: 1px solid rgba(139, 92, 246, 0.14);
        margin-top: 35px;
    }

    .developer-avatar {
        width: 70px;
        height: 70px;
        flex-shrink: 0;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6,
            #06b6d4
        );
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }

    .developer-name {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .developer-role {
        font-size: 13px;
        color: #929db3;
        margin-bottom: 10px;
    }

    .tech-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
    }

    .tech-tag {
        padding: 5px 9px;
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.05);
        color: #b8c1d4;
        font-size: 10px;
        font-weight: 600;
    }


    /* --------------------------------------------------------
       DISCLAIMER
    -------------------------------------------------------- */

    .disclaimer {
        margin-top: 35px;
        padding: 18px 20px;
        border-radius: 16px;
        background: rgba(245, 158, 11, 0.055);
        border: 1px solid rgba(245, 158, 11, 0.15);
        color: #9fa7b8;
        font-size: 11px;
        line-height: 1.7;
    }

    .disclaimer strong {
        color: #fbbf24;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        margin-top: 45px;
        padding-top: 25px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        color: #68738a;
        font-size: 12px;
    }

    .footer-heart {
        color: #f472b6;
    }


    /* --------------------------------------------------------
       JSON / RESULT OUTPUT
    -------------------------------------------------------- */

    [data-testid="stJson"] {
        background: rgba(0, 0, 0, 0.22) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }


    /* --------------------------------------------------------
       MOBILE
    -------------------------------------------------------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .brand {
            margin-bottom: 35px;
        }

        .hero-title {
            font-size: 42px;
            letter-spacing: -2px;
        }

        .hero-description {
            font-size: 14px;
        }

        .main-card {
            padding: 20px;
            border-radius: 20px;
        }

        .developer-card {
            flex-direction: column;
            text-align: center;
        }

        .tech-tags {
            justify-content: center;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BRAND
# ============================================================

st.markdown(
    """
    <div class="brand">
        <div class="brand-logo">🧬</div>

        <div class="brand-name">
            Kidney<span>Vision</span> AI
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✨ &nbsp; AI-Powered Medical Image Classification
        </div>

        <div class="hero-title">
            Kidney Disease<br>
            <span class="gradient-text">Classification</span>
        </div>

        <div class="hero-description">
            An end-to-end Deep Learning application powered by
            Convolutional Neural Networks to analyze kidney images
            and generate an AI-assisted classification result.
        </div>

        <div class="hero-tags">

            <div class="hero-tag">
                🧠 CNN Architecture
            </div>

            <div class="hero-tag">
                👁️ Computer Vision
            </div>

            <div class="hero-tag">
                ⚡ Real-Time Prediction
            </div>

            <div class="hero-tag">
                🔥 TensorFlow
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL / PREDICTION PIPELINE
# ============================================================

@st.cache_resource
def load_prediction_pipeline():

    filename = "inputImage.jpg"

    classifier = PredictionPipeline(filename)

    return classifier


# ============================================================
# LOAD CLASSIFIER
# ============================================================

try:

    classifier = load_prediction_pipeline()

except Exception as e:

    st.error("Unable to load the prediction pipeline.")

    st.code(str(e))

    st.stop()


# ============================================================
# MAIN PREDICTION SECTION
# ============================================================

st.markdown(
    """
    <div class="main-card">

        <div class="section-title">
            🔬 Kidney Image Analysis
        </div>

        <div class="section-subtitle">
            Upload a kidney image and let the trained CNN model analyze it.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE - IMAGE UPLOAD
# ============================================================

with left_column:

    st.markdown(
        """
        <div class="upload-info">

            <div class="upload-icon">
                📤
            </div>

            <div class="upload-heading">
                Upload Kidney Image
            </div>

            <div class="upload-text">
                Choose a kidney image in JPG, JPEG or PNG format
                for AI-powered classification.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.markdown(
            """
            <div class="preview-title">
                IMAGE PREVIEW
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


# ============================================================
# RIGHT SIDE - PREDICTION
# ============================================================

with right_column:

    st.markdown(
        """
        <div class="result-card">

            <div class="result-icon">
                🧠
            </div>

            <div class="result-heading">
                AI Prediction
            </div>

            <div class="result-description">
                Your prediction result will appear here after
                uploading a kidney image.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if uploaded_file is not None:

        st.write("")


        predict_button = st.button(
            "🔍 Analyze Kidney Image"
        )


        if predict_button:

            try:

                # ------------------------------------------------
                # SAVE IMAGE
                # ------------------------------------------------

                image = Image.open(uploaded_file)

                image = image.convert("RGB")

                image.save(
                    "inputImage.jpg",
                    format="JPEG"
                )


                # ------------------------------------------------
                # RUN PREDICTION
                # ------------------------------------------------

                with st.spinner(
                    "Analyzing kidney image..."
                ):

                    result = classifier.predict()


                # ------------------------------------------------
                # DISPLAY RESULT
                # ------------------------------------------------

                st.success(
                    "Prediction completed successfully!"
                )


                st.markdown(
                    """
                    <div class="result-card">

                        <div class="result-icon">
                            ✅
                        </div>

                        <div class="result-heading">
                            Classification Result
                        </div>

                        <div class="result-description">
                            The trained CNN model has completed
                            the analysis of the uploaded image.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.write("")

                # Display whatever your existing
                # PredictionPipeline returns.
                if isinstance(result, dict):

                    st.json(result)

                elif isinstance(result, list):

                    st.write(result)

                else:

                    st.write(result)


            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# ============================================================
# INFORMATION CARDS
# ============================================================

st.write("")
st.write("")


st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:45px;
        margin-bottom:25px;
    ">

        <div style="
            font-size:24px;
            font-weight:800;
            color:#ffffff;
        ">
            About The Application
        </div>

        <div style="
            color:#8994aa;
            font-size:13px;
            margin-top:7px;
        ">
            Built as an end-to-end Deep Learning project
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


info_col1, info_col2, info_col3 = st.columns(
    3,
    gap="medium"
)


# ============================================================
# CARD 1
# ============================================================

with info_col1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🧠
            </div>

            <div class="info-title">
                Deep Learning Model
            </div>

            <div class="info-text">
                The application uses a Convolutional Neural
                Network trained to classify kidney images.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CARD 2
# ============================================================

with info_col2:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                ⚡
            </div>

            <div class="info-title">
                Real-Time Analysis
            </div>

            <div class="info-text">
                Upload an image and receive the model's
                classification result directly through the
                interactive web application.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CARD 3
# ============================================================

with info_col3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🚀
            </div>

            <div class="info-title">
                End-to-End Project
            </div>

            <div class="info-text">
                Designed as an end-to-end Deep Learning
                application with model training, prediction
                and deployment workflow.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DEVELOPER SECTION
# ============================================================

st.markdown(
    """
    <div class="developer-card">

        <div class="developer-avatar">
            DS
        </div>

        <div>

            <div class="developer-name">
                Divyadarshan Srivastava
            </div>

            <div class="developer-role">
                AI/ML Engineer • Deep Learning
            </div>

            <div class="tech-tags">

                <div class="tech-tag">
                    Python
                </div>

                <div class="tech-tag">
                    TensorFlow
                </div>

                <div class="tech-tag">
                    CNN
                </div>

                <div class="tech-tag">
                    Streamlit
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <strong>⚠️ Medical Disclaimer:</strong>
        This application is developed for educational and
        demonstration purposes only. The AI-generated
        classification should not be considered a medical
        diagnosis or a substitute for professional medical
        advice, examination or treatment.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        KidneyVision AI &nbsp; • &nbsp;
        Deep Learning Kidney Disease Classification

        <br><br>

        Made with
        <span class="footer-heart">♥</span>
        by Divyadarshan Srivastava

    </div>
    """,
    unsafe_allow_html=True
)