import json
import streamlit as st
from PIL import Image

from cnnClassifier.pipeline.prediction import PredictionPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KidneyVision AI | Kidney Disease Classification",
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

    /* ========================================================
       GLOBAL
    ======================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html,
    body,
    [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {

        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(99,102,241,0.18),
                transparent 28%
            ),

            radial-gradient(
                circle at 95% 8%,
                rgba(6,182,212,0.12),
                transparent 30%
            ),

            radial-gradient(
                circle at 50% 95%,
                rgba(139,92,246,0.10),
                transparent 35%
            ),

            #060914;

        color: white;
    }


    .block-container {

        max-width: 1140px !important;

        padding-top: 25px !important;

        padding-bottom: 30px !important;
    }


    /* Remove Streamlit default elements */

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       NAVBAR
    ======================================================== */

    .navbar {

        display: flex;

        align-items: center;

        justify-content: space-between;

        padding:

            8px

            0

            22px;

        border-bottom:

            1px solid

            rgba(255,255,255,0.07);

        margin-bottom: 55px;
    }


    .brand {

        display: flex;

        align-items: center;

        gap: 13px;
    }


    .brand-logo {

        width: 55px;

        height: 55px;

        border-radius: 17px;

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 27px;

        background:

            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

        box-shadow:

            0 10px 35px

            rgba(99,102,241,0.35);
    }


    .brand-name {

        font-size: 22px;

        font-weight: 800;

        letter-spacing: -0.7px;

        color: #ffffff;
    }


    .brand-name span {

        color: #a5b4fc;
    }


    .model-status {

        display: flex;

        align-items: center;

        gap: 9px;

        color: #91a0b8;

        font-size: 13px;
    }


    .status-dot {

        width: 9px;

        height: 9px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:

            0 0 14px

            rgba(34,197,94,0.9);
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {

        text-align: center;

        padding:

            0

            10px

            50px;
    }


    .hero-badge {

        display: inline-flex;

        align-items: center;

        gap: 9px;

        padding:

            10px

            18px;

        border-radius: 50px;

        border:

            1px solid

            rgba(129,140,248,0.35);

        background:

            rgba(99,102,241,0.07);

        color: #a5b4fc;

        font-size: 12px;

        font-weight: 600;

        margin-bottom: 25px;
    }


    .hero-title {

        margin: 0;

        color: white;

        font-size:

            clamp(48px, 6vw, 76px);

        line-height: 1.04;

        letter-spacing: -4px;

        font-weight: 800;
    }


    .gradient-text {

        background:

            linear-gradient(
                90deg,
                #818cf8,
                #a78bfa,
                #22d3ee
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        background-clip: text;
    }


    .hero-description {

        max-width: 780px;

        margin:

            25px

            auto

            0;

        color: #8f9bb1;

        font-size: 14px;

        line-height: 1.9;
    }


    .hero-tags {

        display: flex;

        justify-content: center;

        align-items: center;

        flex-wrap: wrap;

        gap: 9px;

        margin-top: 25px;
    }


    .hero-tag {

        display: inline-flex;

        align-items: center;

        gap: 7px;

        padding:

            8px

            13px;

        border-radius: 30px;

        background:

            rgba(255,255,255,0.035);

        border:

            1px solid

            rgba(255,255,255,0.07);

        color: #9da9bd;

        font-size: 10px;
    }


    .hero-tag-icon {

        color: #818cf8;
    }


    /* ========================================================
       SECTION HEADER
    ======================================================== */

    .section-header {

        margin-top: 5px;

        margin-bottom: 22px;
    }


    .section-title {

        color: white;

        font-size: 21px;

        font-weight: 700;

        margin-bottom: 7px;
    }


    .section-subtitle {

        color: #8290a6;

        font-size: 12px;
    }


    /* ========================================================
       MAIN CARDS
    ======================================================== */

    .main-card {

        min-height: 390px;

        padding: 27px;

        border-radius: 21px;

        background:

            linear-gradient(
                145deg,
                rgba(15,23,42,0.72),
                rgba(15,23,42,0.43)
            );

        border:

            1px solid

            rgba(255,255,255,0.08);

        box-shadow:

            0 20px 55px

            rgba(0,0,0,0.18);
    }


    .step-label {

        color: #818cf8;

        font-size: 10px;

        font-weight: 700;

        letter-spacing: 1.2px;

        text-transform: uppercase;

        margin-bottom: 9px;
    }


    .card-title {

        color: white;

        font-size: 20px;

        font-weight: 700;

        margin-bottom: 7px;
    }


    .card-subtitle {

        color: #8290a6;

        font-size: 11px;

        margin-bottom: 20px;
    }


    .upload-box {

        min-height: 180px;

        display: flex;

        flex-direction: column;

        align-items: center;

        justify-content: center;

        text-align: center;

        padding: 25px;

        border-radius: 17px;

        border:

            1px dashed

            rgba(129,140,248,0.35);

        background:

            linear-gradient(
                145deg,
                rgba(99,102,241,0.07),
                rgba(6,182,212,0.025)
            );
    }


    .upload-icon {

        font-size: 38px;

        margin-bottom: 12px;
    }


    .upload-title {

        color: white;

        font-size: 14px;

        font-weight: 700;

        margin-bottom: 6px;
    }


    .upload-text {

        color: #78859b;

        font-size: 10px;

        line-height: 1.6;
    }


    /* ========================================================
       STREAMLIT FILE UPLOADER
    ======================================================== */

    [data-testid="stFileUploader"] {

        margin-top: 12px;
    }


    [data-testid="stFileUploaderDropzone"] {

        background:

            rgba(10,16,32,0.75) !important;

        border:

            1px dashed

            rgba(129,140,248,0.35) !important;

        border-radius:

            15px !important;
    }


    [data-testid="stFileUploaderDropzone"]:hover {

        border-color:

            #818cf8 !important;
    }


    [data-testid="stFileUploaderDropzoneInstructions"] {

        color:

            #a5b4fc !important;
    }


    /* ========================================================
       PREVIEW IMAGE
    ======================================================== */

    .preview-label {

        color: #818cf8;

        font-size: 10px;

        font-weight: 700;

        letter-spacing: 1px;

        margin-top: 18px;

        margin-bottom: 8px;

        text-transform: uppercase;
    }


    /* ========================================================
       ANALYZE BUTTON
    ======================================================== */

    .stButton {

        margin-top: 14px;
    }


    .stButton > button {

        width: 100% !important;

        min-height: 48px !important;

        border-radius: 13px !important;

        border: none !important;

        color: white !important;

        background:

            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6,
                #06b6d4
            ) !important;

        font-family: "Inter", sans-serif !important;

        font-size: 13px !important;

        font-weight: 700 !important;

        box-shadow:

            0 10px 30px

            rgba(99,102,241,0.25) !important;

        transition: all 0.25s ease !important;
    }


    .stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:

            0 15px 35px

            rgba(99,102,241,0.38) !important;
    }


    /* ========================================================
       PREDICTION RESULT
    ======================================================== */

    .prediction-result {

        margin-top: 20px;

        padding: 20px;

        border-radius: 17px;

        background:

            rgba(34,197,94,0.045);

        border:

            1px solid

            rgba(34,197,94,0.16);
    }


    .prediction-heading {

        color: #86efac;

        font-size: 14px;

        font-weight: 700;

        margin-bottom: 10px;
    }


    .prediction-value {

        padding: 14px;

        border-radius: 12px;

        background:

            rgba(0,0,0,0.25);

        border:

            1px solid

            rgba(255,255,255,0.06);

        color: #dbe4f5;

        font-family: monospace;

        font-size: 12px;

        line-height: 1.7;

        white-space: pre-wrap;

        overflow-x: auto;
    }


    /* ========================================================
       ABOUT SECTION
    ======================================================== */

    .about-section {

        text-align: center;

        margin-top: 65px;

        margin-bottom: 25px;
    }


    .about-title {

        color: white;

        font-size: 25px;

        font-weight: 800;

        margin-bottom: 8px;
    }


    .about-subtitle {

        color: #7f8ba1;

        font-size: 11px;
    }


    /* ========================================================
       INFO CARDS
    ======================================================== */

    .info-card {

        min-height: 190px;

        padding: 24px;

        border-radius: 20px;

        background:

            rgba(15,23,42,0.58);

        border:

            1px solid

            rgba(255,255,255,0.07);

        box-shadow:

            0 15px 40px

            rgba(0,0,0,0.15);
    }


    .info-icon {

        font-size: 27px;

        margin-bottom: 15px;
    }


    .info-title {

        color: white;

        font-size: 14px;

        font-weight: 700;

        margin-bottom: 9px;
    }


    .info-description {

        color: #7f8ba1;

        font-size: 11px;

        line-height: 1.75;
    }


    /* ========================================================
       DEVELOPER CARD
    ======================================================== */

    .developer-card {

        margin-top: 45px;

        padding: 25px;

        border-radius: 21px;

        display: flex;

        align-items: center;

        gap: 21px;

        background:

            linear-gradient(
                135deg,
                rgba(99,102,241,0.09),
                rgba(6,182,212,0.045)
            );

        border:

            1px solid

            rgba(129,140,248,0.16);

        box-shadow:

            0 15px 40px

            rgba(0,0,0,0.20);
    }


    .developer-photo {

        width: 88px;

        height: 88px;

        min-width: 88px;

        border-radius: 50%;

        object-fit: cover;

        display: block;

        border:

            3px solid

            rgba(129,140,248,0.65);

        box-shadow:

            0 8px 30px

            rgba(99,102,241,0.35);

        background:

            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6,
                #06b6d4
            );
    }


    .developer-name {

        color: white;

        font-size: 20px;

        font-weight: 800;

        margin-bottom: 5px;
    }


    .developer-role {

        color: #8f9bb1;

        font-size: 12px;

        margin-bottom: 13px;
    }


    .developer-tags {

        display: flex;

        flex-wrap: wrap;

        gap: 7px;
    }


    .developer-tag {

        padding:

            6px

            10px;

        border-radius: 8px;

        background:

            rgba(255,255,255,0.045);

        border:

            1px solid

            rgba(255,255,255,0.06);

        color: #b8c3d6;

        font-size: 9px;

        font-weight: 600;
    }


    /* ========================================================
       DISCLAIMER
    ======================================================== */

    .disclaimer {

        margin-top: 25px;

        padding:

            17px

            20px;

        border-radius: 15px;

        background:

            rgba(245,158,11,0.045);

        border:

            1px solid

            rgba(245,158,11,0.14);

        color: #8190a7;

        font-size: 10px;

        line-height: 1.75;
    }


    .disclaimer strong {

        color: #fbbf24;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .custom-footer {

        text-align: center;

        margin-top: 35px;

        padding:

            28px

            10px

            35px;

        border-top:

            1px solid

            rgba(255,255,255,0.07);

        color: #68738a;

        font-size: 11px;

        line-height: 2;
    }


    .footer-title {

        color: #aab5c8;

        font-weight: 600;

        margin-bottom: 4px;
    }


    .footer-heart {

        color: #f472b6;

        font-size: 16px;

        padding:

            0

            3px;
    }


    .footer-name {

        color: #a5b4fc;

        font-weight: 700;
    }


    /* ========================================================
       MOBILE RESPONSIVE
    ======================================================== */

    @media(max-width: 700px) {

        .block-container {

            padding-left: 18px !important;

            padding-right: 18px !important;
        }


        .navbar {

            margin-bottom: 40px;
        }


        .brand-name {

            font-size: 17px;
        }


        .brand-logo {

            width: 45px;

            height: 45px;

            font-size: 22px;
        }


        .model-status {

            display: none;
        }


        .hero-title {

            font-size: 43px;

            letter-spacing: -2px;
        }


        .hero-description {

            font-size: 12px;
        }


        .developer-card {

            flex-direction: column;

            text-align: center;
        }


        .developer-tags {

            justify-content: center;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVBAR
# ============================================================

st.markdown(
    """
    <div class="navbar">

        <div class="brand">

            <div class="brand-logo">
                🧬
            </div>

            <div class="brand-name">
                Kidney<span>Vision</span> AI
            </div>

        </div>

        <div class="model-status">

            <span class="status-dot"></span>

            Model Online

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

            ✨

            AI-Powered Medical Image Classification

        </div>


        <h1 class="hero-title">

            Kidney Disease

            <br>

            <span class="gradient-text">
                Classification
            </span>

        </h1>


        <div class="hero-description">

            An end-to-end Deep Learning application powered by
            Convolutional Neural Networks to analyze kidney images
            and generate an AI-assisted classification result.

        </div>


        <div class="hero-tags">

            <div class="hero-tag">

                <span class="hero-tag-icon">
                    🧠
                </span>

                CNN Architecture

            </div>


            <div class="hero-tag">

                <span class="hero-tag-icon">
                    🖼️
                </span>

                Computer Vision

            </div>


            <div class="hero-tag">

                <span class="hero-tag-icon">
                    ⚡
                </span>

                Real-Time Prediction

            </div>


            <div class="hero-tag">

                <span class="hero-tag-icon">
                    🔥
                </span>

                TensorFlow

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ANALYSIS SECTION HEADER
# ============================================================

st.markdown(
    """
    <div class="section-header">

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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_prediction_pipeline():

    filename = "inputImage.jpg"

    classifier = PredictionPipeline(filename)

    return classifier


try:

    classifier = load_prediction_pipeline()

except Exception as e:

    st.error(
        "Unable to load the prediction pipeline."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# MAIN COLUMNS
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT CARD - UPLOAD
# ============================================================

with left_column:

    st.markdown(
        """
        <div class="main-card">

            <div class="step-label">
                STEP 01
            </div>

            <div class="card-title">
                Upload Kidney Image
            </div>

            <div class="card-subtitle">
                Select an image or drag and drop it here.
            </div>


            <div class="upload-box">

                <div class="upload-icon">
                    ☁️
                </div>

                <div class="upload-title">
                    Choose your kidney image
                </div>

                <div class="upload-text">
                    PNG, JPG or JPEG<br>
                    Recommended medical image format
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FILE UPLOADER
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose your kidney image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        st.markdown(
            """
            <div class="preview-label">
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
# RIGHT CARD - PREDICTION
# ============================================================

with right_column:

    st.markdown(
        """
        <div class="main-card">

            <div class="step-label">
                STEP 02
            </div>

            <div style="
                font-size:36px;
                margin-bottom:14px;
            ">
                🧠
            </div>

            <div class="card-title">
                AI Prediction
            </div>

            <div class="card-subtitle">

                Upload an image and click the analyze button
                below to generate the classification result
                using the trained CNN model.

            </div>

            <div style="
                color:#6f7d94;
                font-size:11px;
                line-height:1.8;
                padding-top:8px;
            ">

                The trained Deep Learning model processes
                the uploaded image and returns its predicted
                classification.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if uploaded_file is not None:

        analyze_clicked = st.button(
            "🔍  Analyze Kidney Image",
            key="analyze_button"
        )


        if analyze_clicked:

            try:

                # ============================================
                # SAVE IMAGE
                # ============================================

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                image.save(
                    "inputImage.jpg",
                    format="JPEG"
                )


                # ============================================
                # PREDICTION
                # ============================================

                with st.spinner(
                    "Analyzing kidney image..."
                ):

                    result = classifier.predict()


                # ============================================
                # FORMAT RESULT
                # ============================================

                if isinstance(
                    result,
                    dict
                ):

                    result_text = json.dumps(
                        result,
                        indent=2,
                        default=str
                    )

                else:

                    result_text = str(
                        result
                    )


                # ============================================
                # DISPLAY RESULT
                # ============================================

                st.markdown(
                    """
                    <div class="prediction-result">

                        <div class="prediction-heading">
                            ✅ Prediction Completed
                        </div>

                        <div style="
                            color:#8190a7;
                            font-size:10px;
                            margin-bottom:10px;
                        ">
                            CNN model analysis result
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(
                    f"""
                    <div class="prediction-value">
{result_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            except Exception as e:

                st.error(
                    "Prediction failed: " + str(e)
                )


    else:

        st.markdown(
            """
            <div style="
                margin-top:15px;
                padding:15px;
                border-radius:13px;
                background:rgba(255,255,255,0.025);
                border:1px solid rgba(255,255,255,0.05);
                color:#65738b;
                font-size:10px;
                text-align:center;
            ">

                Upload an image to activate AI prediction.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown(
    """
    <div class="about-section">

        <div class="about-title">
            About The Application
        </div>

        <div class="about-subtitle">
            Built as an end-to-end Deep Learning project
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INFO CARDS
# ============================================================

info1, info2, info3 = st.columns(
    3,
    gap="medium"
)


with info1:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🧠
            </div>

            <div class="info-title">
                Deep Learning Model
            </div>

            <div class="info-description">

                The application uses a
                Convolutional Neural Network trained
                to classify kidney images.

            </div>

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

            <div class="info-title">
                Real-Time Analysis
            </div>

            <div class="info-description">

                Upload an image and receive the model
                classification directly through the
                interactive application.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with info3:

    st.markdown(
        """
        <div class="info-card">

            <div class="info-icon">
                🚀
            </div>

            <div class="info-title">
                End-to-End Project
            </div>

            <div class="info-description">

                Designed as an end-to-end Deep Learning
                project covering training, prediction
                and deployment.

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

        <img
            class="developer-photo"
            src="https://drive.google.com/thumbnail?id=19aVKAKDDHHn04mqvXHZ7Sb6cHvvOJWCQ&sz=w300"
            alt="Divyadarshan Srivastava"
        >


        <div>

            <div class="developer-name">
                Divyadarshan Srivastava
            </div>


            <div class="developer-role">
                AI/ML Engineer • Deep Learning
            </div>


            <div class="developer-tags">

                <span class="developer-tag">
                    Python
                </span>

                <span class="developer-tag">
                    TensorFlow
                </span>

                <span class="developer-tag">
                    CNN
                </span>

                <span class="developer-tag">
                    Deep Learning
                </span>

                <span class="developer-tag">
                    Streamlit
                </span>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <strong>
            ⚠️ Medical Disclaimer:
        </strong>

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
    <div class="custom-footer">

        <div class="footer-title">

            KidneyVision AI
            &nbsp; • &nbsp;
            Deep Learning Kidney Disease Classification

        </div>


        <div>

            Made with

            <span class="footer-heart">
                ♥
            </span>

            by

            <span class="footer-name">
                Divyadarshan Srivastava
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)