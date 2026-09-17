import json
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tensorflow as tf

# ============================================================
# FIX FOR KERAS / TENSORFLOW MODEL LOADING
# ============================================================
#
# Your trained model contains an older CategoricalCrossentropy
# configuration. Streamlit's current TensorFlow/Keras version
# tries to deserialize that configuration and produces:
#
# CategoricalCrossentropy.__init__() got an unexpected
# keyword argument 'fn'
#
# We patch the load_model function used by PredictionPipeline
# so that the model is loaded with compile=False.
#
# This is correct for prediction/inference because we are not
# training the model inside Streamlit.
# ============================================================

import cnnClassifier.pipeline.prediction as prediction_module


def load_model_without_compile(*args, **kwargs):
    kwargs.pop("compile", None)

    return tf.keras.models.load_model(
        *args,
        compile=False,
        **kwargs
    )


# Replace the load_model used inside prediction.py
prediction_module.load_model = load_model_without_compile

PredictionPipeline = prediction_module.PredictionPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KidneyVision AI | Kidney Disease Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GLOBAL STREAMLIT CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
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
                rgba(14, 165, 233, 0.13),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(168, 85, 247, 0.10),
                transparent 35%
            ),
            #060914;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 0 !important;
        padding-bottom: 30px !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }

    [data-testid="stFileUploader"] {
        margin-top: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px dashed rgba(129, 140, 248, 0.45) !important;
        border-radius: 18px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #818cf8 !important;
        background: rgba(129, 140, 248, 0.04) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a5b4fc !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;

        border-radius: 13px !important;

        border: none !important;

        color: white !important;

        font-family: "Inter", sans-serif !important;

        font-size: 13px !important;

        font-weight: 700 !important;

        background:
            linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6,
                #06b6d4
            ) !important;

        box-shadow:
            0 10px 30px
            rgba(99, 102, 241, 0.25) !important;

        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 15px 35px
            rgba(99, 102, 241, 0.35) !important;
    }

    /* Remove excessive Streamlit spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD PREDICTION PIPELINE
# ============================================================

@st.cache_resource
def load_prediction_pipeline():

    filename = "inputImage.jpg"

    classifier = PredictionPipeline(filename)

    return classifier


try:

    classifier = load_prediction_pipeline()

except Exception as e:

    st.error("Unable to load the prediction pipeline.")

    st.code(str(e))

    st.stop()


# ============================================================
# NAVBAR + HERO SECTION
# ============================================================

components.html(
    """
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="UTF-8">

        <link
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
            rel="stylesheet"
        >

        <link
            rel="stylesheet"
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"
        >

        <style>

            * {
                box-sizing: border-box;
            }

            html,
            body {
                margin: 0;
                padding: 0;
                width: 100%;
                background: transparent;
                font-family: "Inter", sans-serif;
                color: white;
            }

            .navbar {
                width: 100%;
                min-height: 78px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom:
                    1px solid
                    rgba(255,255,255,0.07);
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 13px;
            }

            .brand-logo {
                width: 46px;
                height: 46px;
                border-radius: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 23px;

                background:
                    linear-gradient(
                        135deg,
                        #6366f1,
                        #8b5cf6
                    );

                box-shadow:
                    0 8px 30px
                    rgba(99,102,241,0.35);
            }

            .brand-name {
                font-size: 19px;
                font-weight: 800;
                letter-spacing: -0.5px;
            }

            .brand-name span {
                color: #a5b4fc;
            }

            .status {
                display: flex;
                align-items: center;
                gap: 9px;
                color: #91a0b8;
                font-size: 12px;
            }

            .status-dot {
                width: 9px;
                height: 9px;
                border-radius: 50%;
                background: #22c55e;

                box-shadow:
                    0 0 14px
                    rgba(34,197,94,0.85);
            }

            .hero {
                text-align: center;
                padding:
                    50px
                    10px
                    40px;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 9px;

                padding:
                    10px
                    17px;

                border-radius: 50px;

                border:
                    1px solid
                    rgba(129,140,248,0.32);

                background:
                    rgba(99,102,241,0.07);

                color: #a5b4fc;

                font-size: 12px;
                font-weight: 600;
                margin-bottom: 25px;
            }

            .hero-title {
                margin: 0;
                font-size:
                    clamp(42px, 6vw, 72px);
                line-height: 1.05;
                letter-spacing: -3px;
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
            }

            .hero-description {
                max-width: 760px;

                margin:
                    24px auto 0;

                color: #8f9bb1;

                font-size: 14px;

                line-height: 1.8;
            }

            .hero-tags {
                display: flex;
                justify-content: center;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 27px;
            }

            .hero-tag {
                display: inline-flex;
                align-items: center;
                gap: 7px;

                padding:
                    8px 13px;

                border-radius: 30px;

                background:
                    rgba(255,255,255,0.035);

                border:
                    1px solid
                    rgba(255,255,255,0.07);

                color: #9da9bd;

                font-size: 10px;
            }

            .hero-tag i {
                color: #818cf8;
            }

            @media(max-width:700px) {

                .brand-name {
                    font-size: 16px;
                }

                .status {
                    display: none;
                }

                .hero {
                    padding-top: 40px;
                }

                .hero-title {
                    font-size: 42px;
                    letter-spacing: -2px;
                }
            }

        </style>

    </head>

    <body>

        <div class="navbar">

            <div class="brand">

                <div class="brand-logo">
                    🧬
                </div>

                <div class="brand-name">
                    Kidney<span>Vision</span> AI
                </div>

            </div>

            <div class="status">

                <span class="status-dot"></span>

                Model Online

            </div>

        </div>


        <section class="hero">

            <div class="hero-badge">

                <i class="fa-solid fa-wand-magic-sparkles"></i>

                AI-Powered Medical Image Classification

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

            <div class="hero-tags">

                <div class="hero-tag">
                    <i class="fa-solid fa-brain"></i>
                    CNN Architecture
                </div>

                <div class="hero-tag">
                    <i class="fa-solid fa-image"></i>
                    Computer Vision
                </div>

                <div class="hero-tag">
                    <i class="fa-solid fa-bolt"></i>
                    Real-Time Prediction
                </div>

                <div class="hero-tag">
                    <i class="fa-solid fa-fire"></i>
                    TensorFlow
                </div>

            </div>

        </section>

    </body>

    </html>
    """,
    height=450,
    scrolling=False
)


# ============================================================
# ANALYSIS HEADER
# ============================================================

components.html(
    """
    <html>
    <head>

        <link
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
            rel="stylesheet"
        >

        <style>

            body {
                margin: 0;
                background: transparent;
                font-family: "Inter", sans-serif;
                color: white;
            }

            .header {
                padding: 8px 0 18px;
            }

            .title {
                font-size: 19px;
                font-weight: 700;
                margin-bottom: 7px;
            }

            .subtitle {
                color: #8290a6;
                font-size: 11px;
                line-height: 1.6;
            }

        </style>

    </head>

    <body>

        <div class="header">

            <div class="title">
                🔬 Kidney Image Analysis
            </div>

            <div class="subtitle">
                Upload a kidney image and let the trained CNN model analyze it.
            </div>

        </div>

    </body>
    </html>
    """,
    height=80,
    scrolling=False
)


# ============================================================
# MAIN TWO-COLUMN SECTION
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    components.html(
        """
        <html>

        <head>

            <link
                href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
                rel="stylesheet"
            >

            <style>

                body {
                    margin: 0;
                    background: transparent;
                    font-family: "Inter", sans-serif;
                    color: white;
                }

                .card {
                    padding: 25px;
                    border-radius: 20px;

                    background:
                        rgba(15,23,42,0.58);

                    border:
                        1px solid
                        rgba(255,255,255,0.08);
                }

                .step {
                    color: #818cf8;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    margin-bottom: 9px;
                    text-transform: uppercase;
                }

                .title {
                    font-size: 18px;
                    font-weight: 700;
                    margin-bottom: 7px;
                }

                .subtitle {
                    color: #7f8ba1;
                    font-size: 11px;
                    margin-bottom: 22px;
                }

                .upload-info {
                    padding: 20px;
                    border-radius: 17px;

                    background:
                        linear-gradient(
                            135deg,
                            rgba(99,102,241,0.09),
                            rgba(6,182,212,0.04)
                        );

                    border:
                        1px solid
                        rgba(129,140,248,0.15);
                }

                .upload-icon {
                    font-size: 29px;
                    margin-bottom: 9px;
                }

                .upload-title {
                    font-size: 14px;
                    font-weight: 700;
                    margin-bottom: 5px;
                }

                .upload-text {
                    color: #7f8ba1;
                    font-size: 11px;
                    line-height: 1.6;
                }

            </style>

        </head>

        <body>

            <div class="card">

                <div class="step">
                    Step 01
                </div>

                <div class="title">
                    Upload Kidney Image
                </div>

                <div class="subtitle">
                    Select an image or drag and drop it here.
                </div>

                <div class="upload-info">

                    <div class="upload-icon">
                        ☁️
                    </div>

                    <div class="upload-title">
                        Choose your kidney image
                    </div>

                    <div class="upload-text">
                        PNG, JPG or JPEG • Recommended medical image format
                    </div>

                </div>

            </div>

        </body>

        </html>
        """,
        height=230,
        scrolling=False
    )


    # ========================================================
    # STREAMLIT UPLOADER
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload Kidney Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    components.html(
        """
        <html>

        <head>

            <link
                href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
                rel="stylesheet"
            >

            <style>

                body {
                    margin: 0;
                    background: transparent;
                    font-family: "Inter", sans-serif;
                    color: white;
                }

                .card {
                    min-height: 230px;
                    padding: 25px;
                    border-radius: 20px;

                    background:
                        linear-gradient(
                            145deg,
                            rgba(99,102,241,0.08),
                            rgba(6,182,212,0.035)
                        );

                    border:
                        1px solid
                        rgba(129,140,248,0.14);
                }

                .step {
                    color: #818cf8;
                    font-size: 10px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                    margin-bottom: 15px;
                }

                .icon {
                    font-size: 32px;
                    margin-bottom: 13px;
                }

                .title {
                    font-size: 19px;
                    font-weight: 700;
                    margin-bottom: 9px;
                }

                .text {
                    color: #7f8ba1;
                    font-size: 11px;
                    line-height: 1.8;
                }

            </style>

        </head>

        <body>

            <div class="card">

                <div class="step">
                    Step 02
                </div>

                <div class="icon">
                    🧠
                </div>

                <div class="title">
                    AI Prediction
                </div>

                <div class="text">

                    Upload an image and click the analyze button
                    below to generate the classification result
                    using the trained CNN model.

                </div>

            </div>

        </body>

        </html>
        """,
        height=230,
        scrolling=False
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if uploaded_file is not None:

        if st.button(
            "🔍  Analyze Kidney Image",
            key="analyze_button"
        ):

            try:

                # ==================================================
                # SAVE IMAGE
                # ==================================================

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                image.save(
                    "inputImage.jpg",
                    format="JPEG"
                )


                # ==================================================
                # RUN MODEL
                # ==================================================

                with st.spinner(
                    "Analyzing kidney image..."
                ):

                    result = classifier.predict()


                # ==================================================
                # SUCCESS MESSAGE
                # ==================================================

                components.html(
                    """
                    <html>

                    <head>

                        <style>

                            body {
                                margin: 0;
                                background: transparent;
                                font-family: "Inter", sans-serif;
                                color: white;
                            }

                            .result {
                                margin-top: 16px;
                                padding: 20px;
                                border-radius: 18px;

                                background:
                                    rgba(34,197,94,0.055);

                                border:
                                    1px solid
                                    rgba(34,197,94,0.18);
                            }

                            .heading {
                                font-size: 15px;
                                font-weight: 700;
                                margin-bottom: 6px;
                            }

                            .text {
                                color: #8290a6;
                                font-size: 11px;
                                line-height: 1.6;
                            }

                        </style>

                    </head>

                    <body>

                        <div class="result">

                            <div class="heading">
                                ✅ Prediction Completed
                            </div>

                            <div class="text">
                                The CNN model has successfully
                                analyzed the uploaded kidney image.
                            </div>

                        </div>

                    </body>

                    </html>
                    """,
                    height=105,
                    scrolling=False
                )


                # ==================================================
                # FORMAT RESULT
                # ==================================================

                if isinstance(result, dict):

                    result_text = json.dumps(
                        result,
                        indent=2,
                        default=str
                    )

                else:

                    result_text = str(result)


                # ==================================================
                # RESULT DISPLAY
                # ==================================================

                st.markdown(
                    f"""
                    <div style="
                        margin-top:10px;
                        padding:18px;
                        border-radius:15px;

                        background:
                            rgba(3,7,18,0.60);

                        border:
                            1px solid
                            rgba(255,255,255,0.07);

                        color:#dbe4f5;

                        font-family:monospace;

                        font-size:12px;

                        line-height:1.7;

                        white-space:pre-wrap;

                        overflow-x:auto;
                    ">{result_text}</div>
                    """,
                    unsafe_allow_html=True
                )


            except Exception as e:

                st.error(
                    "Prediction failed: " + str(e)
                )


# ============================================================
# ABOUT SECTION
# ============================================================

st.write("")
st.write("")


components.html(
    """
    <html>

    <head>

        <style>

            body {
                margin: 0;
                background: transparent;
                font-family: "Inter", sans-serif;
                color: white;
            }

            .heading {
                text-align: center;
                padding: 25px 0 22px;
            }

            .title {
                font-size: 25px;
                font-weight: 800;
            }

            .subtitle {
                color: #7f8ba1;
                font-size: 11px;
                margin-top: 8px;
            }

        </style>

    </head>

    <body>

        <div class="heading">

            <div class="title">
                About The Application
            </div>

            <div class="subtitle">
                Built as an end-to-end Deep Learning project
            </div>

        </div>

    </body>

    </html>
    """,
    height=100,
    scrolling=False
)


# ============================================================
# INFORMATION CARD FUNCTION
# ============================================================

def create_info_card(
    icon,
    title,
    description
):

    components.html(
        f"""
        <html>

        <head>

            <style>

                body {{
                    margin: 0;
                    background: transparent;
                    font-family: "Inter", sans-serif;
                    color: white;
                }}

                .card {{
                    min-height: 180px;
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
                }}

                .icon {{
                    font-size: 25px;
                    margin-bottom: 15px;
                }}

                .title {{
                    font-size: 14px;
                    font-weight: 700;
                    margin-bottom: 9px;
                }}

                .description {{
                    color: #7f8ba1;
                    font-size: 11px;
                    line-height: 1.75;
                }}

            </style>

        </head>

        <body>

            <div class="card">

                <div class="icon">
                    {icon}
                </div>

                <div class="title">
                    {title}
                </div>

                <div class="description">
                    {description}
                </div>

            </div>

        </body>

        </html>
        """,
        height=190,
        scrolling=False
    )


# ============================================================
# INFORMATION CARDS
# ============================================================

info_col1, info_col2, info_col3 = st.columns(
    3,
    gap="medium"
)


with info_col1:

    create_info_card(
        "🧠",
        "Deep Learning Model",
        "The application uses a Convolutional Neural Network trained to classify kidney images."
    )


with info_col2:

    create_info_card(
        "⚡",
        "Real-Time Analysis",
        "Upload an image and receive the model classification directly through the interactive application."
    )


with info_col3:

    create_info_card(
        "🚀",
        "End-to-End Project",
        "Designed as an end-to-end Deep Learning project covering training, prediction and deployment."
    )


# ============================================================
# DEVELOPER SECTION
# ============================================================

components.html(
    """
    <html>

    <head>

        <style>

            body {
                margin: 0;
                background: transparent;
                font-family: "Inter", sans-serif;
                color: white;
            }

            .developer {

                margin-top: 32px;

                padding: 24px;

                border-radius: 20px;

                display: flex;

                align-items: center;

                gap: 20px;

                background:
                    linear-gradient(
                        135deg,
                        rgba(99,102,241,0.09),
                        rgba(6,182,212,0.045)
                    );

                border:
                    1px solid
                    rgba(129,140,248,0.15);

                box-shadow:
                    0 15px 40px
                    rgba(0,0,0,0.20);
            }

            .avatar {

                width: 82px;
                height: 82px;
                min-width: 82px;

                border-radius: 50%;

                overflow: hidden;

                border:
                    3px solid
                    rgba(129,140,248,0.60);

                box-shadow:
                    0 8px 25px
                    rgba(99,102,241,0.30);

                background:
                    linear-gradient(
                        135deg,
                        #6366f1,
                        #8b5cf6,
                        #06b6d4
                    );
            }

            .avatar img {

                width: 100%;
                height: 100%;

                object-fit: cover;

                display: block;
            }

            .developer-content {
                flex: 1;
            }

            .name {

                font-size: 18px;

                font-weight: 800;

                color: #ffffff;

                margin-bottom: 5px;
            }

            .role {

                color: #8f9bb1;

                font-size: 12px;

                margin-bottom: 12px;
            }

            .tags {

                display: flex;

                flex-wrap: wrap;

                gap: 7px;
            }

            .tag {

                padding:
                    6px 10px;

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

            @media(max-width:600px) {

                .developer {
                    flex-direction: column;
                    text-align: center;
                }

                .tags {
                    justify-content: center;
                }

            }

        </style>

    </head>

    <body>

        <div class="developer">

            <div class="avatar">

                <img
                    src="https://drive.google.com/thumbnail?id=19aVKAKDDHHn04mqvXHZ7Sb6cHvvOJWCQ&sz=w500"
                    alt="Divyadarshan Srivastava"
                >

            </div>

            <div class="developer-content">

                <div class="name">
                    Divyadarshan Srivastava
                </div>

                <div class="role">
                    AI/ML Engineer • Deep Learning
                </div>

                <div class="tags">

                    <div class="tag">
                        Python
                    </div>

                    <div class="tag">
                        TensorFlow
                    </div>

                    <div class="tag">
                        CNN
                    </div>

                    <div class="tag">
                        Deep Learning
                    </div>

                    <div class="tag">
                        Streamlit
                    </div>

                </div>

            </div>

        </div>

    </body>

    </html>
    """,
    height=155,
    scrolling=False
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:25px;
        padding:17px 20px;
        border-radius:14px;

        background:rgba(245,158,11,0.045);

        border:1px solid rgba(245,158,11,0.13);

        color:#818ca0;

        font-family:Inter,sans-serif;

        font-size:10px;

        line-height:1.75;
    ">

        <strong style="color:#fbbf24;">
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
# CUSTOM FOOTER
# ============================================================
#
# IMPORTANT:
# This is intentionally rendered using st.markdown instead
# of components.html().
#
# This prevents the footer from being trapped inside an iframe
# and guarantees the credit is visible.
# ============================================================

st.markdown(
    """
    <div style="
        width:100%;
        text-align:center;

        margin-top:18px;

        padding:
            18px 10px 25px;

        border-top:
            1px solid
            rgba(255,255,255,0.07);

        font-family:Inter,sans-serif;

        line-height:1.6;
    ">

        <div style="
            color:#aab5c8;
            font-size:11px;
            font-weight:600;
            margin-bottom:7px;
        ">

            KidneyVision AI
            <span style="color:#596579;">&nbsp; • &nbsp;</span>
            Deep Learning Kidney Disease Classification

        </div>

        <div style="
            color:#8f9bb1;
            font-size:11px;
            font-weight:500;
        ">

            Made with
            <span style="
                color:#f472b6;
                font-size:15px;
                padding:0 4px;
            ">❤️</span>
            by
            <span style="
                color:#a5b4fc;
                font-weight:700;
            ">Divyadarshan Srivastava</span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)