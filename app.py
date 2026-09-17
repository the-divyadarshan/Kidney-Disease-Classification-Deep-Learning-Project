import os
import json
import streamlit as st
import streamlit.components.v1 as components
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
# GLOBAL STREAMLIT CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(14,165,233,0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(168,85,247,0.10),
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

    footer {
        visibility: hidden;
    }

    [data-testid="stFileUploader"] {
        margin-top: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px dashed rgba(129,140,248,0.45) !important;
        border-radius: 18px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #818cf8 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a5b4fc !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        min-height: 48px;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6,
            #06b6d4
        );
        box-shadow:
            0 10px 30px rgba(99,102,241,0.25);
    }

    .stButton > button:hover {
        color: white;
        border: none;
        transform: translateY(-1px);
        box-shadow:
            0 15px 35px rgba(99,102,241,0.35);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVBAR + HERO
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

            body {
                margin: 0;
                background: transparent;
                color: white;
                font-family: "Inter", sans-serif;
            }

            .navbar {
                height: 78px;
                display: flex;
                align-items: center;
                justify-content: space-between;

                border-bottom:
                    1px solid rgba(255,255,255,0.07);

                margin-bottom: 20px;
            }

            .brand {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .brand-logo {
                width: 45px;
                height: 45px;

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
                    rgba(99,102,241,0.35);
            }

            .brand-name {
                font-size: 18px;
                font-weight: 800;
            }

            .brand-name span {
                color: #a5b4fc;
            }

            .status {
                display: flex;
                align-items: center;
                gap: 8px;

                font-size: 11px;
                color: #94a3b8;
            }

            .status-dot {
                width: 8px;
                height: 8px;

                border-radius: 50%;

                background: #22c55e;

                box-shadow:
                    0 0 12px
                    rgba(34,197,94,0.8);
            }

            .hero {
                text-align: center;
                padding:
                    42px
                    15px
                    48px;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 9px;

                padding:
                    9px
                    16px;

                border-radius: 50px;

                border:
                    1px solid
                    rgba(129,140,248,0.28);

                background:
                    rgba(99,102,241,0.08);

                color: #a5b4fc;

                font-size: 11px;
                font-weight: 600;

                margin-bottom: 24px;
            }

            .hero h1 {
                margin: 0;

                font-size:
                    clamp(43px, 6vw, 72px);

                line-height: 1.03;

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
                max-width: 700px;

                margin:
                    24px auto 0;

                color: #8f9bb1;

                font-size: 14px;

                line-height: 1.8;
            }

            .hero-stats {
                display: flex;

                justify-content: center;

                flex-wrap: wrap;

                gap: 10px;

                margin-top: 27px;
            }

            .hero-stat {
                display: flex;
                align-items: center;

                gap: 8px;

                padding:
                    8px
                    13px;

                border-radius: 30px;

                background:
                    rgba(255,255,255,0.035);

                border:
                    1px solid
                    rgba(255,255,255,0.07);

                color: #9ca8bd;

                font-size: 10px;
            }

            .hero-stat i {
                color: #818cf8;
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


            <h1>

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


            <div class="hero-stats">

                <div class="hero-stat">
                    <i class="fa-solid fa-brain"></i>
                    CNN Architecture
                </div>

                <div class="hero-stat">
                    <i class="fa-solid fa-image"></i>
                    Computer Vision
                </div>

                <div class="hero-stat">
                    <i class="fa-solid fa-bolt"></i>
                    Real-Time Prediction
                </div>

                <div class="hero-stat">
                    <i class="fa-solid fa-code"></i>
                    Deep Learning
                </div>

            </div>

        </section>

    </body>

    </html>
    """,
    height=360,
    scrolling=False
)


# ============================================================
# APPLICATION CARD HEADER
# ============================================================

components.html(
    """
    <style>

        body {
            margin: 0;
            background: transparent;
            font-family: Inter, sans-serif;
            color: white;
        }

        .header {
            padding: 24px 28px 4px;
        }

        .title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 7px;
        }

        .subtitle {
            font-size: 11px;
            color: #7f8ba1;
        }

    </style>

    <div class="header">

        <div class="title">
            🔬 Kidney Image Analysis
        </div>

        <div class="subtitle">
            Upload a kidney image and let the trained CNN model analyze it.
        </div>

    </div>
    """,
    height=85,
    scrolling=False
)


# ============================================================
# MAIN CARD
# ============================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE
# ============================================================

with left:

    components.html(
        """
        <style>

            body {
                margin: 0;
                background: transparent;
                font-family: Inter, sans-serif;
                color: white;
            }

            .card {
                padding: 25px;

                border-radius: 20px;

                background:
                    rgba(15,23,42,0.58);

                border:
                    1px solid
                    rgba(255,255,255,0.07);
            }

            .step {
                font-size: 10px;
                color: #818cf8;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }

            .title {
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 6px;
            }

            .subtitle {
                font-size: 11px;
                color: #7f8ba1;
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
                    rgba(129,140,248,0.14);
            }

            .icon {
                font-size: 28px;
                margin-bottom: 10px;
            }

            .upload-title {
                font-size: 14px;
                font-weight: 700;
            }

            .upload-text {
                font-size: 11px;
                color: #7f8ba1;
                line-height: 1.7;
                margin-top: 5px;
            }

        </style>


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

                <div class="icon">
                    ☁️
                </div>

                <div class="upload-title">
                    Choose your kidney image
                </div>

                <div class="upload-text">
                    PNG, JPG or JPEG • Maximum 10 MB
                </div>

            </div>

        </div>
        """,
        height=225,
        scrolling=False
    )


    uploaded_file = st.file_uploader(
        "Upload Kidney Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )


# ============================================================
# RIGHT SIDE
# ============================================================

with right:

    components.html(
        """
        <style>

            body {
                margin: 0;
                background: transparent;
                font-family: Inter, sans-serif;
                color: white;
            }

            .card {
                min-height: 225px;

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
                    rgba(129,140,248,0.13);
            }

            .step {
                font-size: 10px;
                color: #818cf8;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }

            .title {
                font-size: 18px;
                font-weight: 700;
                margin-bottom: 12px;
            }

            .icon {
                font-size: 31px;
                margin-bottom: 12px;
            }

            .text {
                font-size: 11px;
                line-height: 1.7;
                color: #7f8ba1;
            }

        </style>


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
        """,
        height=225,
        scrolling=False
    )


    if uploaded_file:

        if st.button(
            "🔍  Analyze Kidney Image",
            key="predict"
        ):

            try:

                # ============================================
                # SAVE INPUT IMAGE
                # ============================================

                image = Image.open(uploaded_file).convert("RGB")

                image.save(
                    "inputImage.jpg",
                    format="JPEG"
                )


                # ============================================
                # LOAD PIPELINE
                # ============================================

                @st.cache_resource
                def get_classifier():

                    return PredictionPipeline(
                        filename="inputImage.jpg"
                    )


                classifier = get_classifier()


                # ============================================
                # PREDICTION
                # ============================================

                with st.spinner(
                    "Analyzing kidney image..."
                ):

                    result = classifier.predict()


                # ============================================
                # RESULT HEADER
                # ============================================

                components.html(
                    """
                    <style>

                        body {
                            margin: 0;
                            background: transparent;
                            font-family: Inter, sans-serif;
                        }

                        .result {
                            margin-top: 15px;
                            padding: 20px;

                            border-radius: 18px;

                            background:
                                rgba(34,197,94,0.06);

                            border:
                                1px solid
                                rgba(34,197,94,0.18);

                            color: white;
                        }

                        .heading {
                            font-size: 15px;
                            font-weight: 700;
                            margin-bottom: 6px;
                        }

                        .text {
                            color: #8290a6;
                            font-size: 11px;
                        }

                    </style>

                    <div class="result">

                        <div class="heading">
                            ✅ Prediction Completed
                        </div>

                        <div class="text">
                            The CNN model has successfully
                            analyzed the uploaded kidney image.
                        </div>

                    </div>
                    """,
                    height=105,
                    scrolling=False
                )


                # ============================================
                # SHOW RESULT
                # ============================================

                if isinstance(result, dict):

                    result_text = json.dumps(
                        result,
                        indent=2,
                        default=str
                    )

                else:

                    result_text = str(result)


                st.markdown(
                    f"""
                    <div style="
                        margin-top:10px;
                        padding:18px;
                        border-radius:15px;
                        background:rgba(3,7,18,0.55);
                        border:1px solid rgba(255,255,255,0.07);
                        color:#dbe4f5;
                        font-family:monospace;
                        font-size:12px;
                        white-space:pre-wrap;
                    ">
{result_text}
                    </div>
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
    <style>

        body {
            margin: 0;
            background: transparent;
            font-family: Inter, sans-serif;
            color: white;
        }

        .heading {
            text-align: center;
            margin-top: 25px;
            margin-bottom: 28px;
        }

        .title {
            font-size: 25px;
            font-weight: 800;
        }

        .subtitle {
            color: #7f8ba1;
            font-size: 11px;
            margin-top: 7px;
        }

    </style>

    <div class="heading">

        <div class="title">
            About The Application
        </div>

        <div class="subtitle">
            Built as an end-to-end Deep Learning project
        </div>

    </div>
    """,
    height=90,
    scrolling=False
)


# ============================================================
# INFO CARDS
# ============================================================

info1, info2, info3 = st.columns(3)


def info_card(icon, title, text):

    components.html(
        f"""
        <style>

            body {{
                margin: 0;
                background: transparent;
                font-family: Inter, sans-serif;
                color: white;
            }}

            .card {{
                padding: 24px;

                min-height: 180px;

                border-radius: 20px;

                background:
                    rgba(15,23,42,0.58);

                border:
                    1px solid
                    rgba(255,255,255,0.07);
            }}

            .icon {{
                font-size: 24px;
                margin-bottom: 15px;
            }}

            .title {{
                font-size: 14px;
                font-weight: 700;
                margin-bottom: 8px;
            }}

            .text {{
                font-size: 11px;
                color: #7f8ba1;
                line-height: 1.7;
            }}

        </style>

        <div class="card">

            <div class="icon">
                {icon}
            </div>

            <div class="title">
                {title}
            </div>

            <div class="text">
                {text}
            </div>

        </div>
        """,
        height=190,
        scrolling=False
    )


with info1:

    info_card(
        "🧠",
        "Deep Learning Model",
        "The application uses a Convolutional Neural Network trained to classify kidney images."
    )


with info2:

    info_card(
        "⚡",
        "Real-Time Analysis",
        "Upload an image and receive the model classification directly through the application."
    )


with info3:

    info_card(
        "🚀",
        "End-to-End Project",
        "Designed as an end-to-end Deep Learning project covering training, prediction and deployment."
    )


# ============================================================
# DEVELOPER
# ============================================================

components.html(
    """
    <style>

        body {
            margin: 0;
            background: transparent;
            font-family: Inter, sans-serif;
            color: white;
        }

        .developer {
            margin-top: 32px;

            padding: 24px;

            border-radius: 20px;

            display: flex;
            align-items: center;
            gap: 18px;

            background:
                linear-gradient(
                    135deg,
                    rgba(99,102,241,0.08),
                    rgba(6,182,212,0.04)
                );

            border:
                1px solid
                rgba(129,140,248,0.13);
        }

        .avatar {
            width: 62px;
            height: 62px;

            border-radius: 50%;

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 20px;
            font-weight: 800;

            background:
                linear-gradient(
                    135deg,
                    #6366f1,
                    #8b5cf6,
                    #06b6d4
                );
        }

        .name {
            font-size: 16px;
            font-weight: 700;
        }

        .role {
            color: #7f8ba1;
            font-size: 11px;
            margin-top: 4px;
        }

        .tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }

        .tag {
            padding: 5px 9px;

            border-radius: 7px;

            background:
                rgba(255,255,255,0.045);

            color: #aeb9cb;

            font-size: 9px;
        }

    </style>


    <div class="developer">

        <div class="avatar">
            DS
        </div>

        <div>

            <div class="name">
                Divyadarshan Srivastava
            </div>

            <div class="role">
                AI/ML Engineer • Deep Learning
            </div>

            <div class="tags">

                <div class="tag">Python</div>

                <div class="tag">TensorFlow</div>

                <div class="tag">CNN</div>

                <div class="tag">Streamlit</div>

            </div>

        </div>

    </div>
    """,
    height=125,
    scrolling=False
)


# ============================================================
# DISCLAIMER + FOOTER
# ============================================================

components.html(
    """
    <style>

        body {
            margin: 0;
            background: transparent;
            font-family: Inter, sans-serif;
            color: white;
        }

        .disclaimer {
            margin-top: 25px;

            padding: 17px 20px;

            border-radius: 14px;

            background:
                rgba(245,158,11,0.045);

            border:
                1px solid
                rgba(245,158,11,0.12);

            color: #818ca0;

            font-size: 10px;

            line-height: 1.7;
        }

        .disclaimer strong {
            color: #fbbf24;
        }

        .footer {
            text-align: center;

            margin-top: 35px;

            padding-top: 22px;

            border-top:
                1px solid
                rgba(255,255,255,0.06);

            color: #59657a;

            font-size: 10px;

            line-height: 1.8;
        }

        .heart {
            color: #f472b6;
        }

    </style>


    <div class="disclaimer">

        <strong>⚠️ Medical Disclaimer:</strong>

        This application is developed for educational and
        demonstration purposes only. The AI-generated
        classification should not be considered a medical
        diagnosis or a substitute for professional medical
        advice, examination or treatment.

    </div>


    <div class="footer">

        KidneyVision AI • Deep Learning Kidney Disease Classification

        <br>

        Made with <span class="heart">♥</span>
        by <strong>Divyadarshan Srivastava</strong>

    </div>
    """,
    height=150,
    scrolling=False
)