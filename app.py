import os
import json
import streamlit as st
from PIL import Image

from cnnClassifier.pipeline.prediction import PredictionPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KidneyVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(99,102,241,.18), transparent 30%),
        radial-gradient(circle at 100% 0%, rgba(6,182,212,.12), transparent 30%),
        #050914;
    color: white;
}

.block-container {
    max-width: 1140px !important;
    padding-top: 25px !important;
    padding-bottom: 30px !important;
}

header {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ============================================================
   NAVBAR
   ============================================================ */

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 20px 0;
    border-bottom: 1px solid rgba(255,255,255,.07);
    margin-bottom: 70px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-logo {
    width: 56px;
    height: 56px;
    border-radius: 17px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 27px;
    box-shadow: 0 12px 35px rgba(99,102,241,.35);
}

.brand-name {
    font-size: 22px;
    font-weight: 800;
    color: white;
}

.brand-name span {
    color: #a5b4fc;
}

.status {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #91a0b8;
    font-size: 13px;
}

.status-dot {
    width: 9px;
    height: 9px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 15px rgba(34,197,94,.9);
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    text-align: center;
    margin-bottom: 65px;
}

.hero-badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 50px;
    border: 1px solid rgba(129,140,248,.35);
    background: rgba(99,102,241,.06);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 28px;
}

.hero-title {
    font-size: clamp(48px,6vw,76px);
    line-height: 1.03;
    letter-spacing: -4px;
    font-weight: 800;
    margin: 0;
    color: white;
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
    max-width: 780px;
    margin: 28px auto 0 auto;
    color: #8f9bb1;
    font-size: 14px;
    line-height: 1.9;
}

.hero-tags {
    margin-top: 25px;
    display: flex;
    justify-content: center;
    gap: 9px;
    flex-wrap: wrap;
}

.hero-tag {
    padding: 8px 13px;
    border-radius: 30px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.07);
    color: #9da9bd;
    font-size: 10px;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: white;
    margin-bottom: 7px;
}

.section-subtitle {
    color: #8290a6;
    font-size: 12px;
    margin-bottom: 22px;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {
    background:
        linear-gradient(
            145deg,
            rgba(15,23,42,.76),
            rgba(10,15,30,.58)
        );

    border: 1px solid rgba(255,255,255,.08);

    border-radius: 21px;

    padding: 27px;

    min-height: 390px;

    box-shadow:
        0 20px 55px rgba(0,0,0,.20);
}

.step {
    color: #818cf8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
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
    line-height: 1.7;
}


/* ============================================================
   UPLOAD AREA
   ============================================================ */

.upload-area {
    margin-top: 25px;
    min-height: 170px;
    border: 1px dashed rgba(129,140,248,.38);
    border-radius: 17px;
    background:
        linear-gradient(
            145deg,
            rgba(99,102,241,.08),
            rgba(6,182,212,.025)
        );

    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    text-align: center;
}

.upload-icon {
    font-size: 38px;
    margin-bottom: 12px;
}

.upload-title {
    font-size: 14px;
    font-weight: 700;
    color: white;
}

.upload-text {
    margin-top: 7px;
    color: #78859b;
    font-size: 10px;
}


/* ============================================================
   STREAMLIT UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    margin-top: -65px;
    position: relative;
    opacity: 0;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 160px !important;
    cursor: pointer !important;
}


/* ============================================================
   IMAGE PREVIEW
   ============================================================ */

.preview-title {
    margin-top: 15px;
    margin-bottom: 10px;
    color: #818cf8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}


/* ============================================================
   BUTTON
   ============================================================ */

.stButton {
    margin-top: 25px;
}

.stButton button {
    width: 100%;
    height: 48px;
    border: none !important;
    border-radius: 13px !important;

    background:
        linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6,
            #06b6d4
        ) !important;

    color: white !important;
    font-size: 13px !important;
    font-weight: 700 !important;

    box-shadow:
        0 10px 30px rgba(99,102,241,.25);
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 15px 40px rgba(99,102,241,.38);
}


/* ============================================================
   RESULT
   ============================================================ */

.result-box {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;

    background: rgba(34,197,94,.05);

    border:
        1px solid rgba(34,197,94,.16);
}

.result-title {
    color: #86efac;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-content {
    color: #cbd5e1;
    background: rgba(0,0,0,.25);
    padding: 15px;
    border-radius: 11px;
    font-family: monospace;
    font-size: 11px;
    white-space: pre-wrap;
}


/* ============================================================
   ABOUT
   ============================================================ */

.about {
    text-align: center;
    margin-top: 70px;
    margin-bottom: 25px;
}

.about-title {
    color: white;
    font-size: 25px;
    font-weight: 800;
}

.about-subtitle {
    color: #7f8ba1;
    font-size: 11px;
    margin-top: 7px;
}


/* ============================================================
   INFO CARDS
   ============================================================ */

.info-card {
    min-height: 190px;
    padding: 24px;
    border-radius: 20px;

    background: rgba(15,23,42,.58);

    border:
        1px solid rgba(255,255,255,.07);

    box-shadow:
        0 15px 40px rgba(0,0,0,.15);
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

.info-text {
    color: #7f8ba1;
    font-size: 11px;
    line-height: 1.75;
}


/* ============================================================
   DEVELOPER
   ============================================================ */

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
            rgba(99,102,241,.09),
            rgba(6,182,212,.045)
        );

    border:
        1px solid rgba(129,140,248,.16);
}

.developer-photo {
    width: 88px;
    height: 88px;

    min-width: 88px;

    border-radius: 50%;

    object-fit: cover;

    border: 3px solid rgba(129,140,248,.65);

    box-shadow:
        0 8px 30px rgba(99,102,241,.35);
}

.developer-name {
    color: white;
    font-size: 20px;
    font-weight: 800;
}

.developer-role {
    color: #8f9bb1;
    font-size: 12px;
    margin-top: 5px;
}

.dev-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 13px;
}

.dev-tag {
    padding: 6px 10px;
    border-radius: 8px;
    background: rgba(255,255,255,.045);
    border: 1px solid rgba(255,255,255,.06);
    color: #b8c3d6;
    font-size: 9px;
    font-weight: 600;
}


/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer {
    margin-top: 25px;

    padding: 18px 20px;

    border-radius: 15px;

    background: rgba(245,158,11,.045);

    border:
        1px solid rgba(245,158,11,.14);

    color: #8190a7;

    font-size: 10px;

    line-height: 1.8;
}

.disclaimer strong {
    color: #fbbf24;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    margin-top: 35px;

    padding:
        30px
        10px
        45px;

    border-top:
        1px solid rgba(255,255,255,.07);

    color: #68738a;

    font-size: 11px;

    line-height: 2;
}

.footer-main {
    color: #aab5c8;
    font-weight: 600;
}

.footer-love {
    margin-top: 7px;
}

.heart {
    color: #f472b6;
    font-size: 17px;
    padding: 0 3px;
}

.footer-name {
    color: #a5b4fc;
    font-weight: 700;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media(max-width: 700px) {

    .block-container {
        padding-left: 18px !important;
        padding-right: 18px !important;
    }

    .navbar {
        margin-bottom: 45px;
    }

    .model-status {
        display: none;
    }

    .brand-name {
        font-size: 18px;
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

    .dev-tags {
        justify-content: center;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# NAVBAR
# ============================================================

st.markdown("""
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

        <div class="status-dot"></div>

        Model Online

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
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

    <div class="hero-tags">

        <div class="hero-tag">
            🧠 CNN Architecture
        </div>

        <div class="hero-tag">
            🖼️ Computer Vision
        </div>

        <div class="hero-tag">
            ⚡ Real-Time Prediction
        </div>

        <div class="hero-tag">
            🔥 TensorFlow
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION HEADER
# ============================================================

st.markdown("""
<div class="section-title">
    🔬 Kidney Image Analysis
</div>

<div class="section-subtitle">
    Upload a kidney image and let the trained CNN model analyze it.
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_classifier():

    filename = "inputImage.jpg"

    return PredictionPipeline(filename)


try:

    classifier = get_classifier()

except Exception as e:

    st.error("Model loading failed.")

    st.code(str(e))

    st.stop()


# ============================================================
# TWO COLUMNS
# ============================================================

left, right = st.columns(
    2,
    gap="large"
)


# ============================================================
# LEFT
# ============================================================

with left:

    st.markdown("""
    <div class="card">

        <div class="step">
            STEP 01
        </div>

        <div class="card-title">
            Upload Kidney Image
        </div>

        <div class="card-subtitle">
            Select an image or drag and drop it here.
        </div>

        <div class="upload-area">

            <div class="upload-icon">
                ☁️
            </div>

            <div class="upload-title">
                Choose your kidney image
            </div>

            <div class="upload-text">
                PNG, JPG or JPEG
                <br>
                Recommended image format
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


    if uploaded_file:

        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown(
            '<div class="preview-title">IMAGE PREVIEW</div>',
            unsafe_allow_html=True
        )

        st.image(
            uploaded_image,
            use_container_width=True
        )


# ============================================================
# RIGHT
# ============================================================

with right:

    st.markdown("""
    <div class="card">

        <div class="step">
            STEP 02
        </div>

        <div style="
            font-size:38px;
            margin-bottom:12px;
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
            margin-top:20px;
            color:#6f7d94;
            font-size:11px;
            line-height:1.8;
        ">

            The trained Deep Learning model processes
            the uploaded kidney image and returns its
            predicted classification.

        </div>

    </div>
    """, unsafe_allow_html=True)


    if uploaded_file:

        analyze = st.button(
            "🔍  Analyze Kidney Image"
        )

        if analyze:

            try:

                uploaded_image.save(
                    "inputImage.jpg",
                    format="JPEG"
                )

                with st.spinner(
                    "Analyzing kidney image..."
                ):

                    result = classifier.predict()


                if isinstance(result, dict):

                    result_text = json.dumps(
                        result,
                        indent=2,
                        default=str
                    )

                else:

                    result_text = str(result)


                st.markdown(f"""
                <div class="result-box">

                    <div class="result-title">
                        ✅ Prediction Completed
                    </div>

                    <div class="result-content">
{result_text}
                    </div>

                </div>
                """, unsafe_allow_html=True)


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

    else:

        st.markdown("""
        <div style="
            margin-top:25px;
            padding:17px;
            border-radius:13px;
            background:rgba(255,255,255,.025);
            border:1px solid rgba(255,255,255,.05);
            color:#65738b;
            font-size:10px;
            text-align:center;
        ">
            Upload an image to activate AI prediction.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# ABOUT
# ============================================================

st.markdown("""
<div class="about">

    <div class="about-title">
        About The Application
    </div>

    <div class="about-subtitle">
        Built as an end-to-end Deep Learning project
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INFO CARDS
# ============================================================

c1, c2, c3 = st.columns(3, gap="medium")


with c1:

    st.markdown("""
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
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">
            ⚡
        </div>

        <div class="info-title">
            Real-Time Analysis
        </div>

        <div class="info-text">
            Upload an image and receive the model
            classification directly through the application.
        </div>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">
            🚀
        </div>

        <div class="info-title">
            End-to-End Project
        </div>

        <div class="info-text">
            Designed as an end-to-end Deep Learning project
            covering training, prediction and deployment.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DEVELOPER
# ============================================================

photo_path = "assets/divyadarshan.jpg"

if os.path.exists(photo_path):

    with open(photo_path, "rb") as image_file:

        import base64

        encoded_image = base64.b64encode(
            image_file.read()
        ).decode()

    photo_src = (
        "data:image/jpeg;base64,"
        + encoded_image
    )

else:

    photo_src = (
        "https://drive.google.com/thumbnail"
        "?id=19aVKAKDDHHn04mqvXHZ7Sb6cHvvOJWCQ"
        "&sz=w300"
    )


st.markdown(f"""
<div class="developer-card">

    <img
        class="developer-photo"
        src="{photo_src}"
        alt="Divyadarshan Srivastava"
    >

    <div>

        <div class="developer-name">
            Divyadarshan Srivastava
        </div>

        <div class="developer-role">
            AI/ML Engineer • Deep Learning
        </div>

        <div class="dev-tags">

            <span class="dev-tag">
                Python
            </span>

            <span class="dev-tag">
                TensorFlow
            </span>

            <span class="dev-tag">
                CNN
            </span>

            <span class="dev-tag">
                Deep Learning
            </span>

            <span class="dev-tag">
                Streamlit
            </span>

        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="disclaimer">

    <strong>⚠️ Medical Disclaimer:</strong>

    This application is developed for educational and
    demonstration purposes only. The AI-generated
    classification should not be considered a medical
    diagnosis or a substitute for professional medical
    advice, examination or treatment.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    <div class="footer-main">

        KidneyVision AI
        &nbsp; • &nbsp;
        Deep Learning Kidney Disease Classification

    </div>

    <div class="footer-love">

        Made with

        <span class="heart">♥</span>

        by

        <span class="footer-name">
            Divyadarshan Srivastava
        </span>

    </div>

</div>
""", unsafe_allow_html=True)