```python
import os
import streamlit as st
from PIL import Image

from cnnClassifier.pipeline.prediction import PredictionPipeline


# ---------------------------------------------------------
# Environment settings
# ---------------------------------------------------------
os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Kidney Disease Classification",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }

        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .result-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.markdown(
    '<div class="title">🩺 Kidney Disease Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a kidney CT scan image and let the CNN model classify it.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Image Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Kidney Scan Image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Kidney Scan",
        use_container_width=True
    )

    st.write("")

    if st.button("🔍 Predict", use_container_width=True):

        try:

            # Save uploaded image using the filename
            # expected by PredictionPipeline
            image.save("inputImage.jpg")

            with st.spinner("Analyzing image..."):

                result = classifier.predict()

            st.success("Prediction completed successfully!")

            st.write("### Prediction Result")

            st.json(result)

        except Exception as e:

            st.error("❌ Prediction failed.")

            st.exception(e)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    "<div style='text-align:center;'>"
    "Made with ❤️ by Divyadarshan Srivastava"
    "</div>",
    unsafe_allow_html=True
)
```
