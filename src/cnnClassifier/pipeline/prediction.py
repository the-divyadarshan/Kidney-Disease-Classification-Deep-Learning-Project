import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        # Load trained model
        model_path = os.path.join("model", "model.h5")

        model = load_model(
            model_path,
            compile=False
        )

        # Load uploaded image
        imagename = self.filename

        test_image = image.load_img(
            imagename,
            target_size=(224, 224)
        )

        # Convert image to numpy array
        test_image = image.img_to_array(test_image)

        # Add batch dimension
        test_image = np.expand_dims(
            test_image,
            axis=0
        )

        # Make prediction
        result = np.argmax(
            model.predict(test_image),
            axis=1
        )

        print("Prediction result:", result)

        # Class prediction
        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]