# Libraries
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import io
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from flask import Flask, request, jsonify

model = keras.models.load_model(r"D:\bangkit\scantion-ml-model-main\model.h5")


# transform image input
def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = data / 255.0
    data = data[100, 255, 100]  # [1,x,y,1]
    data = tf.image.resize(data, [100, 255, 100])
    return data


# Prediction Function
def predict(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert(
                "RGB"
            )  # Convert to RGB
            pillow_img = pillow_img.resize((640, 480))  # Resize to 100x100
            tensor = transform_image(pillow_img)
            prediction = predict(tensor)  # Fix the function name
            data = {"prediction": int(prediction)}
            return jsonify(data)

        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
