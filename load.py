import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

model = keras.models.load_model(r"D:\bangkit\scantion-ml-model-main\model.h5")

with open(
    r"D:\bangkit\scantion-ml-model-main\Dataset\test\malignant\ISIC_0033779.jpg", "rb"
) as file:
    image_bytes = file.read()
    pillow_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
    pillow_img = pillow_img.resize((100, 100))  # Resize to 100x100

data = np.asarray(pillow_img)
data = data / 255.0
data = data[np.newaxis, ...]  # Add batch dimension

predictions = model.predict(data)
pred0 = predictions[0]
label0 = np.argmax(pred0)
print(label0)
