import cv2
import numpy as np
from keras.models import load_model

# Load Yahoo OpenNSFW model (you need model file)
model = load_model("nsfw_model.h5")

def detect_nsfw(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]

    nsfw_score = float(preds[0])
    sfw_score = float(preds[1])

    return nsfw_score, sfw_score