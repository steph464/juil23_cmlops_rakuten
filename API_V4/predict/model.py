import numpy as np
from PIL import Image
from keras.applications import ResNet50, imagenet_utils
from keras.applications.imagenet_utils import (decode_predictions, preprocess_input)
from keras.preprocessing.image import img_to_array


def load_modelresnet():
    """
    Loads and returns the pretrained model
    """
    model = ResNet50(weights="imagenet")
    print("Model loaded")
    return model


def prepare_image(image, target):

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


def predict(image, model):
    # We keep the 2 classes with the highest confidence score
    results = decode_predictions(model.predict(image), 2)[0]
    response = [
        {"class": result[1], "score": float(round(result[2], 3))} for result in results
    ]
    return response

