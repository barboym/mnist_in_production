from flask import Flask
from flask import render_template
from flask import request

import base64
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("static/best_model.hdf5")


def covert_image_from_txt_to_tensor(txt_image) -> tf.constant:
    """
    converting the image from base64 to bytes and then to tensor
    :param txt_image:
    :return: tensor
    """
    base64_img = base64.b64decode(bytes(txt_image, 'utf-8'))
    tensor_img = tf.image.decode_image(base64_img)
    return tensor_img


def resize_to_model_input_shape_and_normalize(tensor_img: tf.constant):
    tensor_img = tf.image.resize(tensor_img, model.input.shape[1:3])
    tensor_img = tf.reduce_max(tensor_img, axis=2, keepdims=True)
    tensor_img = tensor_img[np.newaxis, ...]
    tensor_img = tensor_img / (tf.reduce_max(tensor_img) + 10 ** -8)
    return tensor_img


def preprocess_image(image_txt) -> tf.constant:
    tensor_img = covert_image_from_txt_to_tensor(image_txt.split(",")[-1])
    tensor_img = resize_to_model_input_shape_and_normalize(tensor_img)
    return tensor_img


@app.route("/", methods=["GET", "POST"])
def home():
    image_txt = dict(request.form).get("image", 0)
    if image_txt == 0:
        return render_template("index.html", res="")

    input_image = preprocess_image(image_txt)
    input_image_to_send = tf.image.resize(input_image, (200, 200))[0].numpy().tobytes()
    result = model.predict(input_image).argmax(axis=1)[0]
    return render_template("index.html", result=result, input_image=input_image_to_send)


if __name__ == '__main__':
    app.debug = True
    # app.run()
    app.run(debug = True)
