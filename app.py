from flask import Flask
from flask import render_template
from flask import request

import base64
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("static/best_model.hdf5")
CANVAS_SHAPE = (200, 200)


def covert_image_from_txt_to_tensor(txt_image: str) -> tf.constant:
    """
    converting the image from base64 to bytes and then to tensor
    :param txt_image:
    :return: tensor
    """
    bytes_image = bytes(txt_image, 'utf-8')
    base64_img = base64.b64decode(bytes_image)
    tensor_img = tf.io.decode_png(base64_img)
    return tensor_img


def convert_image_from_tensor_to_txt(tensor_image: tf.constant) -> str:
    """
    converting the image from tensor to string of base64 bytes
    (not reversing covert_image_from_txt_to_tensor!!)
    :param: tensor
    :return: txt_image
    """
    tensor_image = tf.cast(tensor_image * 255, np.uint8)
    image_bytes = tf.io.encode_png(tensor_image[0]).numpy()
    image_base64 = base64.b64encode(image_bytes)
    image_base64_str = image_base64.decode()  # 'utf-8')
    return image_base64_str


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


def convert_from_grey_scale_to_rgba(tensor):
    """
    Convert tesor from shape (1,w,h,1) to (1,w,h,4).
    The first three channel of the output are identical to the grayscale input.
    The last channel is filled with ones.
    :param grayscale tensor:
    :return: RGBA tensor
    """
    rgb = (1 - tensor) * np.ones(shape=(1, 1, 1, 3))
    transparency = np.ones(shape=tensor.shape)
    return tf.concat([rgb, transparency], axis=3)


@app.route("/", methods=["GET", "POST"])
def home():
    image_txt = dict(request.form).get("image", "NO_IMAGE")
    input_image_txt = convert_image_from_tensor_to_txt(
        tf.constant(np.ones(shape=(1, *CANVAS_SHAPE, 1))))  # white image
    if image_txt == "NO_IMAGE":
        return render_template("index.html", res="", input_image=input_image_txt)

    input_image = preprocess_image(image_txt)
    result = model.predict(input_image).argmax(axis=1)[0]
    # converting to RGBA shape
    input_image_rgba = convert_from_grey_scale_to_rgba(input_image)
    input_image_txt = convert_image_from_tensor_to_txt(tf.image.resize(input_image_rgba, CANVAS_SHAPE))
    return render_template("index.html", result=result, input_image=input_image_txt)


if __name__ == '__main__':
    app.debug = True
    # app.run()
    app.run(debug=True)
