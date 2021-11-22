from flask import Flask
from flask import render_template,redirect,url_for,request
app = Flask(__name__)

from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("static/best_model.hdf5")

def preprocess_image(image_txt):
    image_txt2 = image_txt.split(",")[-1]
    decoded_img = base64.b64decode(bytes(image_txt2,'utf-8'))
    img = Image.open(BytesIO(decoded_img))
    img = np.array(img).max(axis=2)
    img = cv2.resize(img, (28,28), interpolation= cv2.INTER_LINEAR)
    img = img / (img.max() + 10**-8)
    return img

@app.route("/",methods=["GET","POST"])
def home():
    # print("\n",dict(request.form)["image"],"\n")
    # print(image_txt)

    # image_txt = dict(request.form)["image"]
    # # print(image_txt)
    # preprocessed_image = preprocess_image(image_txt)
    # input_image = preprocessed_image[np.newaxis,...,np.newaxis]
    # result = model.predict(input_image).argmax(axis=1)[0]
    # return render_template("index.html",res=result)
    return render_template("index.html", res="")


if __name__ == '__main__':
    app.debug = True
    app.run()
    app.run(debug = True)
