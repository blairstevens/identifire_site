import cv2
from keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import argparse
import imutils
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=["GET","POST"])
def index():
  if request.method == "POST":
    image_path = request.files["image"]
    if image_path and allowed_file(image_path.filename):
      filename = secure_filename(image_path.filename)
      image_path.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      return redirect(url_for("prediction", image_path=image_path.filename))
  else:
    return render_template('index.html')

@app.route('/prediction/<image_path>')
def prediction(image_path):
  image = cv2.imread('uploads/'+image_path)
  output = image.copy()
  output = imutils.resize(output, width=400)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (224, 224))

  # convert the image to a floating point data type and perform mean
  # subtraction
  image = image.astype("float32")
  mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
  image -= mean

  # load the trained model from disk
  print("[INFO] loading model...................................................................................")
  model = load_model('trained10.model')

  # pass the image through the network to obtain our predictions
  preds = model.predict(np.expand_dims(image, axis=0))[0]
  i = np.argmax(preds)
  classes=['ak_47', 'fal_rifle', 'm16_platform', 'steyr_aug']

  label = classes[i]
  return label


if __name__ == "__main__":
  app.run()
