from flask import Flask
from flask.templating import render_template
from flask import request
import os
import pickle
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models')

model_sgd = pickle.load(open(os.path.join(
    MODEL_PATH, 'dsa_image_classification_sgd.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_PATH, 'dsa_scaler.pkl'), 'rb'))


@app.errorhandler(404)
def error404(error):
    return render_template('error.html')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename
        if filename.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
            path_to_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_to_save)
            results = pipeline_model(path_to_save, scaler, model_sgd)
            return render_template('upload.html', extension=False, fileupload=True, data=results, image_filename=filename)

        else:
            return render_template('upload.html', extension=True)
    return render_template('upload.html', extension=False, fileupload=False)


@app.route('/about')
def about():
    return render_template('about.html')


def pipeline_model(path, scaler_transform, model_sgd):
    image = skimage.io.imread(path)
    image_resize = skimage.transform.resize(image, (80, 80))
    image_scale = 255 * image_resize
    image_transform = image_scale.astype(np.uint8)

    gray = skimage.color.rgb2gray(image_transform)
    feature_vector = skimage.feature.hog(
        gray, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

    scaleX = scaler_transform.transform(feature_vector.reshape(1, -1))
    result = model_sgd.predict(scaleX)

    decision = model_sgd.decision_function(scaleX).flatten()
    labels = model_sgd.classes_

    z = scipy.stats.zscore(decision)
    prob_value = scipy.special.softmax(z)

    top_five_prob_index = prob_value.argsort()[::-1][:5]

    top_labels = model_sgd.classes_[top_five_prob_index]
    top_probs = prob_value[top_five_prob_index]

    top_dict = dict()
    for key, value in zip(top_labels, top_probs):
        top_dict[key] = np.round(value, 2)

    return top_dict


if __name__ == "__main__":
    app.run(debug=True)
