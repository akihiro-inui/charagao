#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26th July on the flight to Dublin
@author: Akihiro Inui
"""

import os
import base64
import shutil
from PIL import Image
from io import BytesIO
from werkzeug import secure_filename
from flask import Flask, render_template, request, redirect, url_for
from src.image_similarity_evaluation import ImageSimilarityEvaluation

# Init Flask app
app = Flask(__name__)
app.config.update(
    STATIC_FOLDER='../static',
    APPLICATION_ROOT='../../',
    UPLOAD_FOLDER='dataset/tmp',
    REFERENCE_FOLDER='dataset/train',
    TARGET_FILE='',
    ALLOWED_EXTENSIONS=set(['png', 'jpg', 'jpeg', 'PNG', 'JPG']),
    SECRET_KEY=os.urandom(24),
    BOOTSTRAP_SERVE_LOCAL=True,
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=20,
    DROPZONE_UPLOAD_ON_CLICK=True
)

# Init for Image Similarity Evaluation
app.ISE = ImageSimilarityEvaluation(setting_file='config/master_config.ini')
model_file_path = 'model/inada.model'


def routine():
    # Remove folder and recreate it
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.mkdir(app.config['UPLOAD_FOLDER'])
    print("Image files removed")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    # routine()
    return render_template('index.html')


@app.route('/send', methods=['GET', 'POST'])
def send():
    # Get image data if it uses POST method
    if request.method == 'GET' or 'POST':
        img_file = request.files['img_file']

        # Get image file path if file extension is good
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            app.config['TARGET_FILE'] = secure_filename(filename)
        else:
            return ''' <p>This file format is not currently supported</p> '''

        # Save image file into tmp directory
        if not os.path.isdir(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])

        # Save target image file into tmp folder
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['TARGET_FILE'])
        img_file.save(original_file_path)

        # Convert image data to binary
        image = Image.open(img_file)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        input_image_data = base64.b64encode(buffer.getvalue()).decode().replace("'", "")

        # Run evaluation script
        similar_image_paths = app.ISE.evaluation(image_file_path=app.config['TARGET_FILE'], model_file_path=model_file_path)
        similar_image_path = similar_image_paths[0]
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], app.config['TARGET_FILE']))
        return render_template('result.html', input_image_data=input_image_data, similar_image_path=similar_image_path)

    else:
        return redirect(url_for('index'))


@app.route('/retry', methods=['GET', 'POST'])
def retry():
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Run routine every ten mins
    app.jinja_env.cache = {}
    app.run(debug=True, host='0.0.0.0', port=5000)
