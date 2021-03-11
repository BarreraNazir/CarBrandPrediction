# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:29:20 2021

@author: 97155
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
# Keras

from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'
# Load your trained model
model = load_model(MODEL_PATH)
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)

