# -*- coding: utf-8 -*-
"""Mobilenet-Transfer language

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19f6OISNazfmsgoc9MQJ1Uuc056YiicbD
"""

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import imagenet_utils

from IPython.display import Image

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')

print("Num GPUs Available: ", len(physical_devices))

from google.colab import drive

drive.mount('/content/gdrive', force_remount=True)

model = tf.keras.applications.MobileNet()

model.summary()

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    print(img_array.shape)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    print(img_array_expanded_dims.shape)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

!/content/942d3a0ac09ec9e5eb3a-238f720ff059c1f82f368259d1ca4ffa5dd8f9f5.zip

Image(filename='input_image-1.png', width=300,height=200)

preprocessed_image = prepare_image('input_image-1.png')
predictions = model.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)
results

Image(filename='python.jpeg', width=300,height=200)

preprocessed_image = prepare_image('python.jpeg')
predictions = model.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)
results

