import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import glob
import os
import requests

parser = argparse.ArgumentParser('Inference on trained model')
parser.add_argument('--model', help='model trained on top of MobileNetV2')
parser.add_argument('--url', help='image url to download')
args = parser.parse_args()

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

model = None
model_name = ""
if args.model is not None:
    model = load_model(args.model, compile=False)
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    model_name = os.path.basename(args.model).replace(".h5", "")
else:
    model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')
    model_name = "mobilenet_v2_1.0_224"

#download image
response = requests.get(args.url)
full_size_image = tf.io.decode_image(response.content)
resized = tf.image.resize(full_size_image, [IMAGE_SIZE, IMAGE_SIZE], tf.image.ResizeMethod.BILINEAR)
#casted = tf.cast(resized, tf.float32)
offset = tf.constant(255, dtype=tf.float32)
normalized = tf.math.divide(resized, offset)
batched = tf.reshape(normalized, [1, IMAGE_SIZE, IMAGE_SIZE, 3])
result = model.predict(batched)[0]
print(result)




#load and inference image