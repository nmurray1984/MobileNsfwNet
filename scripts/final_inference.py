from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse

parser = argparse.ArgumentParser('Inference on trained model')

parser.add_argument('--model',
                    help='model trained on top of MobileNetV2')

parser.add_argument('--scan_folder',
                    help='folder to scan for images')

args = parser.parse_args()

def prediction_min(y_true, y_pred):
    final = K.min(y_pred)
    return final

def prediction_max(y_true, y_pred):
    final = K.max(y_pred)
    return final

def prediction_variance(y_true, y_pred):
    final = K.var(y_pred)
    return final

model = load_model(args.model, compile=False)

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def predict_img(file_name):
    img = image.load_img(file_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    image_array = np.vstack([x])
    print(image_array.shape)
    classes = model.predict(image_array)
    print(classes)
    if int(classes[0][0]) == 1:
        print("{},{}".format(file_name, int(classes[0][0])))

import glob

files = glob.glob('/Users/nathanmurray/Downloads/e1a1dd3')
print(len(files))

#for file_name in files:
#    predict_img(file_name)

predict_img('/Users/nathanmurray/source/MobileNSFW/flower_photos/roses/7683456068_02644b8382_m.jpg')
predict_img('/Users/nathanmurray/source/MobileNSFW/flower_photos/sunflowers/6953297_8576bf4ea3.jpg')
predict_img('/Users/nathanmurray/source/MobileNSFW/flower_photos/sunflowers/23204123212_ef32fbafbe.jpg')
