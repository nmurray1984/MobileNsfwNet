import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import glob
import uuid
import os

parser = argparse.ArgumentParser('Inference on trained model')

parser.add_argument('--model',
                    help='model trained on top of MobileNetV2')

parser.add_argument('--scan_folder',
                    help='folder to scan for images')

parser.add_argument('--output_folder',
                    help='folder to output flagged images')

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

def predict_img(img):
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)
    image_array = np.vstack([x])
    classes = model.predict(image_array)
    print(classes)
    return classes[0]

files = glob.glob(args.scan_folder + "*.jpg")

print('Found {} files to scan from directory {}'.format(len(files), args.scan_folder))

save_directory = os.path.join(args.scan_folder, 'positives')
if not os.path.exists(save_directory):
    os.mkdir(save_directory)

scanned_count = 0
flagged_count = 0
error_count = 0
for filename in files:
    try:
        img = image.load_img(filename, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    except OSError:
        #file is not formed correctly or can't be read by PIL
        os.remove(filename)
        error_count += 1
        continue
    scanned_count += 1
    result = predict_img(img)
    if result[1]> .50 or result[2] > .50:
        filename = "{}.jpg".format(uuid.uuid1())
        save_filepath = os.path.join(save_directory, filename)
        image.save_img(save_filepath, img)
        flagged_count += 1

print("Scanned {} images".format(scanned_count))
print("Flagged {} images".format(flagged_count))
print("Errors: {} images".format(error_count))