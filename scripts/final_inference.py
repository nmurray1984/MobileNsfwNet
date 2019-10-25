
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser('Inference on trained model')

parser.add_argument('--model',
                    help='model trained on top of MobileNetV2')

parser.add_argument('--base_dir',
                    help='folder to scan for images')

args = parser.parse_args()

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 100

def predict_img(file_name):
    img = image.load_img(file_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)
    image_array = np.vstack([x])
    classes = model.predict(image_array)
    print("{},{},{},{}".format(file_name, classes[0][0], classes[0][1], classes[0][2]))

model = load_model(args.model, compile=False)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

files = glob.glob(args.base_dir + '*/*.jpg')
print(len(files))

num_of_batches = int(len(files) / BATCH_SIZE) + 1 

for batch in range(0, num_of_batches):
    start = batch * BATCH_SIZE
    end = start + BATCH_SIZE - 1
    batch_files = files[start:end]
    images = np.empty([BATCH_SIZE, 224, 224, 3])
    i = 0
    for file_name in batch_files:
        img = image.load_img(file_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        x = image.img_to_array(img) / 255.
        images[i] = x
        i += 1
    classes = model.predict(images)
    
    for i in range(0, len(batch_files)):
        print('{},{},{},{}'.format(batch_files[i], classes[i][0], classes[i][1], classes[i][2]))