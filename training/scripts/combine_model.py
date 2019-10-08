from tensorflow.keras.models import load_model
import tensorflow as tf

top_model = load_model('training/output.h5')

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')

model = tf.keras.Sequential([
  base_model,
  top_model
])

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

def predict_img(file_name):
    img = image.load_img(file_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    image_array = np.vstack([x])
    classes = model.predict(image_array)
    if int(classes[0][0]) == 1:
        print("{},{}".format(file_name, int(classes[0][0])))

import glob

files = glob.glob('/Users/nathanmurray/Downloads/e1a1dd3c-3dff-4c12-8639-69269926f49d/*')
print(len(files))

for file_name in files:
    predict_img(file_name)
